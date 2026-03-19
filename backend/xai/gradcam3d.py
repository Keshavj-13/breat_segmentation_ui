"""
3D Grad-CAM for volumetric medical image segmentation models.

Hooks into the last convolutional layer of a 3D U-Net (or similar CNN),
captures feature maps and gradients during a backward pass, then generates
a 3D class-activation heatmap aligned to the original MRI volume.
"""

import torch
import torch.nn.functional as F
import numpy as np


class GradCAM3D:
    """
    Compute 3D Grad-CAM heatmaps for volumetric segmentation networks.

    Usage:
        gcam = GradCAM3D(model, target_layer)
        heatmap = gcam.generate(input_volume)   # returns np.ndarray [D,H,W] in [0,1]
        gcam.remove_hooks()
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: A 3D segmentation network (e.g. AttentionAllDecoderUNet).
            target_layer: nn.Module whose output feature maps will be used.
                          If None, the last conv3d layer in the encoder is used.
        """
        self.model = model
        self.target_layer = target_layer or self._find_last_conv(model)
        self.feature_maps = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------
    def _register_hooks(self):
        def forward_hook(_module, _input, output):
            self.feature_maps = output.detach()

        def backward_hook(_module, _grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.feature_maps = None
        self.gradients = None

    # ------------------------------------------------------------------
    # Layer auto-detection
    # ------------------------------------------------------------------
    @staticmethod
    def _find_last_conv(model):
        """Walk the model tree and return the last Conv3d layer found."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv3d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv3d layer found in model")
        return last_conv

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def generate(self, input_tensor, class_idx=0):
        """
        Generate a 3D Grad-CAM heatmap.

        Args:
            input_tensor: torch.Tensor of shape [1, C, D, H, W] on the model device.
            class_idx: For single-class segmentation use 0.  For multi-class,
                       specify the target channel index.

        Returns:
            heatmap_np: np.ndarray [D, H, W] with values in [0, 1], same spatial
                        size as the *input_tensor* (trilinear upsampled).
        """
        self.model.zero_grad()

        # Ensure input requires grad for backward pass
        input_tensor = input_tensor.detach().requires_grad_(True)

        output = self.model(input_tensor)  # [1, C_out, D', H', W']

        # Select target class channel
        if output.shape[1] > 1:
            target = output[:, class_idx]
        else:
            target = output[:, 0]

        # Tumour score: mean of sigmoid-activated voxels inside the prediction
        score = torch.sigmoid(target).mean()
        score.backward(retain_graph=False)

        # Free the computation graph and activations immediately to release VRAM
        del output, target, score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.gradients is None or self.feature_maps is None:
            raise RuntimeError("Hooks did not capture data. Check target_layer.")

        # Global-average-pool the gradients → channel weights  [1, C_feat, 1, 1, 1]
        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)

        # Weighted combination of feature maps
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)  # [1,1,d,h,w]

        # ReLU – only keep positive influence
        cam = F.relu(cam)

        # Upsample to original spatial size
        original_size = input_tensor.shape[2:]  # (D, H, W)
        cam = F.interpolate(cam, size=original_size, mode="trilinear", align_corners=False)

        # Convert to numpy & normalise to [0, 1]
        cam_np = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        del cam, weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def generate_slicewise(self, input_tensor, class_idx=0):
        """
        Convenience: returns the full 3D heatmap plus per-axis best slices.

        Returns:
            dict with keys:
                heatmap_3d  – np.ndarray [D,H,W]
                axial       – 2D slice at z with max activation
                coronal     – 2D slice at y with max activation
                sagittal    – 2D slice at x with max activation
                best_indices – dict {z, y, x}
        """
        heatmap = self.generate(input_tensor, class_idx=class_idx)

        best_z = int(heatmap.mean(axis=(1, 2)).argmax())
        best_y = int(heatmap.mean(axis=(0, 2)).argmax())
        best_x = int(heatmap.mean(axis=(0, 1)).argmax())

        return {
            "heatmap_3d": heatmap,
            "axial": heatmap[best_z],
            "coronal": heatmap[:, best_y, :],
            "sagittal": heatmap[:, :, best_x],
            "best_indices": {"z": best_z, "y": best_y, "x": best_x},
        }

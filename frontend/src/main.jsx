import React from "react";
import ReactDOM from "react-dom/client";
import "./styles.css";
import "./xai.css";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import XaiApp from "./XaiApp.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <XaiApp />
    </ErrorBoundary>
  </React.StrictMode>
);

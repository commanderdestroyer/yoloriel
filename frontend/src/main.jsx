import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render( // note: bắt đầu tạo react rôt để react bắt đầu quản lý UI
  <StrictMode>
    <App />
  </StrictMode>
);
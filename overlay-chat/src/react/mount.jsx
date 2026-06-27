import React from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";

export const mountReactWindow = (component) => {
  const rootElement = document.getElementById("root");
  if (!rootElement) {
    throw new Error("React root element not found.");
  }

  const root = createRoot(rootElement);
  flushSync(() => {
    root.render(component);
  });
};

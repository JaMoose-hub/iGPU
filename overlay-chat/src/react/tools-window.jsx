import React from "react";
import { mountReactWindow } from "./mount.jsx";
import { ToolsWindow } from "./windows.jsx";

mountReactWindow(<ToolsWindow />);
await import("../tools.js");

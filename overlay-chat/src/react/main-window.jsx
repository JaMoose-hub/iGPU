import React from "react";
import { mountReactWindow } from "./mount.jsx";
import { MainWindow } from "./windows.jsx";

mountReactWindow(<MainWindow />);
await import("../main.js?v=react-shell-20260616");

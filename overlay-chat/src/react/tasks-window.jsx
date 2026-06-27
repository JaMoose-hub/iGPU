import React from "react";
import { mountReactWindow } from "./mount.jsx";
import { TasksWindow } from "./windows.jsx";

mountReactWindow(<TasksWindow />);
await import("../tasks.js");

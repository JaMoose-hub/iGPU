import React from "react";
import { mountReactWindow } from "./mount.jsx";
import { StandbyWindow } from "./windows.jsx";

mountReactWindow(<StandbyWindow />);
await import("../standby.js");

import React from "react";
import { mountReactWindow } from "./mount.jsx";
import { HudWindow } from "./windows.jsx";

mountReactWindow(<HudWindow />);
await import("../hud.js");

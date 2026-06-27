import React from "react";
import { mountReactWindow } from "./mount.jsx";
import { GamePathWindow } from "./windows.jsx";

mountReactWindow(<GamePathWindow />);
await import("../gamepath.js");

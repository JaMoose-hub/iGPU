import React from "react";
import { mountReactWindow } from "./mount.jsx";
import { SearchWindow } from "./windows.jsx";

mountReactWindow(<SearchWindow />);
await import("../search.js");

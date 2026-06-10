Game Companion Hybrid Dev - Hermes option B

This package bundles the local Game Companion app, Python runtime, llama.cpp Vulkan, and Qwen3.5-4B-Q4_K_M.gguf for the local iGPU router.

Hermes itself is not bundled. Each developer should install Hermes in their own environment, then edit:

  config\game_companion.env

Default bridge mode:

  HERMES_WSL_DISTRO=Ubuntu-24.04
  HERMES_USE_CONFIG_MODEL=1
  HERMES_AGENT_WEB_ENABLED=1
  HERMES_AGENT_TOOLSETS=web

Expected runtime layout after install:

  Start-GameCompanion-HybridDev.cmd
    starts local Qwen router on 127.0.0.1:18081 with Vulkan0/iGPU
    starts Game Companion backend on 127.0.0.1:8000
    starts the overlay GUI

  Stop-GameCompanion-HybridDev.cmd
    stops Game Companion backend, overlay, and local router
    does not stop external Hermes

  Diagnose-GameCompanion-HybridDev.cmd
    checks files, Vulkan devices, ports, and backend health

No API keys or Hermes secrets are stored in this installer.

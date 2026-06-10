Game Companion Hybrid Dev Offline Installer

Included:
- Game Companion backend and overlay GUI
- Python runtime
- llama.cpp Vulkan tools
- Qwen3.5-4B-Q4_K_M.gguf local router model
- GamePath RAG Lite code and documentation
- Hermes option B configuration template

Not included:
- Hermes installation
- Tavily/OpenAI/API secrets
- Existing gamepath.sqlite, logs, or user memory

Install:
1. Run Install-GameCompanion-HybridDev.cmd
2. Edit config\game_companion.env if your Hermes WSL distro or endpoint is different
3. Run Start-GameCompanion-HybridDev.cmd

Default local router device is Vulkan0. On the current development machine this is the Intel iGPU.
Use Diagnose-GameCompanion-HybridDev.cmd to confirm Vulkan device ordering on another machine.

Developer handoff and troubleshooting checklist:
docs\hybrid_dev_handoff.md

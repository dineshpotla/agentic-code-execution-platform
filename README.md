# Code-Executing LLM Platform

End-to-end starter project for an LLM that writes and executes Python in a locked-down Docker sandbox.

## What You Get
- FastAPI backend with an `/api/analyze` endpoint
- Docker sandbox with strict limits (no network, CPU/memory capped)
- Simple web UI for file upload + query
- Tool-calling loop: LLM writes code → sandbox executes → LLM explains results

## Prerequisites
- Python 3.10+
- Docker Desktop

## Setup
1. Build the sandbox image:
   ```bash
   bash scripts/build_sandbox_image.sh
   ```
2. Create a virtual environment and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```
3. Create `backend/.env` with your keys:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=YOUR_KEY
   OPENAI_MODEL=gpt-4o-mini
   GEMINI_API_KEY=YOUR_KEY
   GEMINI_MODEL=models/gemma-3-27b-it
   NVIDIA_API_KEY=YOUR_KEY
   NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
   NVIDIA_MODEL=qwen/qwen3.5-9b
   OLLAMA_BASE_URL=http://localhost:11434/v1
   OLLAMA_MODEL=qwen3.5:9b
   ```

   Provider selection:
   - `LLM_PROVIDER=openai` uses `OPENAI_API_KEY` + `OPENAI_MODEL`
   - `LLM_PROVIDER=gemini` uses `GEMINI_API_KEY` + `GEMINI_MODEL`
   - `LLM_PROVIDER=nvidia` uses `NVIDIA_API_KEY` + `NVIDIA_MODEL` via NVIDIA endpoints
   - `LLM_PROVIDER=ollama` uses local Ollama via `OLLAMA_BASE_URL` + `OLLAMA_MODEL`
   - Backward compatibility: legacy `NV_API_KEY`, `NV_API_URL`, and `NV_API_MODEL` are still accepted.

## Run
```bash
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Open `http://localhost:8000` and upload a CSV, then ask questions.

## Notes
- The sandbox image is `code-sandbox:py311`.
- Artifacts (e.g., charts) are served from `/artifacts/<run_id>/`.
- Code validation uses a whitelist of safe imports in `backend/app/security/validators.py`.
- With `LLM_PROVIDER=ollama`, reasoning tokens are streamed into live status updates (shown as `thinking: ...`).
- OpenAI-compatible flows now run with a supervisor+judge loop: execution errors are fed back to the agent for replanning, and answers are judged for completeness before returning.
- For missing dependencies, the agent attempts to add/install requirements before execution retries.

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
   OPENAI_API_KEY=YOUR_KEY
   OPENAI_MODEL=gpt-4o-mini
   NV_API_KEY=YOUR_KEY
   NV_API_MODEL=moonshotai/kimi-k2.5
   NV_API_URL=https://integrate.api.nvidia.com/v1/chat/completions
   ```

## Run
```bash
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Open `http://localhost:8000` and upload a CSV, then ask questions.

## Notes
- The sandbox image is `code-sandbox:py311`.
- Artifacts (e.g., charts) are served from `/artifacts/<run_id>/`.
- Code validation uses a whitelist of safe imports in `backend/app/security/validators.py`.

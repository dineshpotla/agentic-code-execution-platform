import os
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.orchestrator import AnalysisPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

load_dotenv(dotenv_path="/Users/dinesh/Downloads/.env", override=False)

app = FastAPI(title="Code-Executing LLM Platform")
pipeline = AnalysisPipeline()
run_status: dict[str, str] = {}
run_results: dict[str, dict] = {}
chat_memory: dict[str, list[dict]] = {}


def _build_memory_context(memory: list[dict]) -> dict:
    if not memory:
        return {"summary": "", "recent": []}
    recent = memory[-12:]
    earlier = memory[:-12]
    summary = ""
    if earlier:
        compact = " ".join(item.get("content", "") for item in earlier)
        summary = compact[:800]
    return {"summary": summary, "recent": recent}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


def _run_analysis(
    run_id: str,
    session_id: str,
    query: str,
    uploaded: dict,
    artifact_dir: Path,
) -> None:
    def status_cb(message: str) -> None:
        run_status[run_id] = message

    memory_all = chat_memory.get(session_id, [])
    memory = _build_memory_context(memory_all)
    run_status[run_id] = "starting"
    result = pipeline.process(
        user_query=query,
        uploaded_files=uploaded,
        artifact_dir=str(artifact_dir),
        status_cb=status_cb,
        memory=memory,
    )

    if not result.get("success"):
        run_results[run_id] = {"error": result.get("error", "Unknown error.")}
        run_status[run_id] = "error"
        return

    artifact_urls = []
    for path in result.get("artifacts", []):
        filename = Path(path).name
        artifact_urls.append(f"/artifacts/{run_id}/{filename}")

    response_payload = {
        "text": result.get("text", ""),
        "artifacts": artifact_urls,
        "raw_output": result.get("raw_output", ""),
    }
    run_results[run_id] = response_payload
    run_status[run_id] = "done"

    chat_memory.setdefault(session_id, []).append({"role": "user", "content": query})
    chat_memory[session_id].append({"role": "assistant", "content": response_payload["text"]})


@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    query: str = Form(...),
    session_id: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
    uploaded = {}
    for upload in files:
        content = await upload.read()
        uploaded[upload.filename] = content

    run_id = str(uuid.uuid4())
    artifact_dir = ARTIFACTS_DIR / run_id
    os.makedirs(artifact_dir, exist_ok=True)

    run_status[run_id] = "queued"
    background_tasks.add_task(_run_analysis, run_id, session_id, query, uploaded, artifact_dir)
    return {"run_id": run_id}


@app.get("/api/status/{run_id}")
def get_status(run_id: str):
    return {"status": run_status.get(run_id, "unknown")}


@app.get("/api/result/{run_id}")
def get_result(run_id: str):
    if run_id not in run_results:
        return JSONResponse({"status": "running"}, status_code=202)
    result = run_results[run_id]
    if "error" in result:
        return JSONResponse({"error": result["error"]}, status_code=400)
    return result


@app.get("/api/provider")
def get_provider():
    return {
        "provider": pipeline._provider(),
        "nv_api_key_present": bool(os.getenv("NV_API_KEY")),
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "nv_api_model": os.getenv("NV_API_MODEL"),
        "nv_api_url": os.getenv("NV_API_URL"),
    }


@app.post("/api/memory/clear")
def clear_memory(session_id: str = Form(...)):
    chat_memory.pop(session_id, None)
    return {"status": "cleared"}

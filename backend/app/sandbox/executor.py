import os
import shutil
import uuid
from typing import Any, Optional

import docker
from docker.errors import ContainerError, ImageNotFound


class CodeSandbox:
    def __init__(self, image: str = "code-sandbox:py311") -> None:
        self.client = docker.from_env()
        self.image = image

    def _write_file(self, path: str, content: Any) -> None:
        if isinstance(content, bytes):
            with open(path, "wb") as handle:
                handle.write(content)
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(content))

    def execute(
        self,
        code: str,
        files: dict[str, Any],
        timeout_seconds: int = 30,
        artifact_dir: Optional[str] = None,
        requirements: Optional[list[str]] = None,
    ) -> dict:
        container_id = str(uuid.uuid4())
        host_path = f"/tmp/sandbox/{container_id}"
        os.makedirs(host_path, exist_ok=True)

        for fname, content in files.items():
            self._write_file(os.path.join(host_path, fname), content)

        self._write_file(os.path.join(host_path, "script.py"), code)

        container = None
        try:
            if requirements:
                req_path = os.path.join(host_path, "requirements.txt")
                self._write_file(req_path, "\n".join(requirements))
                install_cmd = (
                    "python -m venv /sandbox/.venv && "
                    "/sandbox/.venv/bin/pip install --no-cache-dir -r /sandbox/requirements.txt"
                )
                install_container = self.client.containers.run(
                    self.image,
                    command=install_cmd,
                    volumes={host_path: {"bind": "/sandbox", "mode": "rw"}},
                    environment={"MPLCONFIGDIR": "/sandbox/.mplconfig"},
                    mem_limit="512m",
                    cpu_quota=100000,
                    network_disabled=False,
                    pids_limit=256,
                    security_opt=["no-new-privileges"],
                    remove=True,
                    detach=True,
                )
                install_result = install_container.wait(timeout=timeout_seconds)
                install_logs = install_container.logs().decode("utf-8", errors="replace")
                if install_result.get("StatusCode", 1) != 0:
                    return {"success": False, "error": install_logs or "Dependency install failed."}

            exec_cmd = (
                "/sandbox/.venv/bin/python /sandbox/script.py"
                if requirements
                else "python /sandbox/script.py"
            )
            container = self.client.containers.run(
                self.image,
                command=exec_cmd,
                volumes={host_path: {"bind": "/sandbox", "mode": "rw"}},
                environment={"MPLCONFIGDIR": "/sandbox/.mplconfig"},
                mem_limit="512m",
                cpu_quota=100000,
                network_disabled=True,
                pids_limit=256,
                security_opt=["no-new-privileges"],
                read_only=True,
                remove=False,
                detach=True,
            )
            result = container.wait(timeout=timeout_seconds)
            logs = container.logs().decode("utf-8", errors="replace")

            artifacts = []
            for filename in os.listdir(host_path):
                if filename not in files and filename != "script.py":
                    artifact_path = os.path.join(host_path, filename)
                    if artifact_dir:
                        os.makedirs(artifact_dir, exist_ok=True)
                        dest_path = os.path.join(artifact_dir, filename)
                        shutil.copyfile(artifact_path, dest_path)
                        artifacts.append(dest_path)
                    else:
                        artifacts.append(artifact_path)

            status_code = result.get("StatusCode", 1)
            payload = {
                "success": status_code == 0,
                "output": logs,
                "artifacts": artifacts,
            }
            if status_code != 0:
                payload["error"] = logs or f"Process exited with status {status_code}."
            return payload
        except ImageNotFound as exc:
            return {
                "success": False,
                "error": (
                    f"Docker image '{self.image}' not found. "
                    "Build it using scripts/build_sandbox_image.sh."
                ),
            }
        except ContainerError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": str(exc)}
        finally:
            if container is not None:
                try:
                    container.kill()
                except Exception:
                    pass
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            shutil.rmtree(host_path, ignore_errors=True)

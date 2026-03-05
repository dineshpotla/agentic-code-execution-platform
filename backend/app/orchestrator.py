import ast
import json
import os
import re
import time
from typing import Any, Optional

from google import genai
from google.genai import errors as genai_errors
from openai import OpenAI
import requests

from app.llm.tools import tools
from app.sandbox.executor import CodeSandbox
from app.security.validators import CodeValidationError, DEFAULT_ALLOWED_MODULES, validate_python_code


class AnalysisPipeline:
    _PREINSTALLED_MODULES = (
        "pandas, numpy, sklearn, scipy, seaborn, statsmodels, matplotlib, "
        "json, math, statistics"
    )
    _AUTO_REQUIREMENTS_BLOCKLIST = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "glob",
        "importlib",
        "inspect",
        "builtins",
        "typing",
        "tempfile",
        "ctypes",
        "multiprocessing",
        "threading",
        "signal",
        "resource",
        "pwd",
        "grp",
    }

    def _extract_missing_module(self, error: str) -> Optional[str]:
        match = re.search(r"No module named '([^']+)'", error)
        if match:
            return match.group(1)
        return None

    def _extract_disallowed_import(self, error: str) -> Optional[str]:
        match = re.search(r"Import '([^']+)' not allowed\.", error)
        if match:
            return match.group(1)
        return None

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        if text.strip():
            return {"tool": "none", "response": text.strip()}
        return {"tool": "error", "error": "Model returned empty response."}

    def _normalize_openai_compatible_base_url(self, raw_url: str) -> str:
        url = raw_url.strip().rstrip("/")
        if url.endswith("/chat/completions"):
            return url[: -len("/chat/completions")]
        return url

    def __init__(self, sandbox: Optional[CodeSandbox] = None) -> None:
        self.sandbox = sandbox or CodeSandbox()
        self.provider_override = os.getenv("LLM_PROVIDER", "").strip().lower()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.nvidia_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NV_API_KEY")
        self.nvidia_base_url = self._normalize_openai_compatible_base_url(
            os.getenv("NVIDIA_BASE_URL")
            or os.getenv("NV_API_URL")
            or "https://integrate.api.nvidia.com/v1"
        )
        self.ollama_base_url = self._normalize_openai_compatible_base_url(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        )
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        self.ollama_configured = bool(
            os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_API_KEY")
        )
        self.openai_client = OpenAI(api_key=self.openai_key) if self.openai_key else None
        self.nvidia_client = (
            OpenAI(api_key=self.nvidia_key, base_url=self.nvidia_base_url)
            if self.nvidia_key
            else None
        )
        self.ollama_client = OpenAI(api_key=self.ollama_api_key, base_url=self.ollama_base_url)
        self.gemini_client = genai.Client(api_key=self.gemini_key) if self.gemini_key else None
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.gemini_model = os.getenv("GEMINI_MODEL", "models/gemma-3-27b-it")
        self.nvidia_model = (
            os.getenv("NVIDIA_MODEL") or os.getenv("NV_API_MODEL") or "qwen/qwen3.5-9b"
        )
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

    def _system_prompt(self) -> str:
        return (
            "You are a data analysis assistant. When users upload files:\n"
            "1. Analyze their request.\n"
            "2. Decide and execute the required analysis yourself using execute_python.\n"
            "3. Write code assuming files are in current directory "
            "(e.g., pd.read_csv('data.csv')).\n"
            "4. Save charts using plt.savefig('chart.png') so they can be returned.\n"
            "5. Do not import matplotlib or create charts unless the user asks for a visualization.\n"
            "6. Keep code concise and focused on the request.\n"
            "7. Always inspect available columns first; if a requested column is missing, "
            "print df.columns and provide a helpful message instead of failing.\n"
            "8. If train/test schemas differ, align feature columns and "
            "handle missing targets by using available proxy targets or "
            "skip validation with a clear note.\n"
            "9. When a column mentioned by the user is missing, infer a likely "
            "alternative based on similar column names (case-insensitive) and "
            "explain the mapping.\n"
            "10. For date/time analysis, detect existing date-like columns; "
            "if none exist but year/month/day are present, construct a date.\n"
            "11. When using non-standard libraries, include a requirements list "
            "so dependencies can be installed before execution.\n"
            "12. Use only these available packages unless you list them in "
            "requirements: "
            f"{self._PREINSTALLED_MODULES}.\n"
            "13. Final responses should be concise and directly answer the user's question. "
            "Avoid repeating intermediate logs or unnecessary details.\n"
            "14. Before finalizing, verify the answer satisfies the user's question. "
            "If not, update the plan and continue.\n"
            "15. Output format:\n"
            "<exact answer for user question very concise>\n"
            "<optional insights only if necessary>\n"
            "16. Use deep thinking for complex tasks before answering.\n"
        )

    def _output_prompt(self, user_query: str, outputs: list[str]) -> str:
        return (
            "Return only:\n"
            "<pin point answer for user original question>\n\n"
            f"Question: {user_query}\n"
            f"Outputs: {outputs}\n"
        )

    def _requirements_to_allowed(self, requirements: list[str]) -> set[str]:
        allowed = set(DEFAULT_ALLOWED_MODULES)
        for req in requirements:
            name = req.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0]
            name = name.strip().replace("-", "_")
            if name:
                allowed.add(name.split(".")[0])
        return allowed

    def _normalize_requirement_name(self, requirement: str) -> str:
        name = requirement.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0]
        return name.strip().replace("-", "_").split(".")[0]

    def _dedupe_requirements(self, requirements: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for req in requirements:
            norm = self._normalize_requirement_name(req)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(req)
        return deduped

    def _extract_import_roots(self, code: str) -> list[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        roots: list[str] = []
        seen: set[str] = set()
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0].replace("-", "_").strip()
                if root and root not in seen:
                    seen.add(root)
                    roots.append(root)
        return roots

    def _is_safe_dependency_candidate(self, module_name: str) -> bool:
        root = module_name.split(".")[0].replace("-", "_").strip().lower()
        if not root:
            return False
        if not re.fullmatch(r"[a-zA-Z0-9_]{1,64}", root):
            return False
        if root in self._AUTO_REQUIREMENTS_BLOCKLIST:
            return False
        return True

    def _augment_requirements_from_code(
        self, code: str, requirements: list[str]
    ) -> tuple[list[str], list[str]]:
        reqs = self._dedupe_requirements(requirements)
        known = {self._normalize_requirement_name(r) for r in reqs}
        auto_added: list[str] = []
        for root in self._extract_import_roots(code):
            if root in DEFAULT_ALLOWED_MODULES or root in known:
                continue
            if self._is_safe_dependency_candidate(root):
                reqs.append(root)
                known.add(root)
                auto_added.append(root)
        return reqs, auto_added

    def _judge_prompt_openai_compatible(
        self, user_query: str, candidate_answer: str, outputs: list[str], memory: dict | None = None
    ) -> str:
        memory = memory or {"summary": "", "recent": []}
        return (
            "You are a strict judge supervising a data-analysis agent.\n"
            "Decide if the candidate answer fully satisfies the user request.\n"
            "Return JSON only:\n"
            '{"done":true,"reason":"...","missing":"..."}\n'
            f"Conversation memory: {memory}\n"
            f"User request: {user_query}\n"
            f"Candidate answer: {candidate_answer}\n"
            f"Execution outputs: {outputs}\n"
        )

    def _judge_openai_compatible_answer(
        self,
        llm_client: OpenAI,
        llm_model: str,
        provider: str,
        user_query: str,
        candidate_answer: str,
        outputs: list[str],
        memory: dict | None = None,
        status_cb: Optional[callable] = None,
    ) -> dict:
        judge_payload = self._call_openai_compatible_message(
            llm_client=llm_client,
            llm_model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON-only evaluator. "
                        "Always return valid JSON and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": self._judge_prompt_openai_compatible(
                        user_query=user_query,
                        candidate_answer=candidate_answer,
                        outputs=outputs,
                        memory=memory,
                    ),
                },
            ],
            status_cb=status_cb if provider == "ollama" else None,
        )
        if judge_payload.get("error"):
            return {
                "done": False,
                "reason": judge_payload["error"],
                "missing": "Judge call failed.",
            }

        parsed = self._extract_json(judge_payload.get("content", ""))
        if not isinstance(parsed, dict):
            return {
                "done": False,
                "reason": "Invalid judge response.",
                "missing": "Judge response was not JSON.",
            }
        return {
            "done": parsed.get("done") is True,
            "reason": str(parsed.get("reason", "")),
            "missing": str(parsed.get("missing", "")),
        }

    def _provider(self) -> str:
        if self.provider_override in {"openai", "gemini", "nvidia", "ollama"}:
            return self.provider_override
        if self.gemini_key:
            return "gemini"
        if self.nvidia_key:
            return "nvidia"
        if self.openai_key:
            return "openai"
        if self.ollama_configured:
            return "ollama"
        return "openai"

    def _openai_compatible_client_and_model(self) -> tuple[Optional[OpenAI], str]:
        provider = self._provider()
        if provider == "nvidia":
            return self.nvidia_client, self.nvidia_model
        if provider == "ollama":
            return self.ollama_client, self.ollama_model
        return self.openai_client, self.openai_model

    def _gemini_tool_prompt(
        self, user_query: str, file_names: list[str], memory: dict | None = None
    ) -> str:
        memory = memory or {"summary": "", "recent": []}
        return (
            f"{self._system_prompt()}\n"
            "Return a single JSON object and nothing else.\n"
            "If you need to execute code, respond with:\n"
            '{"tool":"execute_python","code":"...","requirements":["pkg"],"reasoning":"..."}\n'
            "If no code is needed, respond with:\n"
            '{"tool":"none","response":"..."}\n'
            f"Conversation memory: {memory}\n"
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
        )

    def _gemini_plan_prompt(
        self, user_query: str, file_names: list[str], memory: dict | None = None
    ) -> str:
        memory = memory or {"summary": "", "recent": []}
        return (
            f"{self._system_prompt()}\n"
            "Create a multi-step analysis plan. Return JSON only with:\n"
            '{"steps":[{"id":1,"goal":"...","notes":"..."}]}\n'
            f"Conversation memory: {memory}\n"
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
        )

    def _gemini_step_prompt(
        self,
        user_query: str,
        file_names: list[str],
        step: dict,
        prior_outputs: list[str],
        memory: dict | None = None,
    ) -> str:
        memory = memory or {"summary": "", "recent": []}
        return (
            f"{self._system_prompt()}\n"
            "You are executing a step from a plan. Return JSON only.\n"
            "If code is needed:\n"
            '{"tool":"execute_python","code":"...","requirements":["pkg"],"reasoning":"..."}\n'
            "If no code is needed:\n"
            '{"tool":"none","response":"..."}\n'
            f"Conversation memory: {memory}\n"
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
            f"Step: {step}\n"
            f"Prior outputs: {prior_outputs}\n"
        )

    def _gemini_refine_prompt(
        self,
        user_query: str,
        file_names: list[str],
        step: dict,
        error: str,
        memory: dict | None = None,
    ) -> str:
        memory = memory or {"summary": "", "recent": []}
        return (
            f"{self._system_prompt()}\n"
            "The previous execution failed. Provide corrected code only.\n"
            '{"tool":"execute_python","code":"...","requirements":["pkg"],"reasoning":"..."}\n'
            f"Conversation memory: {memory}\n"
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
            f"Step: {step}\n"
            f"Error: {error}\n"
        )

    def _gemini_refine_prompt_with_attempt(
        self,
        user_query: str,
        file_names: list[str],
        step: dict,
        error: str,
        attempt: int,
        memory: dict | None = None,
    ) -> str:
        memory = memory or {"summary": "", "recent": []}
        approach_note = ""
        if attempt >= 10:
            approach_note = (
                "The same error persisted many times. Use a different approach "
                "or a simpler method to complete the task.\n"
            )
        missing_module = self._extract_missing_module(error) or ""
        missing_note = ""
        if missing_module:
            missing_note = (
                f"Missing module detected: {missing_module}. "
                "Avoid this import and use available packages instead.\n"
            )
        return (
            f"{self._system_prompt()}\n"
            "The previous execution failed. Provide corrected code only.\n"
            f"{approach_note}"
            f"{missing_note}"
            '{"tool":"execute_python","code":"...","requirements":["pkg"],"reasoning":"..."}\n'
            f"Conversation memory: {memory}\n"
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
            f"Step: {step}\n"
            f"Error: {error}\n"
            f"Attempt: {attempt}\n"
        )

    def _gemini_judge_prompt(
        self, user_query: str, plan: dict, outputs: list[str], memory: dict | None = None
    ) -> str:
        memory = memory or {"summary": "", "recent": []}
        return (
            f"{self._system_prompt()}\n"
            "Judge if the analysis fully answers the user's question. Return JSON only:\n"
            '{"done":true,"reason":"...","missing":"..."}\n'
            f"Conversation memory: {memory}\n"
            f"Request: {user_query}\n"
            f"Plan: {plan}\n"
            f"Outputs: {outputs}\n"
        )

    def _call_gemini_text(self, prompt: str) -> str:
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
            )
            return response.text or ""
        except genai_errors.ClientError as exc:
            return f"Gemini API error: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Gemini call failed: {exc}"

    def _compact_thinking(self, text: str, limit: int = 220) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return f"...{compact[-limit:]}"

    def _message_to_payload(self, message: Any) -> dict:
        payload = {"content": message.content or "", "tool_calls": []}
        if message.tool_calls:
            tool_calls = []
            for call in message.tool_calls:
                tool_calls.append(
                    {
                        "id": getattr(call, "id", ""),
                        "type": getattr(call, "type", "function"),
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                )
            payload["tool_calls"] = tool_calls
        return payload

    def _call_ollama_stream(
        self,
        messages: list[dict[str, Any]],
        tools_payload: list[dict] | None = None,
        tool_choice: str | None = None,
        status_cb: Optional[callable] = None,
    ) -> dict:
        url = f"{self.ollama_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"

        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": True,
        }
        if tools_payload is not None:
            payload["tools"] = tools_payload
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_index: dict[int, dict] = {}
        last_emit = 0.0

        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=300
            ) as response:
                response.raise_for_status()
                response.encoding = "utf-8"
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    delta = (choices[0] or {}).get("delta") or {}
                    reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
                        if status_cb:
                            now = time.monotonic()
                            if now - last_emit >= 0.35:
                                status_cb(f"thinking: {self._compact_thinking(''.join(reasoning_parts))}")
                                last_emit = now

                    content_delta = delta.get("content")
                    if isinstance(content_delta, str) and content_delta:
                        content_parts.append(content_delta)

                    for tool_call in delta.get("tool_calls") or []:
                        idx = tool_call.get("index", 0)
                        entry = tool_calls_by_index.setdefault(
                            idx,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tool_call.get("id"):
                            entry["id"] = tool_call["id"]
                        if tool_call.get("type"):
                            entry["type"] = tool_call["type"]
                        function_payload = tool_call.get("function") or {}
                        if function_payload.get("name"):
                            entry["function"]["name"] += function_payload["name"]
                        if function_payload.get("arguments"):
                            entry["function"]["arguments"] += function_payload["arguments"]

            if reasoning_parts and status_cb:
                status_cb(f"thinking: {self._compact_thinking(''.join(reasoning_parts))}")

            ordered_tool_calls = [
                tool_calls_by_index[i] for i in sorted(tool_calls_by_index.keys())
            ]
            return {
                "content": "".join(content_parts),
                "tool_calls": ordered_tool_calls,
            }
        except requests.RequestException as exc:
            return {"error": f"Ollama API error: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Ollama call failed: {exc}"}

    def _call_openai_compatible_message(
        self,
        llm_client: OpenAI,
        llm_model: str,
        messages: list[dict[str, Any]],
        tools_payload: list[dict] | None = None,
        tool_choice: str | None = None,
        status_cb: Optional[callable] = None,
    ) -> dict:
        provider = self._provider()
        if provider == "ollama":
            return self._call_ollama_stream(
                messages=messages,
                tools_payload=tools_payload,
                tool_choice=tool_choice,
                status_cb=status_cb,
            )

        request_payload: dict[str, Any] = {
            "model": llm_model,
            "messages": messages,
        }
        if tools_payload is not None:
            request_payload["tools"] = tools_payload
        if tool_choice is not None:
            request_payload["tool_choice"] = tool_choice

        response = llm_client.chat.completions.create(**request_payload)
        return self._message_to_payload(response.choices[0].message)

    def process(
        self,
        user_query: str,
        uploaded_files: dict[str, Any],
        artifact_dir: str,
        status_cb: Optional[callable] = None,
        memory: dict | None = None,
    ) -> dict:
        def status(message: str) -> None:
            if status_cb:
                status_cb(message)

        fast_response = "return only" in user_query.lower() or "no explanation" in user_query.lower()
        provider = self._provider()
        if provider == "gemini" and not self.gemini_key:
            return {"success": False, "error": "LLM_PROVIDER=gemini requires GEMINI_API_KEY."}
        if provider == "nvidia" and not self.nvidia_key:
            return {"success": False, "error": "LLM_PROVIDER=nvidia requires NVIDIA_API_KEY."}
        if provider == "openai" and not self.openai_key:
            return {"success": False, "error": "LLM_PROVIDER=openai requires OPENAI_API_KEY."}
        if provider == "ollama" and not self.ollama_model:
            return {"success": False, "error": "LLM_PROVIDER=ollama requires OLLAMA_MODEL."}
        if (
            not self.openai_key
            and not self.gemini_key
            and not self.nvidia_key
            and provider != "ollama"
        ):
            return {
                "success": False,
                "error": (
                    "Set OPENAI_API_KEY, GEMINI_API_KEY, or NVIDIA_API_KEY before running, "
                    "or use LLM_PROVIDER=ollama."
                ),
            }

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Files available: {list(uploaded_files.keys())}\n\n"
                    f"Request: {user_query}"
                ),
            },
        ]

        if provider == "gemini":
            file_names = list(uploaded_files.keys())
            status("planning analysis")
            plan_response = self._call_gemini_text(
                self._gemini_plan_prompt(user_query, file_names, memory)
            )
            plan = self._extract_json(plan_response)
            if not isinstance(plan, dict):
                plan = {}
            if plan.get("tool") == "error":
                return {"success": False, "error": plan.get("error", "Plan error.")}

            steps = plan.get("steps", [])
            if not isinstance(steps, list) or not steps:
                steps = [
                    {
                        "id": 1,
                        "goal": "Answer the user request directly",
                        "notes": "Fallback single-step plan",
                    }
                ]

            outputs: list[str] = []
            artifacts: list[str] = []
            max_steps = min(len(steps), 6)
            max_retries = 10

            for idx, step in enumerate(steps[:max_steps], start=1):
                goal = step.get("goal", "step")
                status(f"step {idx}/{max_steps}: {goal}")
                step_result = self._call_gemini_text(
                    self._gemini_step_prompt(user_query, file_names, step, outputs, memory)
                )
                action = self._extract_json(step_result)
                if action.get("tool") == "error":
                    return {"success": False, "error": action.get("error", "Step error.")}
                if action.get("tool") == "none":
                    outputs.append(action.get("response", ""))
                    continue

                code = action.get("code", "")
                requirements = action.get("requirements") or []
                try:
                    status("validating generated code")
                    validate_python_code(code, self._requirements_to_allowed(requirements))
                except CodeValidationError as exc:
                    return {"success": False, "error": str(exc)}

                status(f"executing code for step {idx}/{max_steps}")
                result = self.sandbox.execute(
                    code=code,
                    files=uploaded_files,
                    artifact_dir=artifact_dir,
                    requirements=requirements,
                )
                attempt = 0
                while not result.get("success") and attempt < max_retries:
                    attempt += 1
                    status(f"refining after error in step {idx}/{max_steps}")
                    refine = self._call_gemini_text(
                        self._gemini_refine_prompt_with_attempt(
                            user_query,
                            file_names,
                            step,
                            result.get("error", ""),
                            attempt,
                            memory,
                        )
                    )
                    refine_action = self._extract_json(refine)
                    if refine_action.get("tool") != "execute_python":
                        return {
                            "success": False,
                            "error": result.get("error", "Execution failed."),
                        }
                    code = refine_action.get("code", "")
                    requirements = refine_action.get("requirements") or requirements
                    try:
                        status("validating refined code")
                        validate_python_code(code, self._requirements_to_allowed(requirements))
                    except CodeValidationError as exc:
                        return {"success": False, "error": str(exc)}
                    status(f"re-executing refined code for step {idx}/{max_steps}")
                    result = self.sandbox.execute(
                        code=code,
                        files=uploaded_files,
                        artifact_dir=artifact_dir,
                        requirements=requirements,
                    )

                if not result.get("success"):
                    return {
                        "success": False,
                        "error": (
                            "Unable to complete the task after multiple retries. "
                            "Please try a different approach or simplify the request."
                        ),
                    }

                outputs.append(result.get("output", ""))
                artifacts.extend(result.get("artifacts", []))

                status("judging completion")
                judge = self._extract_json(
                    self._call_gemini_text(
                        self._gemini_judge_prompt(user_query, plan, outputs, memory)
                    )
                )
                if judge.get("done") is True:
                    break
                if judge.get("missing"):
                    status("updating plan")
                    plan_response = self._call_gemini_text(
                        self._gemini_plan_prompt(user_query, file_names, memory)
                    )
                    plan = self._extract_json(plan_response)
                    steps = plan.get("steps", steps)

            if fast_response:
                status("done")
                return {
                    "success": True,
                    "text": "\n".join(outputs),
                    "artifacts": artifacts,
                    "raw_output": "\n".join(outputs),
                }

            status("summarizing")
            final_text = self._call_gemini_text(self._output_prompt(user_query, outputs))
            status("done")
            return {
                "success": True,
                "text": final_text,
                "artifacts": artifacts,
                "raw_output": "\n".join(outputs),
            }

        llm_client, llm_model = self._openai_compatible_client_and_model()
        if llm_client is None:
            return {
                "success": False,
                "error": "No OpenAI-compatible client configured for the selected provider.",
            }

        outputs: list[str] = []
        artifacts: list[str] = []
        analysis_messages: list[dict[str, Any]] = list(messages)
        max_rounds = 8
        max_exec_attempts = 3

        for round_idx in range(1, max_rounds + 1):
            status(f"thinking (round {round_idx}/{max_rounds})")
            message_payload = self._call_openai_compatible_message(
                llm_client=llm_client,
                llm_model=llm_model,
                messages=analysis_messages,
                tools_payload=tools,
                tool_choice="auto",
                status_cb=status if provider == "ollama" else None,
            )
            if message_payload.get("error"):
                return {"success": False, "error": message_payload["error"]}

            assistant_content = message_payload.get("content", "") or ""
            tool_calls = message_payload.get("tool_calls") or []

            if not tool_calls:
                candidate_answer = assistant_content.strip()
                if not candidate_answer:
                    analysis_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Supervisor: no actionable response was produced. "
                                "Reassess the plan and continue."
                            ),
                        }
                    )
                    continue

                if fast_response:
                    status("done")
                    return {"success": True, "text": candidate_answer, "artifacts": []}

                status("supervisor judging response")
                judge = self._judge_openai_compatible_answer(
                    llm_client=llm_client,
                    llm_model=llm_model,
                    provider=provider,
                    user_query=user_query,
                    candidate_answer=candidate_answer,
                    outputs=outputs,
                    memory=memory,
                    status_cb=status,
                )
                if judge.get("done") is True:
                    status("done")
                    return {
                        "success": True,
                        "text": candidate_answer,
                        "artifacts": list(dict.fromkeys(artifacts)),
                        "raw_output": "\n".join(outputs),
                    }

                analysis_messages.append({"role": "assistant", "content": candidate_answer})
                missing = judge.get("missing") or judge.get("reason") or "Answer incomplete."
                status("supervisor reassessing plan")
                analysis_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Supervisor feedback: the answer is not fully satisfying yet.\n"
                            f"Missing: {missing}\n"
                            "Reassess, run additional analysis with execute_python if needed, "
                            "and try again."
                        ),
                    }
                )
                continue

            tool_call = tool_calls[0]
            function_payload = tool_call.get("function") or {}
            function_name = function_payload.get("name")
            tool_call_id = tool_call.get("id") or f"call_round_{round_idx}"
            assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant_content}
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name or "",
                        "arguments": function_payload.get("arguments", ""),
                    },
                }
            ]
            analysis_messages.append(assistant_message)

            if function_name != "execute_python":
                analysis_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name or "unknown_tool",
                        "content": (
                            "Unsupported tool requested. Use execute_python only and continue."
                        ),
                    }
                )
                analysis_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Supervisor: only execute_python is allowed. "
                            "Reassess and retry with execute_python."
                        ),
                    }
                )
                continue

            try:
                args = json.loads(function_payload.get("arguments", "{}"))
            except json.JSONDecodeError as exc:
                analysis_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "execute_python",
                        "content": f"Invalid tool arguments JSON: {exc}",
                    }
                )
                analysis_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Supervisor: read the argument error and send a corrected "
                            "execute_python call."
                        ),
                    }
                )
                continue

            code = args.get("code", "")
            requirements = args.get("requirements") or []
            requirements, auto_added = self._augment_requirements_from_code(code, requirements)
            if auto_added:
                status(f"installing dependencies first: {', '.join(auto_added)}")

            execution_result: dict | None = None
            execution_error = ""
            for exec_attempt in range(1, max_exec_attempts + 1):
                try:
                    status("validating generated code")
                    validate_python_code(code, self._requirements_to_allowed(requirements))
                except CodeValidationError as exc:
                    execution_error = str(exc)
                    blocked_module = self._extract_disallowed_import(execution_error)
                    if (
                        blocked_module
                        and self._is_safe_dependency_candidate(blocked_module)
                        and self._normalize_requirement_name(blocked_module)
                        not in {self._normalize_requirement_name(r) for r in requirements}
                    ):
                        requirements.append(blocked_module)
                        requirements = self._dedupe_requirements(requirements)
                        status(f"installing missing dependency first: {blocked_module}")
                        continue
                    break

                status("executing generated code")
                result = self.sandbox.execute(
                    code=code,
                    files=uploaded_files,
                    artifact_dir=artifact_dir,
                    requirements=requirements,
                )
                if result.get("success"):
                    execution_result = result
                    break

                execution_error = result.get("error", "Unknown error.")
                missing_module = self._extract_missing_module(execution_error)
                missing_root = (
                    missing_module.split(".")[0].replace("-", "_").strip()
                    if missing_module
                    else ""
                )
                if (
                    missing_root
                    and self._is_safe_dependency_candidate(missing_root)
                    and missing_root
                    not in {self._normalize_requirement_name(r) for r in requirements}
                ):
                    requirements.append(missing_root)
                    requirements = self._dedupe_requirements(requirements)
                    status(f"installing missing dependency first: {missing_root}")
                    continue
                if exec_attempt < max_exec_attempts:
                    status("reassessing after execution error")

            if execution_result is None:
                analysis_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "execute_python",
                        "content": (
                            "Execution failed.\n"
                            f"Error:\n{execution_error}\n"
                            "Read this error carefully, update plan/code/requirements, "
                            "and retry."
                        ),
                    }
                )
                analysis_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Supervisor: do not stop on this error. Reassess next plan and "
                            "try again until the task is solved."
                        ),
                    }
                )
                continue

            output_text = execution_result.get("output", "")
            outputs.append(output_text)
            artifacts.extend(execution_result.get("artifacts", []))
            analysis_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": "execute_python",
                    "content": (
                        "Execution succeeded.\n"
                        f"Output:\n{output_text}\n"
                        f"Artifacts: {execution_result.get('artifacts', [])}"
                    ),
                }
            )

            if fast_response:
                status("done")
                return {
                    "success": True,
                    "text": output_text,
                    "artifacts": list(dict.fromkeys(artifacts)),
                    "raw_output": "\n".join(outputs),
                }

            status("summarizing")
            summary_payload = self._call_openai_compatible_message(
                llm_client=llm_client,
                llm_model=llm_model,
                messages=analysis_messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Provide the concise final answer to the user's original request."
                        ),
                    }
                ],
                status_cb=status if provider == "ollama" else None,
            )
            if summary_payload.get("error"):
                analysis_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Supervisor: summarization failed. Reassess and continue."
                        ),
                    }
                )
                continue

            candidate_answer = (summary_payload.get("content") or "").strip()
            if not candidate_answer:
                analysis_messages.append(
                    {
                        "role": "user",
                        "content": "Supervisor: empty answer. Reassess and continue.",
                    }
                )
                continue

            status("supervisor judging response")
            judge = self._judge_openai_compatible_answer(
                llm_client=llm_client,
                llm_model=llm_model,
                provider=provider,
                user_query=user_query,
                candidate_answer=candidate_answer,
                outputs=outputs,
                memory=memory,
                status_cb=status,
            )
            if judge.get("done") is True:
                status("done")
                return {
                    "success": True,
                    "text": candidate_answer,
                    "artifacts": list(dict.fromkeys(artifacts)),
                    "raw_output": "\n".join(outputs),
                }

            analysis_messages.append({"role": "assistant", "content": candidate_answer})
            missing = judge.get("missing") or judge.get("reason") or "Answer incomplete."
            status("supervisor reassessing plan")
            analysis_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Supervisor feedback: the answer is still not satisfactory.\n"
                        f"Missing: {missing}\n"
                        "Reassess the next plan and continue working until solved."
                    ),
                }
            )

        return {
            "success": False,
            "error": (
                "Unable to fully satisfy the request after multiple supervisor iterations. "
                "Please simplify the request and try again."
            ),
        }

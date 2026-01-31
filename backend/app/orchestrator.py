import json
import os
import re
from typing import Any, Optional

import requests
from openai import OpenAI

from app.llm.tools import tools
from app.sandbox.executor import CodeSandbox
from app.security.validators import CodeValidationError, DEFAULT_ALLOWED_MODULES, validate_python_code


class AnalysisPipeline:
    _PREINSTALLED_MODULES = (
        "pandas, numpy, sklearn, scipy, seaborn, statsmodels, matplotlib, "
        "json, math, statistics"
    )

    def _extract_missing_module(self, error: str) -> Optional[str]:
        match = re.search(r"No module named '([^']+)'", error)
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

    def __init__(self, sandbox: Optional[CodeSandbox] = None) -> None:
        self.sandbox = sandbox or CodeSandbox()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.nvapi_key = os.getenv("NV_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_key) if self.openai_key else None
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.nvapi_model = os.getenv("NV_API_MODEL", "moonshotai/kimi-k2.5")
        self.nvapi_url = os.getenv(
            "NV_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions"
        )

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
        )

    def _requirements_to_allowed(self, requirements: list[str]) -> set[str]:
        allowed = set(DEFAULT_ALLOWED_MODULES)
        for req in requirements:
            name = req.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0]
            name = name.strip().replace("-", "_")
            if name:
                allowed.add(name.split(".")[0])
        return allowed

    def _provider(self) -> str:
        if self.nvapi_key:
            return "nvidia"
        return "openai"

    def _nvidia_tool_prompt(self, user_query: str, file_names: list[str]) -> str:
        return (
            f"{self._system_prompt()}\n"
            "Return a single JSON object and nothing else.\n"
            "If you need to execute code, respond with:\n"
            '{"tool":"execute_python","code":"...","requirements":["pkg"],"reasoning":"..."}\n'
            "If no code is needed, respond with:\n"
            '{"tool":"none","response":"..."}\n'
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
        )

    def _nvidia_plan_prompt(self, user_query: str, file_names: list[str]) -> str:
        return (
            f"{self._system_prompt()}\n"
            "Create a multi-step analysis plan. Return JSON only with:\n"
            '{"steps":[{"id":1,"goal":"...","notes":"..."}]}\n'
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
        )

    def _nvidia_step_prompt(
        self, user_query: str, file_names: list[str], step: dict, prior_outputs: list[str]
    ) -> str:
        return (
            f"{self._system_prompt()}\n"
            "You are executing a step from a plan. Return JSON only.\n"
            "If code is needed:\n"
            '{"tool":"execute_python","code":"...","requirements":["pkg"],"reasoning":"..."}\n'
            "If no code is needed:\n"
            '{"tool":"none","response":"..."}\n'
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
            f"Step: {step}\n"
            f"Prior outputs: {prior_outputs}\n"
        )

    def _nvidia_refine_prompt(
        self, user_query: str, file_names: list[str], step: dict, error: str
    ) -> str:
        return (
            f"{self._system_prompt()}\n"
            "The previous execution failed. Provide corrected code only.\n"
            '{"tool":"execute_python","code":"...","requirements":["pkg"],"reasoning":"..."}\n'
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
            f"Step: {step}\n"
            f"Error: {error}\n"
        )

    def _nvidia_refine_prompt_with_attempt(
        self,
        user_query: str,
        file_names: list[str],
        step: dict,
        error: str,
        attempt: int,
    ) -> str:
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
            f"Files available: {file_names}\n"
            f"Request: {user_query}\n"
            f"Step: {step}\n"
            f"Error: {error}\n"
            f"Attempt: {attempt}\n"
        )

    def _nvidia_judge_prompt(
        self, user_query: str, plan: dict, outputs: list[str]
    ) -> str:
        return (
            f"{self._system_prompt()}\n"
            "Judge if the analysis fully answers the user's question. Return JSON only:\n"
            '{"done":true,"reason":"...","missing":"..."}\n'
            f"Request: {user_query}\n"
            f"Plan: {plan}\n"
            f"Outputs: {outputs}\n"
        )

    def _call_nvidia_text(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.nvapi_key}",
            "Accept": "application/json",
        }
        payload = {
            "model": self.nvapi_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": False,
            "chat_template_kwargs": {"thinking": True},
        }
        try:
            response = requests.post(self.nvapi_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            return f"NVIDIA API error: {exc}"

    def process(
        self,
        user_query: str,
        uploaded_files: dict[str, Any],
        artifact_dir: str,
        status_cb: Optional[callable] = None,
    ) -> dict:
        def status(message: str) -> None:
            if status_cb:
                status_cb(message)

        fast_response = "return only" in user_query.lower() or "no explanation" in user_query.lower()
        if not self.openai_key and not self.nvapi_key:
            return {
                "success": False,
                "error": "Set OPENAI_API_KEY or NV_API_KEY before running.",
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

        if self._provider() == "nvidia":
            file_names = list(uploaded_files.keys())
            status("planning analysis")
            plan_response = self._call_nvidia_text(
                self._nvidia_plan_prompt(user_query, file_names)
            )
            plan = self._extract_json(plan_response)
            if plan.get("tool") == "error":
                return {"success": False, "error": plan.get("error", "Plan error.")}

            steps = plan.get("steps", [])
            if not isinstance(steps, list) or not steps:
                return {"success": False, "error": "Planner returned no steps."}

            outputs: list[str] = []
            artifacts: list[str] = []
            max_steps = min(len(steps), 6)
            max_retries = 10

            for idx, step in enumerate(steps[:max_steps], start=1):
                goal = step.get("goal", "step")
                status(f"step {idx}/{max_steps}: {goal}")
                step_result = self._call_nvidia_text(
                    self._nvidia_step_prompt(user_query, file_names, step, outputs)
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
                    refine = self._call_nvidia_text(
                        self._nvidia_refine_prompt_with_attempt(
                            user_query,
                            file_names,
                            step,
                            result.get("error", ""),
                            attempt,
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
                    self._call_nvidia_text(self._nvidia_judge_prompt(user_query, plan, outputs))
                )
                if judge.get("done") is True:
                    break
                if judge.get("missing"):
                    status("updating plan")
                    plan_response = self._call_nvidia_text(
                        self._nvidia_plan_prompt(user_query, file_names)
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
            final_text = self._call_nvidia_text(
                "Summarize the analysis clearly based on these outputs:\n"
                + "\n".join(outputs)
            )
            status("done")
            return {
                "success": True,
                "text": final_text,
                "artifacts": artifacts,
                "raw_output": "\n".join(outputs),
            }

        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.function.name != "execute_python":
                return {
                    "success": False,
                    "error": f"Unsupported tool: {tool_call.function.name}",
                }
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                return {
                    "success": False,
                    "error": f"Invalid tool arguments: {exc}",
                }

            code = args.get("code", "")
            requirements = args.get("requirements") or []
            try:
                validate_python_code(code, self._requirements_to_allowed(requirements))
            except CodeValidationError as exc:
                return {"success": False, "error": str(exc)}

            result = self.sandbox.execute(
                code=code,
                files=uploaded_files,
                artifact_dir=artifact_dir,
                requirements=requirements,
            )
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "Unknown error.")}

            if fast_response:
                return {
                    "success": True,
                    "text": result.get("output", ""),
                    "artifacts": result.get("artifacts", []),
                    "raw_output": result.get("output", ""),
                }

            followup = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": user_query},
                    {
                        "role": "assistant",
                        "content": (
                            "Executed code successfully.\n"
                            f"Output:\n{result.get('output','')}"
                        ),
                    },
                    {"role": "user", "content": "Explain these results clearly."},
                ],
            )
            return {
                "success": True,
                "text": followup.choices[0].message.content,
                "artifacts": result.get("artifacts", []),
                "raw_output": result.get("output", ""),
            }

        return {"success": True, "text": message.content, "artifacts": []}

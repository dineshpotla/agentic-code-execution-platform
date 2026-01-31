import ast
from typing import Optional, Set

DEFAULT_ALLOWED_MODULES = {
    "pandas",
    "numpy",
    "matplotlib",
    "sklearn",
    "scipy",
    "seaborn",
    "statsmodels",
    "json",
    "math",
    "statistics",
}


class CodeValidationError(ValueError):
    pass


def validate_python_code(code: str, allowed_modules: Optional[Set[str]] = None) -> None:
    allowed = allowed_modules or DEFAULT_ALLOWED_MODULES
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeValidationError(f"Invalid Python syntax: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    modules = [node.module]
            for module in modules:
                if module.split(".")[0] not in allowed:
                    raise CodeValidationError(f"Import '{module}' not allowed.")

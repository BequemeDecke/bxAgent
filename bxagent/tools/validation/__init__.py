from typing import Optional

from .factory import create_audit_tools
from .implementations import FileExistenceValidation, JavaCompilationValidation
from .executor import ValidationExecutor
from .types import Validation


def build_audit_tools(audits: Optional[dict[str, Validation]] = None):
    """Factory function to create audit tools. This can be extended to include more audits in the future."""
    if audits is None:
        audits = {
            "file_existence": FileExistenceValidation(),
            "java_compilation": JavaCompilationValidation(),
        }
    return create_audit_tools(audits=audits)


__all__ = ["FileExistenceValidation", "JavaCompilationValidation", "create_audit_tools", "build_audit_tools", "ValidationExecutor", "Validation"]

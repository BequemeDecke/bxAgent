from typing import Optional

from .factory import create_audit_tools
from .implementations import FileExistenceAudit, JavaCompilationAudit
from .executor import AuditExecutor
from .types import Audit


def build_audit_tools(audits: Optional[dict[str, Audit]] = None):
    """Factory function to create audit tools. This can be extended to include more audits in the future."""
    if audits is None:
        audits = {
            "file_existence": FileExistenceAudit,
            "java_compilation": JavaCompilationAudit,
        }
    return create_audit_tools(audits=audits)


__all__ = ["FileExistenceAudit", "JavaCompilationAudit", "create_audit_tools", "build_audit_tools", "AuditExecutor", "Audit"]

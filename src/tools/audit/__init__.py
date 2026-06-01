from .factory import create_audit_tools
from .implementations import FileExistenceAudit, JavaCompilationAudit

"""This module utilizes the auditing core as a tool for the agents.
"""
audit_tools = create_audit_tools(
    audits={
        "file_existence": FileExistenceAudit,
        "java_compilation": JavaCompilationAudit,
    }
)

__all__ = ["audit_tools"]
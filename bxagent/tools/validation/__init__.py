from typing import Optional

from .factory import create_validation_tools
from .implementations import FileExistenceValidation, JavaCompilationValidation
from .executor import ValidationExecutor
from .types import Validation


def build_validation_tools(validations: Optional[dict[str, Validation]] = None):
    """Factory function to create validation tools. This can be extended to include more validations in the future."""
    if validations is None:
        validations = {
            "file_existence": FileExistenceValidation(),
            "java_compilation": JavaCompilationValidation(),
        }
    return create_validation_tools(validations=validations)


__all__ = ["FileExistenceValidation", "JavaCompilationValidation", "create_validation_tools", "build_validation_tools", "ValidationExecutor", "Validation"]

import shutil
import logging

from typing import List, Tuple

from ..types import AuditResult, AuditError, Audit


class JavaCompilationAudit(Audit):
    """
    Audit that checks if Java files can be compiled successfully.

    It uses the `javac` command to attempt to compile the provided Java files. If `javac` is not installed on the system, the setup will fail. The run method will return an empty list of results and errors for now, as the actual compilation logic is not implemented yet.
    """

    def __init__(self, files: List[str]):
        self.files = files

    async def setup(self):
        """Check if `javac` is available on the system. If not, raise a RuntimeError.
        """
        if shutil.which("javac") is None:
            logging.error("javac is not installed on the system, but is required for Java compilation.")
            raise RuntimeError("javac is not installed on the system.")        
        
        logging.debug("javac is available on the system. JavaCompilationAudit setup completed successfully.")

    async def run(self) -> Tuple[AuditResult, List[AuditError]]:
        return [], []

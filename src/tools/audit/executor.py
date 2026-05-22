from typing import List

from .types import Audit

class AuditExecutor:
    def __init__(self, audits: List[Audit] = None):
        self.audits: List[Audit] = []
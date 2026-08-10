import subprocess

from .state import PreparationState

BENCHMARX_REPOSITORY_URL = "https://github.com/tbuchmann/benchmarxUpdates.git"
BENCHMARX_REPOSITORY_NAME = "benchmarx"

def create_download_benchmarx_node(benchmarx_repo_url: str = BENCHMARX_REPOSITORY_URL):
    def download_benchmarx(state: PreparationState) -> PreparationState | None:
        """
        Downloads the benchmarx tool into the workspace if it is not already present.
        """
        if not state.get("install_benchmarx", True):
            return  # Skip installation if the flag is set to False)

        workspace_path = state.get("workspace_path")  # Safe operation

        cp = subprocess.run(
            ["git", "clone", "--depth", "1", benchmarx_repo_url, BENCHMARX_REPOSITORY_NAME],
            cwd=workspace_path,
            check=True,
        )

        if cp.returncode != 0:
            raise RuntimeError(
                f"Failed to clone benchmarx repository. Return code: {cp.returncode}"
            )

        return {
            "benchmarx_path": workspace_path / BENCHMARX_REPOSITORY_NAME,
        }
    
    return download_benchmarx

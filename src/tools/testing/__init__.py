import logging
import subprocess

from langchain.tools import tool
from pathlib import Path

@tool("check_file_existence")
def check_file_existence(file_path: Path):
    """Checks if a file exists on the filesystem

    Args:
        file_path (Path): The path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    logging.debug(f"Checking existence of file: {file_path}")
    return file_path.exists()

@tool("check_compilation")
def check_compilation(working_directory: Path):
    """Checks if the program is able to compile. Uses the `javac` Command.

    Args:
        working_directory (Path): _description_
    """
    
    result = subprocess.run(["javac", "*.java"], cwd=working_directory, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

@tool("execute_benchmark")
def execute_benchmark(working_directory: Path, benchmark_command: str):
    """Executes a benchmark command in the specified working directory.

    Args:
        working_directory (Path): The directory where the benchmark command should be executed.
        benchmark_command (str): The benchmark command to execute.

    Returns:
        tuple: A tuple containing the return code, stdout, and stderr of the benchmark execution.
    """
    result = subprocess.run(benchmark_command.split(), cwd=working_directory, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


testing_tools = (check_file_existence, check_compilation, execute_benchmark)
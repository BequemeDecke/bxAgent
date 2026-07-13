from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from typing import List, Optional, TypedDict
from bxagent.util import register_all_namespaces

class Dependency(TypedDict):
    group_id: str
    artifact_id: str
    version: Optional[str]


def add_module_to_pom(pom_path: Path, group_id: str, artifact_id: str, version: Optional[str] = None):
    """
    Add a module to the given pom.xml content.
    Returns the modified pom.xml content as a string.
    """
    register_all_namespaces(pom_path)
    tree = ET.parse(pom_path)
    root = tree.getroot()
    modules_element = root.find("modules")

    if modules_element is None:
        modules_element = ET.SubElement(root, "modules")

    module_element = ET.SubElement(modules_element, "module")
    module_element.text = artifact_id

    # Write the modified XML back to the pom.xml file
    tree.write(pom_path, encoding="utf-8", xml_declaration=True)


def add_dependencies_to_pom(pom_path: Path, dependencies: List[Dependency]):
    """
    Add dependencies to the given pom.xml content.
    Returns the modified pom.xml content as a string.
    """
    register_all_namespaces(pom_path)
    tree = ET.parse(pom_path)
    root = tree.getroot()
    dependencies_element = root.find("dependencies")

    if dependencies_element is None:
        dependencies_element = ET.SubElement(root, "dependencies")

    for dep in dependencies:
        dependency_element = ET.SubElement(dependencies_element, "dependency")
        group_id_element = ET.SubElement(dependency_element, "groupId")
        group_id_element.text = dep["group_id"]

        artifact_id_element = ET.SubElement(dependency_element, "artifactId")
        artifact_id_element.text = dep["artifact_id"]

        if dep.get("version"):
            version_element = ET.SubElement(dependency_element, "version")
            version_element.text = dep["version"]

    # Write the modified XML back to the pom.xml file
    tree.write(pom_path, encoding="utf-8", xml_declaration=True)


def install_dependencies(workspace: Path):
    cp_process = subprocess.run(["mvn", "validate"], cwd=workspace, check=True)
    if cp_process.returncode != 0:
        raise RuntimeError(
            f"Failed to create Maven project. Return code: {cp_process.returncode}"
        )

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

from mdeagent.util import get_all_namespaces


class Dependency(TypedDict):
    group_id: str
    artifact_id: str
    version: str | None


class Plugin(TypedDict):
    group_id: str
    artifact_id: str
    version: str | None
    configuration: str | None


def add_module_to_pom(
    pom_path: Path, group_id: str, artifact_id: str, version: str | None = None
):
    """
    Add a module to the given pom.xml content.
    Returns the modified pom.xml content as a string.
    """
    namespaces = get_all_namespaces(pom_path)
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])

    tree = ET.parse(pom_path)
    root = tree.getroot()
    modules_element = root.find("modules", namespaces)

    if modules_element is None:
        modules_element = ET.SubElement(root, "modules")

    module_element = ET.SubElement(modules_element, "module")
    module_element.text = artifact_id

    # Write the modified XML back to the pom.xml file
    tree.write(pom_path, encoding="utf-8", xml_declaration=True)


def add_dependencies_to_pom(pom_path: Path, dependencies: list[Dependency]):
    """
    Add dependencies to the given pom.xml content.
    Returns the modified pom.xml content as a string.
    """
    namespaces = get_all_namespaces(pom_path)
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])

    tree = ET.parse(pom_path)
    root = tree.getroot()
    dependencies_element = root.find("dependencies", namespaces)

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


def add_plugin_to_pom(pom_path: Path, plugin: Plugin):
    """
    Add a plugin to the given pom.xml content.
    Returns the modified pom.xml content as a string.
    """
    namespaces = get_all_namespaces(pom_path)
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])

    tree = ET.parse(pom_path)
    root = tree.getroot()
    build_element = root.find("build", namespaces)

    if build_element is None:
        build_element = ET.SubElement(root, "build")

    plugins_element = build_element.find("plugins", namespaces)

    if plugins_element is None:
        plugins_element = ET.SubElement(build_element, "plugins")

    plugin_element = ET.SubElement(plugins_element, "plugin")
    group_id_element = ET.SubElement(plugin_element, "groupId")
    group_id_element.text = plugin["group_id"]

    artifact_id_element = ET.SubElement(plugin_element, "artifactId")
    artifact_id_element.text = plugin["artifact_id"]

    if plugin.get("version"):
        version_element = ET.SubElement(plugin_element, "version")
        version_element.text = plugin["version"]

    if plugin.get("configuration"):
        configuration_element = ET.SubElement(plugin_element, "configuration")
        # Parse the configuration XML and append it as a deep copy
        config_tree = ET.fromstring(plugin["configuration"])

        # Create a new element with the same tag and recursively copy children
        def deep_copy_element(elem):
            new_elem = ET.Element(elem.tag, elem.attrib)
            new_elem.text = elem.text
            new_elem.tail = elem.tail
            for child in elem:
                new_elem.append(deep_copy_element(child))
            return new_elem

        configuration_element.append(deep_copy_element(config_tree))

    # Write the modified XML back to the pom.xml file
    tree.write(pom_path, encoding="utf-8", xml_declaration=True)


def format_java_files(workspace: Path):
    """
    Run mvn spotless:apply to format all Java files in the workspace.
    Raises RuntimeError if the formatting fails.
    """
    cp_process = subprocess.run(
        ["mvn", "spotless:apply"], cwd=workspace, capture_output=True, text=True
    )
    if cp_process.returncode != 0:
        raise RuntimeError(
            f"Failed to format Java files. Return code: {cp_process.returncode}\n"
            f"stdout: {cp_process.stdout}\n"
            f"stderr: {cp_process.stderr}"
        )

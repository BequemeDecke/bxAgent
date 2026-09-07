import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict
from warnings import deprecated

from mdeagent.util import get_all_namespaces

# Base POM template for parent/aggregator projects
# This is the same template used in prepare_workspace.py
BASE_POM_XML = """<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>{group_id}</groupId>
  <artifactId>{artifact_id}</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>

  <name>Workspace</name>
  <modules>
    <!-- Module werden hier hinzugefügt -->
  </modules>

  <properties>
    <maven.compiler.source>11</maven.compiler.source>
    <maven.compiler.target>11</maven.compiler.target>
  </properties>

</project>
"""


class Dependency(TypedDict):
    group_id: str
    artifact_id: str
    version: str | None

    @classmethod
    def from_etree_element(cls, element: ET.Element, namespaces: dict[str, str]) -> "Dependency":
        group_id_element = element.find("groupId", namespaces)
        artifact_id_element = element.find("artifactId", namespaces)
        if group_id_element is None or artifact_id_element is None:
            raise ValueError("Dependency must have both groupId and artifactId.")
        group_id = group_id_element.text
        artifact_id = artifact_id_element.text
        if group_id is None or artifact_id is None:
            raise ValueError("Dependency must have both groupId and artifactId.")
        version = element.find("version", namespaces).text if element.find("version", namespaces) is not None else None
        return cls(group_id=group_id, artifact_id=artifact_id, version=version)

    def to_etree_element(self, parent: ET.Element) -> ET.Element:
        dependency_element = ET.SubElement(parent, "dependency")
        group_id_element = ET.SubElement(dependency_element, "groupId")
        group_id_element.text = self["group_id"]
        artifact_id_element = ET.SubElement(dependency_element, "artifactId")
        artifact_id_element.text = self["artifact_id"]
        if self.get("version"):
            version_element = ET.SubElement(dependency_element, "version")
            version_element.text = self["version"]
        return dependency_element


class Plugin(TypedDict):
    group_id: str | None
    artifact_id: str
    version: str | None
    configuration: str | None

    @classmethod
    def from_etree_element(cls, element: ET.Element, namespaces: dict[str, str]) -> "Plugin":
        artifact_id_element = element.find("artifactId", namespaces)
        if artifact_id_element is None:
            raise ValueError("Plugin must have an artifactId.")
        artifact_id = artifact_id_element.text
        if artifact_id is None:
            raise ValueError("Plugin must have an artifactId.")

        group_id_element = element.find("groupId", namespaces)
        group_id = group_id_element.text if group_id_element is not None else None
        version_element = element.find("version", namespaces)
        version = version_element.text if version_element is not None else None
        configuration_element = element.find("configuration", namespaces)
        configuration = (
            ET.tostring(configuration_element, encoding="unicode")
            if configuration_element is not None
            else None
        )

        return cls(
            group_id=group_id,
            artifact_id=artifact_id,
            version=version,
            configuration=configuration,
        )

    def to_etree_element(self, parent: ET.Element) -> ET.Element:
        plugin_element = ET.SubElement(parent, "plugin")
        group_id_element = ET.SubElement(plugin_element, "groupId")
        group_id_element.text = self["group_id"]
        artifact_id_element = ET.SubElement(plugin_element, "artifactId")
        artifact_id_element.text = self["artifact_id"]
        if self.get("version"):
            version_element = ET.SubElement(plugin_element, "version")
            version_element.text = self["version"]
        if self.get("configuration"):
            configuration_element = ET.fromstring(self["configuration"])
            plugin_element.append(configuration_element)
        return plugin_element


class Module(TypedDict):
    artifact_id: str

    @classmethod
    def from_etree_element(cls, element: ET.Element) -> "Module":
        artifact_id = element.text
        if artifact_id is None:
            raise ValueError("Module must have an artifactId.")
        return cls(artifact_id=artifact_id)

    def to_etree_element(self, parent: ET.Element) -> ET.Element:
        module_element = ET.SubElement(parent, "module")
        module_element.text = self["artifact_id"]
        return module_element


class Pom:
    """Proxy class for managing Maven pom.xml files with optimized access patterns.

    This class implements the Proxy design pattern to provide efficient access and
    modification of pom.xml elements. It caches references to key XML elements
    (_modules_element, _dependencies_element, _plugins_element) to avoid repeated
    tree traversals when adding multiple entries.

    The proxy ensures structural integrity by creating required parent elements
    (e.g., <build>, <pluginManagement>) on-demand if they don't exist.

    Attributes:
        pom_path (Path): Filesystem path to the pom.xml file being managed.
        modules (list[Module]): List of module definitions parsed from the pom.xml.
        dependencies (list[Dependency]): List of dependency definitions parsed from the pom.xml.
        plugins (list[Plugin]): List of plugin definitions parsed from the pom.xml.

    Example:
        ```python
        pom = PomProxy(Path("workspace/module/pom.xml"))
        pom.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
        pom.add_plugin({"group_id": "org.apache.maven.plugins", "artifact_id": "maven-compiler-plugin", "version": "3.11.0"})
        pom.save()
        ```
    """

    # Internal state
    pom_path: Path
    modules: list[Module]
    dependencies: list[Dependency]
    plugins: list[Plugin]
    packaging_value: str | None = None

    # Etree elements
    registered_namespaces: dict[str, str]
    _tree: ET.ElementTree
    _modules_element: ET.Element
    _dependencies_element: ET.Element
    _plugins_element: ET.Element
    _packaging_element: ET.Element | None = None

    def __init__(self, pom_path: Path):
        """Initialize the PomProxy with an existing pom.xml file.

        Parses the pom.xml at the specified path, registers XML namespaces, extracts
        existing modules/dependencies/plugins into cache lists, and caches references
        to key structural elements. Creates missing elements (<modules>, <dependencies>,
        <plugins>) to ensure a valid POM structure.

        Args:
            pom_path (Path): Path to the pom.xml file to manage. The file must exist.

        Raises:
            FileNotFoundError: If the pom.xml file does not exist at pom_path.
            ET.ParseError: If the pom.xml file contains malformed XML.
        """
        self.pom_path = pom_path
        self.modules = []
        self.dependencies = []
        self.plugins = []

        self.registered_namespaces = get_all_namespaces(pom_path)
        for ns in self.registered_namespaces:
            ET.register_namespace(ns, self.registered_namespaces[ns])

        self._parse_pom()

    def _parse_pom(self):
        """Parse the pom.xml file and populate the internal state.
        This method reads the pom.xml file, registers XML namespaces, and extracts
        existing modules, dependencies, and plugins into the corresponding lists.
        It also caches references to key XML elements for efficient future modifications.
        """
        self._tree = ET.parse(self.pom_path)
        root = self._tree.getroot()

        if root is None:
            raise ValueError("Cannot parse pom.xml because the root element is None.")

        # Parse modules
        modules_element = root.find("modules", self.registered_namespaces)
        if modules_element is None:
            modules_element = ET.SubElement(root, "modules")

        self._modules_element = modules_element

        for module_elem in modules_element.findall("module", self.registered_namespaces):
            module = Module.from_etree_element(module_elem)
            self.modules.append(module)

        # Parse dependencies
        dependencies_element = root.find("dependencies", self.registered_namespaces)
        if dependencies_element is None:
            dependencies_element = ET.SubElement(root, "dependencies")

        self._dependencies_element = dependencies_element

        for dep_elem in dependencies_element.findall("dependency", self.registered_namespaces):
            dependency = Dependency.from_etree_element(dep_elem, self.registered_namespaces)
            self.dependencies.append(dependency)

        # Parse plugins
        build_element = root.find("build", self.registered_namespaces)
        if build_element is None:
            build_element = ET.SubElement(root, "build")

        plugin_management_element = build_element.find("pluginManagement", self.registered_namespaces)
        if plugin_management_element is None:
            plugin_management_element = ET.SubElement(build_element, "pluginManagement")

        plugins_element = plugin_management_element.find("plugins", self.registered_namespaces)
        if plugins_element is None:
            plugins_element = ET.SubElement(plugin_management_element, "plugins")

        self._plugins_element = plugins_element

        for plugin_elem in plugins_element.findall("plugin", self.registered_namespaces):
            plugin = Plugin.from_etree_element(plugin_elem, self.registered_namespaces)
            self.plugins.append(plugin)

        # Parse packaging
        packaging_element = root.find("packaging", self.registered_namespaces)
        self._packaging_element = packaging_element

        if packaging_element is not None:
            self.packaging_value = packaging_element.text
        

    def add_module(self, module: Module) -> "Pom":
        """Add a new module reference to the pom.xml.

        Creates a new <module> entry within the <modules> section. This is typically
        used in multi-module Maven projects to declare child modules. If a module
        with the same artifact_id already exists, this method does nothing and
        returns self without making changes.

        Args:
            module (Module): A dictionary containing the module's group_id, artifact_id, and version.

        Returns:
            PomProxy: Returns self to enable method chaining. If the module already
                exists, no changes are made but self is still returned.

        Note:
            Changes are buffered in memory and only written to the XML file when
            save() is called.

        Example:
            ```python
            pom.add_module("com.example", "submodule-a")
               .add_module("com.example", "submodule-b")
               .add_module("com.example", "submodule-a")  # Ignored (duplicate)
               .save()
            ```
        """
        # Check for duplicate module (by artifact_id)
        for existing_module in self.modules:
            if existing_module["artifact_id"] == module["artifact_id"]:
                return self  # Module already exists, skip adding

        # Update cache only - XML will be updated on save()
        self.modules.append(
            Module(
                group_id=module["group_id"],
                artifact_id=module["artifact_id"],
                version=module.get("version"),
            )
        )
        return self

    def add_dependency(self, dependency: Dependency) -> "Pom":
        """Add a new dependency to the pom.xml. Checks for duplicates and updates version if provided.

        Args:
            dependency (Dependency): TypedDict containing dependency metadata with keys:
                - group_id (str): Maven groupId (e.g., "org.junit.jupiter")
                - artifact_id (str): Maven artifactId (e.g., "junit-jupiter-api")
                - version (str | None): Optional version specification

        Returns:
            PomProxy: Returns self to enable method chaining. If the dependency already
                exists, its version is updated (if provided) but no duplicate is created.

        Note:
            Changes are buffered in memory and only written to the XML file when
            save() is called.

        Example:
            ```python
            pom.add_dependency({
                "group_id": "org.junit.jupiter",
                "artifact_id": "junit-jupiter-api",
                "version": "5.10.0"
            }).save()
            ```
        """
        group_id = dependency["group_id"]
        artifact_id = dependency["artifact_id"]
        version = dependency.get("version")

        # Check for existing dependency
        for idx, existing_dep in enumerate(self.dependencies):
            if (
                existing_dep["group_id"] == group_id
                and existing_dep["artifact_id"] == artifact_id
            ):
                # Dependency exists - update version if provided
                if version:
                    self.dependencies[idx]["version"] = version
                    self._modified = True
                return self

        # Dependency doesn't exist - add to cache (XML will be updated on save())
        self.dependencies.append(
            Dependency(group_id=group_id, artifact_id=artifact_id, version=version)
        )
        return self

    def add_plugin(self, plugin: Plugin) -> "Pom":
        """Add a new plugin definition to the pom.xml. Checks for duplicates and updates version/configuration if provided.

        Args:
            plugin (Plugin): TypedDict containing plugin metadata with keys:
                - group_id (str): Maven groupId (e.g., "org.apache.maven.plugins")
                - artifact_id (str): Maven artifactId (e.g., "maven-compiler-plugin")
                - version (str | None): Plugin version (required for pluginManagement)
                - configuration (str | None): Optional XML configuration fragment as string
                    (without outer <configuration> tags)

        Returns:
            PomProxy: Returns self to enable method chaining. If the plugin already
                exists, its version and configuration are updated (if provided) but
                no duplicate is created.

        Note:
            Changes are buffered in memory and only written to the XML file when
            save() is called.

        Example:
            ```python
            pom.add_plugin({
                "group_id": "com.diffplug.maven",
                "artifact_id": "spotless-maven-plugin",
                "version": "2.41.0",
                "configuration": "<java><googleJavaFormat/></java>"
            }).save()
            ```
        """
        group_id = plugin["group_id"]
        artifact_id = plugin["artifact_id"]
        version = plugin.get("version")
        configuration = plugin.get("configuration")

        # Check for existing plugin
        for idx, existing_plugin in enumerate(self.plugins):
            if (
                existing_plugin["group_id"] == group_id
                and existing_plugin["artifact_id"] == artifact_id
            ):
                # Plugin exists - update version/configuration if provided
                if version:
                    self.plugins[idx]["version"] = version
                if configuration:
                    self.plugins[idx]["configuration"] = configuration
                return self

        # Plugin doesn't exist - add to cache (XML will be updated on save())
        self.plugins.append(
            Plugin(
                group_id=group_id,
                artifact_id=artifact_id,
                version=version,
                configuration=configuration,
            )
        )
        return self

    def set_packaging(self, packaging: str) -> "Pom":
        """Set the project packaging type in the pom.xml.

        Updates or creates the <packaging> element. Common values include "jar",
        "pom" (for parent/aggregator projects), "war", or "maven-plugin".

        Args:
            packaging (str): The packaging type identifier.

        Returns:
            PomProxy: Returns self to enable method chaining.

        Note:
            Changes are buffered in memory and only written to the XML file when
            save() is called.

        Example:
            ```python
            pom.set_packaging("pom").save()  # For parent POM
            ```
        """
        # Store packaging value for save() to apply
        self.packaging_value = packaging
        return self

    def _apply_changes_to_xml(self) -> None:
        """Apply all cached changes to the XML tree. Called by save()."""
        # Clear existing modules, dependencies, and plugins in the XML tree
        for elem in list(self._modules_element):
            self._modules_element.remove(elem)
        for elem in list(self._dependencies_element):
            self._dependencies_element.remove(elem)
        for elem in list(self._plugins_element):
            self._plugins_element.remove(elem)
        

        # Re-add modules
        for module in self.modules:
            module.to_etree_element(self._modules_element)

        # Re-add dependencies
        for dep in self.dependencies:
            dep.to_etree_element(self._dependencies_element)

        # Re-add plugins
        for plugin in self.plugins:
            plugin.to_etree_element(self._plugins_element)

        # Update packaging if set
        if self.packaging_value is not None:
            if self._packaging_element is None:
                self._packaging_element = ET.SubElement(self._tree.getroot(), "packaging")
            self._packaging_element.text = self.packaging_value

    def save(self) -> None:
        """Persist the current state of the PomProxy to the pom.xml file.

        Returns:
            None

        Raises:
            ValueError: If the internal XML tree (_root) is None, indicating
                improper initialization.

        Warning:
            This operation overwrites the original pom.xml file. Ensure backups
            exist before modifying critical POM files.
        """
        # Apply all cached changes to the XML tree
        self._apply_changes_to_xml()

        # Write to file
        self._tree.write(self.pom_path, encoding="utf-8", xml_declaration=True)

    @classmethod
    def new(cls, workspace: Path, group_id: str, artifact_id: str) -> "Pom":
        """Create a minimal base pom.xml for an empty aggregator/parent module.

        Factory method that generates a new pom.xml in the workspace root directory
        with basic structure including model version declaration, groupId, and
        packaging type 'pom'. Uses the standard BASE_POM_XML template.

        The pom.xml is always created at workspace/pom.xml to ensure consistent
        file naming and location.

        Args:
            workspace (Path): Workspace directory where the pom.xml will be created.
                The pom.xml will be placed at workspace/pom.xml.
            group_id (str): Maven groupId for the project (e.g., "de.hofuniversity").
            artifact_id (str): Maven artifactId for the project (e.g., "my-project").
            
        Returns:
            PomProxy: A configured PomProxy instance pointing to workspace/pom.xml.

        Example:
            ```python
            # Creates workspace/pom.xml with groupId "de.hofuniversity" and artifactId "my-project"
            pom = PomProxy.base(Path("workspace"), "de.hofuniversity", "my-project")
            pom.add_module("de.hofuniversity", "transformation-module").save()
            ```
        """
        pom_path = workspace / "pom.xml"
        pom_content = BASE_POM_XML.format(group_id=group_id, artifact_id=artifact_id)
        workspace.mkdir(parents=True, exist_ok=True)
        pom_path.write_text(pom_content, encoding="utf-8")
        return cls(pom_path)


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


# deprecated
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

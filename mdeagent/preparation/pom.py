import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

from mdeagent.util import get_all_namespaces


class Dependency(TypedDict):
    group_id: str
    artifact_id: str
    version: str | None

    @classmethod
    def from_etree_element(cls, element: ET.Element) -> 'Dependency':
        group_id = element.findtext("groupId")
        artifact_id = element.findtext("artifactId")
        version = element.findtext("version")
        return cls(group_id=group_id, artifact_id=artifact_id, version=version)


class Plugin(TypedDict):
    group_id: str
    artifact_id: str
    version: str | None
    configuration: str | None

    @classmethod
    def from_etree_element(cls, element: ET.Element) -> 'Plugin':
        group_id = element.findtext("groupId")
        artifact_id = element.findtext("artifactId")
        version = element.findtext("version")
        configuration_element = element.find("configuration")
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


class Module(TypedDict):
    group_id: str
    artifact_id: str
    version: str | None

    @classmethod
    def from_etree_element(cls, element: ET.Element) -> 'Module':
        group_id = element.findtext("groupId")
        artifact_id = element.findtext("artifactId")
        version = element.findtext("version")
        return cls(group_id=group_id, artifact_id=artifact_id, version=version)


class PomProxy:
    """Proxy class for managing Maven pom.xml files with optimized access patterns.
    
    This class implements the Proxy design pattern to provide efficient access and 
    modification of pom.xml elements. It caches references to key XML elements 
    (_modules_element, _dependencies_element, _plugins_element) to avoid repeated 
    tree traversals when adding multiple entries.
    
    The proxy ensures structural integrity by creating required parent elements 
    (e.g., <build>, <pluginManagement>) on-demand if they don't exist.
    
    Attributes:
        pom_path (Path): Filesystem path to the pom.xml file being managed.
        modules (list[Module]): Cache of module definitions from the pom.xml.
        dependencies (list[Dependency]): Cache of dependency definitions from the pom.xml.
        plugins (list[Plugin]): Cache of plugin definitions from the pom.xml.
    
    Private Attributes:
        _root (ET.Element | None): Cached reference to the root <project> element.
        _modules_element (ET.Element | None): Cached reference to the <modules> element.
        _dependencies_element (ET.Element | None): Cached reference to the <dependencies> element.
        _plugins_element (ET.Element | None): Cached reference to the <plugins> element 
            within <pluginManagement>. New plugins are added here to define versions.
    
    Example:
        ```python
        pom = PomProxy(Path("workspace/module/pom.xml"))
        pom.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
        pom.add_plugin({"group_id": "org.apache.maven.plugins", "artifact_id": "maven-compiler-plugin", "version": "3.11.0"})
        pom.save()
        ```
    """
    modules: list[Module]
    dependencies: list[Dependency]
    plugins: list[Plugin]

    _root: ET.Element | None = None
    _modules_element: ET.Element | None = None
    _dependencies_element: ET.Element | None = None
    _plugins_element: ET.Element | None = None

    def __init__(self, pom_path: Path):
        """Initialize the PomProxy with an existing pom.xml file.
        
        Parses the pom.xml at the specified path, registers XML namespaces, and caches
        references to key structural elements. Creates missing elements (<modules>, 
        <dependencies>, <plugins>) to ensure a valid POM structure.
        
        Args:
            pom_path (Path): Path to the pom.xml file to manage. The file must exist.
        
        Raises:
            FileNotFoundError: If the pom.xml file does not exist at pom_path.
            ET.ParseError: If the pom.xml file contains malformed XML.
        """
        self.pom_path = pom_path

        namespaces = get_all_namespaces(pom_path)
        for ns in namespaces:
            ET.register_namespace(ns, namespaces[ns])

        tree = ET.parse(pom_path)
        self._root = tree.getroot()

        # Ensure that the modules element exists
        self._modules_element = self._root.find("modules", namespaces)
        if self._modules_element is None:
            self._modules_element = ET.SubElement(self._root, "modules")
        # Ensure that the dependencies element exists
        self._dependencies_element = self._root.find("dependencies", namespaces)
        if self._dependencies_element is None:
            self._dependencies_element = ET.SubElement(self._root, "dependencies")
        # Ensure that the plugins element exists
        self._plugins_element = self._root.find("plugins", namespaces)
        if self._plugins_element is None:
            build_element = self._root.find("build", namespaces)
            if build_element is None:
                build_element = ET.SubElement(self._root, "build")

            plugin_management_element = build_element.find("pluginManagement", namespaces)
            if plugin_management_element is None:
                plugin_management_element = ET.SubElement(build_element, "pluginManagement")

            self._plugins_element = ET.SubElement(plugin_management_element, "plugins")

    def add_module(self, group_id: str, artifact_id: str, version: str | None = None) -> 'PomProxy':
        """Add a new module reference to the pom.xml.
        
        Creates a new <module> entry within the <modules> section. This is typically 
        used in multi-module Maven projects to declare child modules.
        
        Args:
            group_id (str): The Maven groupId of the module (e.g., "com.example").
            artifact_id (str): The Maven artifactId of the module (e.g., "my-module").
            version (str | None): Optional version string. Not typically used for 
                module references but included for completeness.
        
        Returns:
            PomProxy: Returns self to enable method chaining.
        
        Example:
            ```python
            pom.add_module("com.example", "submodule-a")
               .add_module("com.example", "submodule-b")
               .save()
            ```
        """
        pass

    def add_dependency(self, dependency: Dependency) -> 'PomProxy':
        """Add a new dependency to the pom.xml.
        
        Creates a new <dependency> entry within the <dependencies> section. The 
        dependency dictionary must contain at minimum group_id and artifact_id.
        Version is optional for managed dependencies.
        
        Args:
            dependency (Dependency): TypedDict containing dependency metadata with keys:
                - group_id (str): Maven groupId (e.g., "org.junit.jupiter")
                - artifact_id (str): Maven artifactId (e.g., "junit-jupiter-api")
                - version (str | None): Optional version specification
        
        Returns:
            PomProxy: Returns self to enable method chaining.
        
        Example:
            ```python
            pom.add_dependency({
                "group_id": "org.junit.jupiter",
                "artifact_id": "junit-jupiter-api",
                "version": "5.10.0"
            }).save()
            ```
        """
        pass

    def add_plugin(self, plugin: Plugin) -> 'PomProxy':
        """Add a new plugin definition to the pom.xml.
        
        Creates a new <plugin> entry within the <plugins> section under 
        <pluginManagement>. This defines plugin versions for consistent builds 
        across modules. Actual plugin usage occurs in individual module pom.xml files.
        
        Args:
            plugin (Plugin): TypedDict containing plugin metadata with keys:
                - group_id (str): Maven groupId (e.g., "org.apache.maven.plugins")
                - artifact_id (str): Maven artifactId (e.g., "maven-compiler-plugin")
                - version (str | None): Plugin version (required for pluginManagement)
                - configuration (str | None): Optional XML configuration fragment as string
                    (without outer <configuration> tags)
        
        Returns:
            PomProxy: Returns self to enable method chaining.
        
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
        pass

    def set_packaging(self, packaging: str) -> 'PomProxy':
        """Set the project packaging type in the pom.xml.
        
        Updates or creates the <packaging> element. Common values include "jar", 
        "pom" (for parent/aggregator projects), "war", or "maven-plugin".
        
        Args:
            packaging (str): The packaging type identifier.
        
        Returns:
            PomProxy: Returns self to enable method chaining.
        
        Example:
            ```python
            pom.set_packaging("pom").save()  # For parent POM
            ```
        """
        pass

    def save(self) -> None:
        """Persist all pending changes to the pom.xml file.
        
        Writes the modified XML tree back to disk. This method should be called 
        after completing all modifications to minimize I/O operations.
        
        Returns:
            None
        
        Raises:
            ValueError: If the internal XML tree (_root) is None, indicating 
                improper initialization.
        
        Warning:
            This operation overwrites the original pom.xml file. Ensure backups 
            exist before modifying critical POM files.
        """
        if self._root is None:
            raise ValueError("Cannot save pom.xml because the root element is None.")
        else:
            tree = ET.ElementTree(self._root)
            tree.write(self.pom_path, encoding="utf-8", xml_declaration=True)

    @classmethod
    def base(cls) -> 'PomProxy':
        """Create a minimal base pom.xml for an empty aggregator/parent module.
        
        Factory method that generates a new pom.xml with basic structure including
        placeholder groupId/artifactId/version and empty sections for modules,
        dependencies, and plugins. The temporary file is created in the system temp
        directory.
        
        Returns:
            PomProxy: A configured PomProxy instance pointing to the newly created
                temporary pom.xml file.
        
        Note:
            The created pom.xml has placeholder coordinates that should be updated
            via direct XML manipulation before saving. This method is primarily useful
            for testing or scaffolding scenarios.
        
        Example:
            ```python
            pom = PomProxy.base()
            # Customize the generated pom.xml
            pom.set_packaging("pom").save()
            ```
        """
        pass


# deprecated
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

# deprecated
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


# deprecated
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

# deprecated
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

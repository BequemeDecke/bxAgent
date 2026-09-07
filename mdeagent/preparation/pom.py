import subprocess
from warnings import deprecated
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

from mdeagent.util import get_all_namespaces

# Base POM template for parent/aggregator projects
# This is the same template used in prepare_workspace.py
BASE_POM_XML = """<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>{group_id}</groupId>
  <artifactId>workspace</artifactId>
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
        modules (list[Module]): List of module definitions parsed from the pom.xml.
        dependencies (list[Dependency]): List of dependency definitions parsed from the pom.xml.
        plugins (list[Plugin]): List of plugin definitions parsed from the pom.xml.
    
    Private Attributes:
        _root (ET.Element | None): Cached reference to the root <project> element.
        _modules_element (ET.Element | None): Cached reference to the <modules> element.
        _dependencies_element (ET.Element | None): Cached reference to the <dependencies> element.
        _plugins_element (ET.Element | None): Cached reference to the <plugins> element 
            within <pluginManagement>. New plugins are added here to define versions.
        _namespaces (dict): Cached namespace mappings for XML operations.
        _modified (bool): Flag indicating if changes have been made since last save.
    
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
    _namespaces: dict[str, str] = {}
    _modified: bool = False

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
        self._modified = False
        self._packaging_value: str | None = None

        # Parse namespaces and register them
        self._namespaces = get_all_namespaces(pom_path)
        for ns in self._namespaces:
            ET.register_namespace(ns, self._namespaces[ns])

        # Parse the pom.xml file
        tree = ET.parse(pom_path)
        self._root = tree.getroot()

        # Parse and cache existing modules
        self._modules_element = self._root.find("modules", self._namespaces)
        if self._modules_element is None:
            self._modules_element = ET.SubElement(self._root, "modules")
        else:
            for module_elem in self._modules_element.findall("module", self._namespaces):
                # Modules are stored as simple text, not full Module objects
                if module_elem.text:
                    self.modules.append({
                        "group_id": "",
                        "artifact_id": module_elem.text.strip(),
                        "version": None,
                    })

        # Parse and cache existing dependencies
        self._dependencies_element = self._root.find("dependencies", self._namespaces)
        if self._dependencies_element is None:
            self._dependencies_element = ET.SubElement(self._root, "dependencies")
        else:
            for dep_elem in self._dependencies_element.findall("dependency", self._namespaces):
                self.dependencies.append(Dependency.from_etree_element(dep_elem))

        # Parse and cache existing plugins (from pluginManagement/plugins)
        self._plugins_element = self._root.find("plugins", self._namespaces)
        if self._plugins_element is None:
            build_element = self._root.find("build", self._namespaces)
            if build_element is None:
                build_element = ET.SubElement(self._root, "build")

            plugin_management_element = build_element.find("pluginManagement", self._namespaces)
            if plugin_management_element is None:
                plugin_management_element = ET.SubElement(build_element, "pluginManagement")

            self._plugins_element = ET.SubElement(plugin_management_element, "plugins")
        else:
            for plugin_elem in self._plugins_element.findall("plugin", self._namespaces):
                self.plugins.append(Plugin.from_etree_element(plugin_elem))

    def add_module(self, group_id: str, artifact_id: str, version: str | None = None) -> 'PomProxy':
        """Add a new module reference to the pom.xml.
        
        Creates a new <module> entry within the <modules> section. This is typically 
        used in multi-module Maven projects to declare child modules. If a module 
        with the same artifact_id already exists, this method does nothing and 
        returns self without making changes.
        
        Args:
            group_id (str): The Maven groupId of the module (e.g., "com.example").
            artifact_id (str): The Maven artifactId of the module (e.g., "my-module").
            version (str | None): Optional version string. Not typically used for 
                module references but included for completeness.
        
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
            if existing_module["artifact_id"] == artifact_id:
                return self  # Module already exists, skip adding
        
        # Update cache only - XML will be updated on save()
        self.modules.append({
            "group_id": group_id,
            "artifact_id": artifact_id,
            "version": version,
        })
        self._modified = True
        return self

    def add_dependency(self, dependency: Dependency) -> 'PomProxy':
        """Add a new dependency to the pom.xml.
        
        Creates a new <dependency> entry within the <dependencies> section. The 
        dependency dictionary must contain at minimum group_id and artifact_id.
        Version is optional for managed dependencies. If a dependency with the same
        group_id and artifact_id already exists, this method updates the existing
        dependency's version (if provided) instead of creating a duplicate.
        
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
            if existing_dep["group_id"] == group_id and existing_dep["artifact_id"] == artifact_id:
                # Dependency exists - update version if provided
                if version:
                    self.dependencies[idx]["version"] = version
                    self._modified = True
                return self
        
        # Dependency doesn't exist - add to cache (XML will be updated on save())
        self.dependencies.append(dependency)
        self._modified = True
        return self

    def add_plugin(self, plugin: Plugin) -> 'PomProxy':
        """Add a new plugin definition to the pom.xml.
        
        Creates a new <plugin> entry within the <plugins> section under 
        <pluginManagement>. This defines plugin versions for consistent builds 
        across modules. Actual plugin usage occurs in individual module pom.xml files.
        
        If a plugin with the same group_id and artifact_id already exists, this method
        updates the existing plugin's version and configuration instead of creating
        a duplicate.
        
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
            if existing_plugin["group_id"] == group_id and existing_plugin["artifact_id"] == artifact_id:
                # Plugin exists - update version and/or configuration
                updated = False
                if version and existing_plugin.get("version") != version:
                    self.plugins[idx]["version"] = version
                    updated = True
                if configuration and existing_plugin.get("configuration") != configuration:
                    self.plugins[idx]["configuration"] = configuration
                    updated = True
                if updated:
                    self._modified = True
                return self
        
        # Plugin doesn't exist - add to cache (XML will be updated on save())
        self.plugins.append(plugin)
        self._modified = True
        return self

    def set_packaging(self, packaging: str) -> 'PomProxy':
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
        self._packaging_value = packaging
        self._modified = True
        return self
    
    def _apply_changes_to_xml(self) -> None:
        """Apply all cached changes to the XML tree. Called by save()."""
        if not self._modified or self._root is None:
            return
        
        # Apply modules
        if self._modules_element is not None:
            # Clear existing modules
            for child in list(self._modules_element):
                self._modules_element.remove(child)
            # Add all modules from cache
            for module in self.modules:
                module_elem = ET.SubElement(self._modules_element, "module")
                module_elem.text = module["artifact_id"]
        
        # Apply dependencies
        if self._dependencies_element is not None:
            # Clear existing dependencies
            for child in list(self._dependencies_element):
                self._dependencies_element.remove(child)
            # Add all dependencies from cache
            for dep in self.dependencies:
                dep_elem = ET.SubElement(self._dependencies_element, "dependency")
                group_id_elem = ET.SubElement(dep_elem, "groupId")
                group_id_elem.text = dep["group_id"]
                artifact_id_elem = ET.SubElement(dep_elem, "artifactId")
                artifact_id_elem.text = dep["artifact_id"]
                if dep.get("version"):
                    version_elem = ET.SubElement(dep_elem, "version")
                    version_elem.text = dep["version"]
        
        # Apply plugins
        if self._plugins_element is not None:
            # Clear existing plugins
            for child in list(self._plugins_element):
                self._plugins_element.remove(child)
            # Add all plugins from cache
            for plugin in self.plugins:
                plugin_elem = ET.SubElement(self._plugins_element, "plugin")
                group_id_elem = ET.SubElement(plugin_elem, "groupId")
                group_id_elem.text = plugin["group_id"]
                artifact_id_elem = ET.SubElement(plugin_elem, "artifactId")
                artifact_id_elem.text = plugin["artifact_id"]
                if plugin.get("version"):
                    version_elem = ET.SubElement(plugin_elem, "version")
                    version_elem.text = plugin["version"]
                if plugin.get("configuration"):
                    try:
                        config_element = ET.SubElement(plugin_elem, "configuration")
                        wrapped_config = f"<config_root>{plugin['configuration']}</config_root>"
                        config_tree = ET.fromstring(wrapped_config)
                        for child in config_tree:
                            config_element.append(child)
                    except ET.ParseError as e:
                        raise ValueError(
                            f"Failed to parse plugin configuration XML: {plugin['configuration']}. "
                            f"Error: {e}"
                        ) from e
        
        # Apply packaging if set
        if hasattr(self, '_packaging_value') and self._packaging_value:
            packaging_element = self._root.find("packaging", self._namespaces)
            if packaging_element is None:
                # Create packaging element after modelVersion
                model_version = self._root.find("modelVersion", self._namespaces)
                if model_version is not None:
                    packaging_element = ET.Element("packaging")
                    packaging_element.text = self._packaging_value
                    children = list(self._root)
                    idx = children.index(model_version)
                    self._root.insert(idx + 1, packaging_element)
            else:
                packaging_element.text = self._packaging_value

    def save(self) -> None:
        """Persist all pending changes to the pom.xml file.
        
        Applies all cached modifications (modules, dependencies, plugins, packaging) 
        to the XML tree and writes the result to disk. This method should be called 
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
        
        # Apply all cached changes to the XML tree
        self._apply_changes_to_xml()
        
        # Write to file
        tree = ET.ElementTree(self._root)
        tree.write(self.pom_path, encoding="utf-8", xml_declaration=True)
        
        # Reset modified flag
        self._modified = False

    @classmethod
    def base(cls, workspace: Path, group_id: str) -> 'PomProxy':
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
        
        Returns:
            PomProxy: A configured PomProxy instance pointing to workspace/pom.xml.
        
        Example:
            ```python
            # Creates workspace/pom.xml with groupId "de.hofuniversity"
            pom = PomProxy.base(Path("workspace"), "de.hofuniversity")
            pom.add_module("de.hofuniversity", "transformation-module").save()
            ```
        """
        pom_path = workspace / "pom.xml"
        pom_content = BASE_POM_XML.format(group_id=group_id)
        workspace.mkdir(parents=True, exist_ok=True)
        pom_path.write_text(pom_content, encoding="utf-8")
        return cls(pom_path)

    @classmethod
    def new(
        cls,
        workspace: Path,
        group_id: str,
        artifact_id: str,
        version: str = "1.0-SNAPSHOT",
        archetype: str = "maven-archetype-simple",
        archetype_version: str = "1.5",
    ) -> 'PomProxy':
        """Create a new Maven project using the specified archetype.
        
        Factory method that invokes Maven's archetype:generate goal to create a new 
        Maven project structure. The resulting pom.xml is then wrapped in a PomProxy 
        instance for further manipulation.
        
        Args:
            workspace (Path): Directory where the Maven project will be created.
            group_id (str): Maven groupId for the project (e.g., "com.example").
            artifact_id (str): Maven artifactId for the project (e.g., "my-app").
            version (str, optional): Project version. Defaults to "1.0-SNAPSHOT".
            archetype (str, optional): Maven archetype to use. 
                Defaults to "maven-archetype-simple".
            archetype_version (str, optional): Version of the archetype to use. 
                Defaults to "1.5".
        
        Returns:
            PomProxy: A configured PomProxy instance pointing to the newly created
                project's pom.xml file.
        
        Raises:
            RuntimeError: If Maven is not available or the archetype generation fails.
        
        Example:
            ```python
            # Create a simple Maven project
            pom = PomProxy.new(
                workspace=Path("workspace"),
                group_id="com.example",
                artifact_id="my-app"
            )
            
            # Create with a different archetype
            pom = PomProxy.new(
                workspace=Path("workspace"),
                group_id="com.example",
                artifact_id="my-webapp",
                archetype="maven-archetype-webapp"
            )
            ```
        """
        # Ensure workspace exists
        workspace.mkdir(parents=True, exist_ok=True)
        
        # Run Maven archetype:generate
        result = subprocess.run(
            [
                "mvn",
                "archetype:generate",
                f"-DgroupId={group_id}",
                f"-DartifactId={artifact_id}",
                f"-Dversion={version}",
                f"-DarchetypeArtifactId={archetype}",
                f"-DarchetypeVersion={archetype_version}",
                "-DinteractiveMode=false",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create Maven project using archetype '{archetype}'. "
                f"Return code: {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        
        # Return PomProxy for the created project's pom.xml
        pom_path = workspace / artifact_id / "pom.xml"
        if not pom_path.exists():
            raise FileNotFoundError(
                f"Expected pom.xml not found at {pom_path}. "
                "Maven archetype may have failed silently."
            )
        
        return cls(pom_path)


@deprecated
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

@deprecated
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


@deprecated
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

@deprecated
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

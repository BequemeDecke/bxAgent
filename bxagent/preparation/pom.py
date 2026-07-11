import xml.etree.ElementTree as ET
from typing import List, Optional, TypedDict

class Dependency(TypedDict):
    group_id: str
    artifact_id: str
    version: Optional[str]

def add_dependencies_to_pom(pom: str, dependencies: List[Dependency]) -> str:
    """
    Add dependencies to the given pom.xml content.
    Returns the modified pom.xml content as a string.
    """
    root = ET.fromstring(pom)
    dependencies_element = root.find('dependencies')
    
    if dependencies_element is None:
        dependencies_element = ET.SubElement(root, 'dependencies')

    for dep in dependencies:
        dependency_element = ET.SubElement(dependencies_element, 'dependency')
        group_id_element = ET.SubElement(dependency_element, 'groupId')
        group_id_element.text = dep['group_id']
        
        artifact_id_element = ET.SubElement(dependency_element, 'artifactId')
        artifact_id_element.text = dep['artifact_id']
        
        if dep.get('version'):
            version_element = ET.SubElement(dependency_element, 'version')
            version_element.text = dep['version']

    return ET.tostring(root, encoding='unicode')


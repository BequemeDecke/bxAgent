import xml.etree.ElementTree as ET
from pathlib import Path

# Source - https://stackoverflow.com/a/54491129
# Posted by rez, modified by community. See post 'Timeline' for change history
# Retrieved 2026-07-13, License - CC BY-SA 4.0


def register_all_namespaces(filename: Path):
    namespaces = dict([node for _, node in ET.iterparse(filename, events=["start-ns"])])
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])

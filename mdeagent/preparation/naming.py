"""Deterministic naming of the transformation class (and its BxTool adapter).

The name of the transformation class is derived from the *folder names* of the
source and target model packages. Each model package folder carries the name of
its metamodel, so the two names are combined deterministically as
``<Source>To<Target>Transformation`` (and ``<Source>To<Target>BxToolAdapter``
for the adapter). For example, the model folders ``Families`` and ``Persons``
yield ``FamiliesToPersonsTransformation`` and ``FamiliesToPersonsBxToolAdapter``.

Because the naming is purely rule-based, the preparation subgraph does not need
an LLM and stays deterministic.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransformationClassNames:
    """The derived Java class names for the transformation and its BxTool adapter."""

    transformation_class_name: str
    bxtool_adapter_class_name: str


def _metamodel_name(model_path: Path | str | None) -> str:
    """Derive the metamodel name from its package folder path.

    The folder name (``Path(model_path).name``) is the metamodel name. A missing
    path falls back to ``"Unknown"`` so that the naming never fails.
    """
    if model_path is None:
        return "Unknown"
    return Path(model_path).name


def determine_transformation_class_name(
    source_model_path: Path | str | None,
    target_model_path: Path | str | None,
) -> TransformationClassNames:
    """Derive the transformation and BxTool adapter class names deterministically.

    The names follow the pattern ``<Source>To<Target>Transformation`` and
    ``<Source>To<Target>BxToolAdapter``, where ``<Source>`` and ``<Target>`` are
    the folder names of the source and target model packages (each folder
    carries the name of its metamodel).

    Args:
        source_model_path: Path to the source model package folder.
        target_model_path: Path to the target model package folder.

    Returns:
        A :class:`TransformationClassNames` with both derived class names.
    """
    source_name = _metamodel_name(source_model_path)
    target_name = _metamodel_name(target_model_path)
    base = f"{source_name}To{target_name}"
    return TransformationClassNames(
        transformation_class_name=f"{base}Transformation",
        bxtool_adapter_class_name=f"{base}BxToolAdapter",
    )

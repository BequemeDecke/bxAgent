"""Unit tests for the deterministic transformation class naming.

The name of the transformation class (and its BxTool adapter) is derived
deterministically from the *folder names* of the source and target model
packages. Each model package folder carries the name of its metamodel, so the
two names are combined as ``<Source>To<Target>Transformation`` and
``<Source>To<Target>BxToolAdapter`` (e.g. ``Families`` -> ``Persons`` yields
``FamiliesToPersonsTransformation`` / ``FamiliesToPersonsBxToolAdapter``).

This replaces the former LLM-based naming which was too error-prone.
"""

from pathlib import Path
from unittest import TestCase

from mdeagent.preparation.naming import (
    TransformationClassNames,
    determine_transformation_class_name,
)


class TestDetermineTransformationClassName(TestCase):
    def test__derives_names_from_model_folder_paths(self):
        """Families -> Persons yields the documented naming pattern."""
        names = determine_transformation_class_name(
            Path("/some/models/Families"),
            Path("/some/models/Persons"),
        )

        self.assertEqual(
            names.transformation_class_name, "FamiliesToPersonsTransformation"
        )
        self.assertEqual(
            names.bxtool_adapter_class_name, "FamiliesToPersonsBxToolAdapter"
        )

    def test__example_pattern_family_to_person(self):
        """The user-facing example: Family -> Person."""
        names = determine_transformation_class_name(
            Path("models/Family"),
            Path("models/Person"),
        )

        self.assertEqual(
            names.transformation_class_name, "FamilyToPersonTransformation"
        )
        self.assertEqual(
            names.bxtool_adapter_class_name, "FamilyToPersonBxToolAdapter"
        )

    def test__accepts_string_paths(self):
        """String paths are resolved the same way as ``Path`` objects."""
        names = determine_transformation_class_name(
            "/some/path/Families",
            "/some/path/Persons",
        )

        self.assertEqual(
            names.transformation_class_name, "FamiliesToPersonsTransformation"
        )
        self.assertEqual(
            names.bxtool_adapter_class_name, "FamiliesToPersonsBxToolAdapter"
        )

    def test__handles_trailing_slash(self):
        """A trailing slash must not affect the derived folder name."""
        names = determine_transformation_class_name(
            Path("/some/models/Families/"),
            Path("/some/models/Persons/"),
        )

        self.assertEqual(
            names.transformation_class_name, "FamiliesToPersonsTransformation"
        )

    def test__returns_transformation_class_names_instance(self):
        names = determine_transformation_class_name(
            Path("/x/Source"), Path("/x/Target")
        )

        self.assertIsInstance(names, TransformationClassNames)

    def test__missing_source_path_falls_back_to_unknown(self):
        names = determine_transformation_class_name(
            None,
            Path("/x/Persons"),
        )

        self.assertEqual(
            names.transformation_class_name, "UnknownToPersonsTransformation"
        )
        self.assertEqual(
            names.bxtool_adapter_class_name, "UnknownToPersonsBxToolAdapter"
        )

    def test__missing_target_path_falls_back_to_unknown(self):
        names = determine_transformation_class_name(
            Path("/x/Families"),
            None,
        )

        self.assertEqual(
            names.transformation_class_name, "FamiliesToUnknownTransformation"
        )

    def test__both_paths_missing_fall_back_to_unknown(self):
        """When no model paths are available the naming still does not crash."""
        names = determine_transformation_class_name(None, None)

        self.assertEqual(
            names.transformation_class_name, "UnknownToUnknownTransformation"
        )
        self.assertEqual(
            names.bxtool_adapter_class_name, "UnknownToUnknownBxToolAdapter"
        )

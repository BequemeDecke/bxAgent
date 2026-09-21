from pathlib import Path
from unittest import TestCase
from unittest.mock import AsyncMock, Mock, patch

from langchain.chat_models import BaseChatModel

from mdeagent.implementation.transformation.template.generator import (
    BackwardMethodBody,
    CodeGenerator,
    FallbackParser,
    ForwardMethodBody,
    JsonParser,
    StructuredResponseParser,
    SynchMethodBody,
    TransformationClassMetadata,
    TransformationClassSpec,
    TransformationClassTemplateResolver,
    TransformationFieldsAndConstructor,
    YamlLikeParser,
    ainvoke_and_parse,
    create_generate_transformation_node,
    invoke_and_parse,
)
from mdeagent.implementation.state import ImplementationState


class TestTransformationGeneration(TestCase):
    def setUp(self):
        import json
        self.fake_data = {
            "package_name": "com.example.transformation",
            "class_name": "MyTransformation",
            "source_type": "SourceModel",
            "target_type": "TargetModel",
            "decision_type": "TransformationDecisions",
            "fields": [{"type": "String", "name": "label"}],
            "constructor": {
                "parameters": "String label",
                "assignments": [{"target": "label", "value": "label"}],
            },
            "forward_body": "System.out.println(source);",
            "backward_body": "System.out.println(target);",
            "synch_body": 'System.out.println("synced");',
            "transform_source_to_target_body": "forward(source, target, decisions);",
            "transform_target_to_source_body": "backward(target, source, decisions);",
        }

        # Create a mock response with JSON content
        from unittest.mock import AsyncMock
        mocked_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps(self.fake_data)
        mocked_llm.invoke.return_value = mock_response

        self.generate_transformation = create_generate_transformation_node(
            mocked_llm, Path("/tmp/workspace")
        )

    @patch("pathlib.Path.write_text")
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template"
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_generate_transformation__write_template_output(
        self,
        mock_render_template: Mock,
        mock_get_raw_template: Mock,
        mock_write_text: Mock,
    ):
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        state = ImplementationState(
            transformation_md=None,
            task_specification="Implement the transformation class",
            written_java_files=[],
            bxtool_path=Path("/tmp/workspace"),
            transformation_implementation="public class MyTransformation { }",
            latest_evaluation_runs={},
            iteration=1,
        )

        new_state = self.generate_transformation(state)

        self.assertEqual(len(new_state["written_java_files"]), 1)
        self.assertEqual(
            new_state["written_java_files"][0],
            Path("/tmp/workspace/MyTransformation.java"),
        )
        mock_get_raw_template.assert_called_once()
        mock_render_template.assert_called_once_with(
            TransformationClassSpec(**self.fake_data)
        )

    def test_transformation_class_template_resolver__renders_emf_interface(self):
        resolver = TransformationClassTemplateResolver()
        rendered = resolver.render_template(TransformationClassSpec(**self.fake_data))

        self.assertIn(
            "implements AgentTransformationForEMF<",
            rendered,
        )
        self.assertIn("SourceModel", rendered)
        self.assertIn("TargetModel", rendered)
        self.assertIn("TransformationDecisions", rendered)
        self.assertIn("public void forward(SourceModel source", rendered)


class TestStructuredResponseParserInterface(TestCase):
    """Tests for the StructuredResponseParser interface and implementations."""

    def test_json_parser__parses_valid_json(self):
        """JsonParser should parse valid JSON into Pydantic model."""
        parser = JsonParser()
        json_content = '{"package_name": "com.example", "source_type": "Source", "target_type": "Target", "decision_type": "Decisions", "transformation_package": "com.example.api"}'
        
        result = parser.parse(json_content, TransformationClassMetadata)
        
        self.assertEqual(result.package_name, "com.example")
        self.assertEqual(result.source_type, "Source")
        self.assertEqual(result.target_type, "Target")
        self.assertEqual(result.decision_type, "Decisions")

    def test_json_parser__raises_on_invalid_json(self):
        """JsonParser should raise on invalid JSON."""
        parser = JsonParser()
        invalid_json = '{"invalid": json}'
        
        with self.assertRaises(ValueError):
            parser.parse(invalid_json, TransformationClassMetadata)

    def test_yaml_like_parser__parses_simple_yaml(self):
        """YamlLikeParser should parse simple YAML key:value format."""
        parser = YamlLikeParser()
        yaml_content = """package_name: com.example
source_type: SourceModel
target_type: TargetModel
decision_type: Decisions
transformation_package: com.example.api"""
        
        result = parser.parse(yaml_content, TransformationClassMetadata)
        
        self.assertEqual(result.package_name, "com.example")
        self.assertEqual(result.source_type, "SourceModel")

    def test_yaml_like_parser__parses_bold_keys(self):
        """YamlLikeParser should handle **Key:** formatting."""
        parser = YamlLikeParser()
        bold_content = """**Package Name:** com.example.transformation
**Source Type:** SourceModel
**Target Type:** TargetModel
**Decision Type:** Decisions"""
        
        result = parser.parse(bold_content, TransformationClassMetadata)
        
        self.assertEqual(result.package_name, "com.example.transformation")
        self.assertEqual(result.source_type, "SourceModel")

    def test_yaml_like_parser__extracts_from_code_blocks(self):
        """YamlLikeParser should extract content from markdown code blocks."""
        parser = YamlLikeParser()
        code_block_content = '''```json
{"forward_body": "return source.getValue();"}
```'''
        
        result = parser.parse(code_block_content, ForwardMethodBody)
        
        self.assertEqual(result.forward_body, "return source.getValue();")

    def test_yaml_like_parser__handles_single_field_model(self):
        """YamlLikeParser should handle single-field models with code blocks."""
        parser = YamlLikeParser()
        code_content = '''```java
// Java implementation
System.out.println("Hello");
```'''
        
        result = parser.parse(code_content, ForwardMethodBody)
        
        self.assertIn("System.out.println", result.forward_body)

    def test_fallback_parser__tries_json_first(self):
        """FallbackParser should try JSON parser first."""
        parser = FallbackParser()
        json_content = '{"package_name": "com.test", "source_type": "S", "target_type": "T", "decision_type": "D", "transformation_package": "com.test"}'
        
        result = parser.parse(json_content, TransformationClassMetadata)
        
        self.assertEqual(result.package_name, "com.test")

    def test_fallback_parser__falls_back_to_yaml(self):
        """FallbackParser should fall back to YAML parser when JSON fails."""
        parser = FallbackParser()
        yaml_content = """package_name: com.fallback
source_type: Source
target_type: Target
decision_type: Decisions
transformation_package: com.fallback.api"""
        
        result = parser.parse(yaml_content, TransformationClassMetadata)
        
        self.assertEqual(result.package_name, "com.fallback")

    def test_fallback_parser__custom_parser_list(self):
        """FallbackParser should accept custom parser list."""
        yaml_only_parser = FallbackParser(parsers=[YamlLikeParser()])
        yaml_content = "package_name: custom\nsource_type: S\ntarget_type: T\ndecision_type: D\ntransformation_package: c"
        
        result = yaml_only_parser.parse(yaml_content, TransformationClassMetadata)
        
        self.assertEqual(result.package_name, "custom")


class TestCodeGenerator(TestCase):
    """Tests for the CodeGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.fake_metadata = {
            "package_name": "com.example.gen",
            "source_type": "SourceModel",
            "target_type": "TargetModel",
            "decision_type": "Decisions",
            "transformation_package": "com.example",
        }

    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_code_generator_generate__sync_generation(self, mock_render):
        """Test synchronous code generation."""
        mock_render.return_value = "public class Generated { }"
        
        # Setup mock response
        import json
        mock_response = Mock()
        mock_response.content = json.dumps(self.fake_metadata)
        self.mock_llm.invoke.return_value = mock_response
        
        generator = CodeGenerator(self.mock_llm)
        prompt = "Generate metadata"
        
        parsed_model, rendered_code = generator.generate(
            prompt,
            TransformationClassMetadata,
            TransformationClassTemplateResolver,
        )
        
        self.assertIsInstance(parsed_model, TransformationClassMetadata)
        self.assertEqual(parsed_model.package_name, "com.example.gen")
        self.assertEqual(rendered_code, "public class Generated { }")
        mock_render.assert_called_once()

    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_code_generator_generate_async__async_generation(self, mock_render):
        """Test asynchronous code generation."""
        import asyncio
        mock_render.return_value = "public class AsyncGenerated { }"
        
        # Setup mock response
        import json
        mock_response = AsyncMock()
        mock_response.content = json.dumps(self.fake_metadata)
        self.mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        generator = CodeGenerator(self.mock_llm)
        prompt = "Generate metadata async"
        
        async def run_test():
            return await generator.generate_async(
                prompt,
                TransformationClassMetadata,
                TransformationClassTemplateResolver,
            )
        
        parsed_model, rendered_code = asyncio.run(run_test())
        
        self.assertIsInstance(parsed_model, TransformationClassMetadata)
        self.assertEqual(parsed_model.package_name, "com.example.gen")
        self.assertEqual(rendered_code, "public class AsyncGenerated { }")

    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_code_generator_generate_parallel__multiple_prompts(self, mock_render):
        """Test parallel generation of multiple structured responses."""
        import asyncio
        mock_render.return_value = "code"
        
        import json
        # Setup mock responses for parallel calls
        async def mock_ainvoke(prompt):
            mock_response = AsyncMock()
            if "forward" in str(prompt).lower():
                mock_response.content = json.dumps({"forward_body": "// forward"})
            elif "backward" in str(prompt).lower():
                mock_response.content = json.dumps({"backward_body": "// backward"})
            elif "synch" in str(prompt).lower():
                mock_response.content = json.dumps({"synch_body": "// synch"})
            else:
                mock_response.content = json.dumps(self.fake_metadata)
            return mock_response
        
        self.mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        
        generator = CodeGenerator(self.mock_llm)
        prompts_with_models = [
            ("Generate forward", ForwardMethodBody),
            ("Generate backward", BackwardMethodBody),
            ("Generate synch", SynchMethodBody),
        ]
        
        async def run_test():
            return await generator.generate_parallel(prompts_with_models)
        
        results = asyncio.run(run_test())
        
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], ForwardMethodBody)
        self.assertIsInstance(results[1], BackwardMethodBody)
        self.assertIsInstance(results[2], SynchMethodBody)
        self.assertEqual(results[0].forward_body, "// forward")
        self.assertEqual(results[1].backward_body, "// backward")
        self.assertEqual(results[2].synch_body, "// synch")


class TestInvokeAndParseFunctions(TestCase):
    """Tests for the invoke_and_parse helper functions."""

    def test_invoke_and_parse__sync_invocation(self):
        """Test synchronous invoke_and_parse function."""
        mock_llm = Mock(spec=BaseChatModel)
        import json
        mock_response = Mock()
        mock_response.content = json.dumps({
            "package_name": "com.sync",
            "source_type": "S",
            "target_type": "T",
            "decision_type": "D",
            "transformation_package": "com.sync"
        })
        mock_llm.invoke.return_value = mock_response
        
        result = invoke_and_parse(mock_llm, "prompt", TransformationClassMetadata)
        
        self.assertIsInstance(result, TransformationClassMetadata)
        self.assertEqual(result.package_name, "com.sync")

    def test_ainvoke_and_parse__async_invocation(self):
        """Test asynchronous ainvoke_and_parse function."""
        import asyncio
        mock_llm = Mock(spec=BaseChatModel)
        import json
        mock_response = AsyncMock()
        mock_response.content = json.dumps({
            "package_name": "com.async",
            "source_type": "S",
            "target_type": "T",
            "decision_type": "D",
            "transformation_package": "com.async"
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        async def run_test():
            return await ainvoke_and_parse(mock_llm, "prompt", TransformationClassMetadata)
        
        result = asyncio.run(run_test())
        
        self.assertIsInstance(result, TransformationClassMetadata)
        self.assertEqual(result.package_name, "com.async")

    def test_invoke_and_parse__with_custom_parser(self):
        """Test invoke_and_parse with custom parser."""
        mock_llm = Mock(spec=BaseChatModel)
        yaml_content = """package_name: com.custom
source_type: Source
target_type: Target
decision_type: Decisions
transformation_package: com.custom"""
        mock_response = Mock()
        mock_response.content = yaml_content
        mock_llm.invoke.return_value = mock_response
        
        # Use only YamlLikeParser
        from mdeagent.implementation.transformation.template.generator import YamlLikeParser
        custom_parser = YamlLikeParser()
        result = invoke_and_parse(mock_llm, "prompt", TransformationClassMetadata, custom_parser)
        
        self.assertEqual(result.package_name, "com.custom")

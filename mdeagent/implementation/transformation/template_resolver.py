from pathlib import Path

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationRun
from mdeagent.implementation.transformation.template.generator import (
    BackwardMethodBody,
    FallbackParser,
    ForwardMethodBody,
    ImplementationTransformationSpec,
    SynchMethodBody,
    TransformationClassMetadata,
    TransformationClassTemplateResolver,
    TransformationFieldsAndConstructor,
    ainvoke_and_parse,
)
from mdeagent.implementation.transformation.template.prompts import (
    create_backward_body_prompt,
    create_fields_and_constructor_prompt,
    create_forward_body_prompt,
    create_metadata_prompt,
    create_synch_body_prompt,
)
from mdeagent.implementation.types import (
    TransformationClass,
    TransformationClassGenerator,
)


class TemplateResolver(TransformationClassGenerator):
    def synthesize_transformation_class(
        self,
        transformation_plan: TransformationPlan,
        transformation_class: TransformationClass,
        specific_task: str | None = None,
        evaluation_results: dict[str, EvaluationRun] | None = None,
    ) -> list[Path]:
        """Synthesizes the transformation class based on the provided transformation plan and an optional specific task.

        Args:
            transformation_plan (TransformationPlan): The transformation plan to use for generating the transformation class.
            transformation_class (TransformationClass): The transformation class to generate.
            specific_task (str | None, optional): An optional specific task to focus on when generating the transformation class. Defaults to None.
            evaluation_results (dict[str, EvaluationRun] | None, optional): Optional evaluation results that can be used to inform the generation of the transformation class. Defaults to None.

        Returns:
            str: The generated transformation class as a string."""
        raw_template = resolver.get_raw_template()

         # STEP 1: Generate metadata first (needed as context for other parts)
        metadata_prompt = create_metadata_prompt(
            task_specification=specific_task,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            evaluation_results_text=evaluation_results_text,
        )
        metadata_response: TransformationClassMetadata = await invoke_and_parse(
            metadata_prompt, TransformationClassMetadata
        )

        # Prepare fields info string for context in method body generation
        fields_result_dict = metadata_response.model_dump()
        fields_list = fields_result_dict.get("fields", []) or []
        fields_info = "Fields: " + ", ".join(
            f"{f.get('type', 'Object')} {f.get('name', 'field')}" for f in fields_list
        )

        # STEP 2: Generate independent parts in PARALLEL
        # Create all prompts for parallel generation
        fields_prompt = create_fields_and_constructor_prompt(
            task_specification=specific_task,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata_response,
            evaluation_results_text=evaluation_results_text,
        )

        forward_prompt = create_forward_body_prompt(
            task_specification=specific_task,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata_response,
            fields_info=fields_info,
            evaluation_results_text=evaluation_results_text,
        )

        backward_prompt = create_backward_body_prompt(
            task_specification=specific_task,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata_response,
            fields_info=fields_info,
            evaluation_results_text=evaluation_results_text,
        )

        synch_prompt = create_synch_body_prompt(
            task_specification=task_specification,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata_response,
            fields_info=fields_info,
            evaluation_results_text=evaluation_results_text,
        )

        # Execute all four calls in parallel
        (
            fields_result,
            forward_result,
            backward_result,
            synch_result,
        ) = await asyncio.gather(
            invoke_and_parse(fields_prompt, TransformationFieldsAndConstructor),
            invoke_and_parse(forward_prompt, ForwardMethodBody),
            invoke_and_parse(backward_prompt, BackwardMethodBody),
            invoke_and_parse(synch_prompt, SynchMethodBody),
        )

        # STEP 3: Combine all parts into the final spec
        combined_spec = ImplementationTransformationSpec(
            package_name=transformation_package_path,
            source_type=metadata_response.source_type,
            target_type=metadata_response.target_type,
            decision_type=metadata_response.decision_type,
            transformation_package=metadata_response.transformation_package,
            fields=fields_result.fields or [],
            constructor=fields_result.constructor,
            forward_body=forward_result.forward_body,
            backward_body=backward_result.backward_body,
            synch_body=synch_result.synch_body,
            transform_source_to_target_body=None,  # Will default to calling forward
            transform_target_to_source_body=None,  # Will default to calling backward
        )

        # STEP 4: Render the template with the generated specification
        transformation_class_name = transformation_class_path.stem
        rendered_code = resolver.render_template(
            combined_spec, class_name=transformation_class_name
        )

        # STEP 5: Write the generated code to a file
        transformation_class_path.touch(exist_ok=True)
        transformation_class_path.write_text(rendered_code, encoding="utf-8")

        # STEP 6: Retrieve the written files from the state and add the new one
        written_java_files = state.get("written_java_files", []) + [
            transformation_class_path
        ]

        return f"// Transformation class {transformation_class_name} in package {transformation_package}\n"
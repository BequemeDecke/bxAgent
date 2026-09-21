PROMPT_TEMPLATE_WITH_PLAN = """
You are a Java transformation code generator for EMF-based model transformations.
Generate a concrete implementation of the AgentTransformationForEMF interface based on the task specification and the provided template.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return a valid structured result matching the required Java class structure.
The implementation must use the EMF interface methods and the Java generic types for source, target, and decisions.
"""
# Specialized prompts for piecewise generation
METADATA_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Extract the metadata (package names and type names) for the transformation class.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return the package name, source type, target type, decision type, and the transformation package where AgentTransformationForEMF is declared.
"""
FIELDS_AND_CONSTRUCTOR_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Define the fields and constructor for the transformation class.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN METADATA ---
Package: {package_name}
Source Type: {source_type}
Target Type: {target_type}
Decision Type: {decision_type}
--- END METADATA ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return the field declarations and constructor definition for the transformation class.
Fields should be a list of objects with 'type' and 'name'.
The constructor should have 'parameters' (string) and 'assignments' (list of {{target, value}}).
If no fields or constructor are needed, return empty/null values.
"""
FORWARD_BODY_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Implement the forward transformation method body.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN METADATA ---
Package: {package_name}
Source Type: {source_type}
Target Type: {target_type}
Decision Type: {decision_type}
--- END METADATA ---

--- BEGIN FIELDS ---
{fields_info}
--- END FIELDS ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return only the Java code for the forward method body (no method signature, just the body content).
The forward method transforms from {source_type} to {target_type}.
"""
BACKWARD_BODY_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Implement the backward transformation method body.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN METADATA ---
Package: {package_name}
Source Type: {source_type}
Target Type: {target_type}
Decision Type: {decision_type}
--- END METADATA ---

--- BEGIN FIELDS ---
{fields_info}
--- END FIELDS ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return only the Java code for the backward method body (no method signature, just the body content).
The backward method transforms from {target_type} to {source_type}.
"""
SYNCH_BODY_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Implement the synchronization method body.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN METADATA ---
Package: {package_name}
Source Type: {source_type}
Target Type: {target_type}
Decision Type: {decision_type}
--- END METADATA ---

--- BEGIN FIELDS ---
{fields_info}
--- END FIELDS ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return only the Java code for the synch method body (no method signature, just the body content).
The synch method handles incremental updates between {source_type} and {target_type}.
"""
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.backends import LocalShellBackend, CompositeBackend, FilesystemBackend
from pathlib import Path

from src.models import build_base_model
from .test import build_test_agent
from .synthesis import build_synthesis_agent
from .repair import build_repair_agent

ORCHESTRATOR_SYSTEM_PROMPT = """
You are a specialist in incremental and bidirectional model transformations in the Eclipse Modeling Framework (EMF). Your task is to create a Java implementation of a model transformation based on a natural language description.

## Tasks
1. Read and analyze the provided EMF model (source and target model)
2. Understand the metamodels of both models (Ecore structure)
3. Implement the transformation logic that supports both directions (Forward/Backward)
4. Enable incremental updates instead of full regeneration

## Important Requirements

### EMF Structure
- Analyze the EPackages and EClasses of both metamodels
- Understand the references, attributes, and multiplicities
- Consider bidirectional references and their handshaking behavior
- Pay attention to containment relationships and cross-references

### Incremental Transformations
- Use EMF's adapter mechanism for change tracking (notification system)
- Implement a listener for EObject changes
- Propagate only delta changes instead of complete retransformation
- Avoid redundant regenerations

### Bidirectionality
- Specify transformation rules for both directions (Forward → Backward)
- Define unambiguous mappings between source and target model elements
- Implement appropriate rollback logic for backward transformation

### Code Generation
- Generate runnable Java code
- Use EMF's reflective API where necessary (eGet, eSet, eAllContents)
- Use ResourceSet for model management
- Implement XMI serialization/deserialization where required

## Structure of Generated Code
```java
public class <TransformationName>Transformer {
    // Forward transformation with change tracking
    public void transformForward(SourceModel source, TargetModel target);
    
    // Backward transformation
    public void transformBackward(TargetModel target, SourceModel source);
    
    // Incremental update on changes
    public void setupIncrementalTransformation(EObject source, EObject target);
}
"""


def build_orchestrator_backend(workspace_dir: Path):
    """
    Builds the backend for the BxAgent.
    """

    bxagent_skills_dir = Path.cwd() / "bxagent-skills" / "skills"

    return lambda rt: CompositeBackend(
        default=LocalShellBackend(root_dir=workspace_dir, virtual_mode=True),
        routes={
            "/skills/": FilesystemBackend(
                root_dir=bxagent_skills_dir, virtual_mode=True
            )
        },
    )


def build_bx_agent(
    workspace_dir: Path = Path("agent_data"),
    system_prompt: str = ORCHESTRATOR_SYSTEM_PROMPT,
):
    """Builds the BxAgent using the chat model."""
    model = build_base_model()
    backend = build_orchestrator_backend(workspace_dir)

    synthesis_agent = CompiledSubAgent(
        name="synthesis_agent",
        description="Agent responsible for synthesizing the transformation logic based on the provided system prompt.",
        runnable=build_synthesis_agent(),
    )

    test_agent = CompiledSubAgent(
        name="test_agent",
        description="Agent responsible for testing the synthesized transformation logic.",
        runnable=build_test_agent(),
    )

    repair_agent = CompiledSubAgent(
        name="repair_agent",
        description="Agent responsible for repairing any issues found during testing of the synthesized transformation logic.",
        runnable=build_repair_agent(),
    )

    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=SystemMessage(system_prompt),
        checkpointer=InMemorySaver(),
        subagents=[synthesis_agent, test_agent, repair_agent],
        skills=["/skills/"],
    )

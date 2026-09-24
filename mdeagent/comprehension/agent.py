from langchain.agents import AgentState, create_agent
from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage

from mdeagent.comprehension import SerializedTransformationPlan
from mdeagent.comprehension.tools import transformation_plan_tools
from mdeagent.models import build_base_model

COMPREHENSION_SYSTEM_PROMPT = """
You are the planning agent for the Ecore model transformation process.

Your task is to analyze the source and target models, understand the requirements, 
and create a detailed implementation plan in TRANSFORMATION.md (= the current transformation plan).

Workflow:
1. The transformation plan is given by the user input. If it is empty, start writing it from scratch using the input of the user.
2. Analyze the models, requirements, and existing content
3. Identify new difficulties and think through potential obstacles
4. Define implementation steps that provide a roadmap without dictating code details
5. Document your thinking and reasoning throughout
6. Update the transformation plan using the available tools, which also keep track of the transformation plan history
7. Self-validate: review for logical consistency and completeness

Guidelines for Implementation Steps:
- All implementation must be done in Java using EMF (Eclipse Modeling Framework) technologies
- Implementation steps must be DECLARATIVE only - describe WHAT needs to be done, not HOW with code snippets
- The transformation class must implement the `AgentTransformationForEMF<S, T, A>` interface
- Do NOT use the Notifier-Pattern (no EMF ChangeNotifier, no EContentAdapter)
- Use EMF core technologies: Ecore metamodels, EMF resources, XMI serialization, EObject manipulation
- Each step should reference specific EMF concepts and Java patterns appropriate for model transformations

Recommended structure for Implementation Steps:
1. **Project Setup**: Define Maven dependencies for EMF (org.eclipse.emf.ecore, org.eclipse.emf.ecore.xmi)
2. **Metamodel Configuration**: Describe loading and initializing Ecore packages via EPackage.Registry.INSTANCE
3. **Factory Initialization**: Specify creating factory instances for source and target models
4. **Transformation Class Structure**: Define implementing AgentTransformationForEMF interface with type parameters S (source), T (target), A (decisions/configuration)
5. **Resource Management**: Describe setting up ResourceSet and Resources for model persistence
6. **Forward Transformation**: Declaratively describe transformSourceToTarget method behavior
7. **Backward Transformation**: Declaratively describe transformTargetToSource method behavior  
8. **Synchronization**: Describe synch method for bidirectional consistency
9. **Decision Handling**: Explain how configuration decisions (type A) guide transformation choices
10. **Model Element Mapping**: Describe mapping strategies between source and target EObjects
11. **Reference Resolution**: Explain handling of cross-references and containment relationships
12. **Testing Strategy**: Describe unit test approach using EMF assertion utilities

Example of GOOD declarative step:
"Initialize the EMF resource set and register XMI resource factory for both source and target model file extensions"

Example of BAD prescriptive step (avoid):
"resourceSet.getResourceFactoryRegistry().getExtensionToFactoryMap().put(\"family\", new XMIResourceFactoryImpl());"

Return your plan using the predefined response_schema
"""


class ComprehensionAgentState(AgentState):
    transformation_plan: SerializedTransformationPlan


def build_comprehension_agent(
    system_prompt: str = COMPREHENSION_SYSTEM_PROMPT,
    model: BaseChatModel | None = None,
):
    """Builds the ComprehensionAgent using the chat model."""
    if model is None:
        model = build_base_model()

    return create_agent(
        model=model,
        state_schema=ComprehensionAgentState,
        system_prompt=SystemMessage(system_prompt),
        middleware=[],
        # checkpointer=InMemorySaver(
        #     serde=JsonPlusSerializer(
        #         pickle_fallback=True,
        #         allowed_json_modules=[TransformationPlan],
        #         allowed_msgpack_modules=[TransformationPlan],
        #     )
        # ),
        tools=[*transformation_plan_tools],
    )

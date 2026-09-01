"""Technical documentation mode tools (spec section 9)."""
from __future__ import annotations

from app.core.enums import DocumentType, ResultKind, TaskType
from app.tools.prompt_tool import PromptTool, PromptToolSpec
from app.tools.registry import register_tool

_TD = [DocumentType.TECHNICAL_DOCUMENTATION]

_SPECS = [
    PromptToolSpec(
        id="architecture_overview",
        name="Architecture Overview",
        description="Summarize the system architecture described: major components and how they fit "
        "together.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="Summarize the system architecture: major components/services, how they "
        "communicate, and the overall shape of the system as described in the documentation.",
        icon="network",
    ),
    PromptToolSpec(
        id="component_extraction",
        name="Component Extraction",
        description="Extract the individual components/modules/services described and their "
        "responsibilities.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="Extract each distinct component/module/service mentioned and a short description "
        "of its responsibility.",
        result_kind=ResultKind.SECTIONS,
        icon="box",
    ),
    PromptToolSpec(
        id="api_extraction",
        name="API Extraction",
        description="Extract API endpoints/methods, parameters, and return values described.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="Extract every API endpoint/method documented: its name/path, HTTP verb if "
        "applicable, parameters (with types if given), and return value/response shape.",
        result_kind=ResultKind.SECTIONS,
        top_k=16,
        icon="plug",
    ),
    PromptToolSpec(
        id="requirements_extraction",
        name="Requirements Extraction",
        description="Extract functional and non-functional requirements stated in the documentation.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="Extract the functional and non-functional requirements stated in the "
        "documentation, grouped into those two categories.",
        result_kind=ResultKind.SECTIONS,
        icon="clipboard-check",
    ),
    PromptToolSpec(
        id="dependency_analysis",
        name="Dependency Analysis",
        description="List external dependencies, libraries, services, and versions mentioned.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="List external dependencies (libraries, frameworks, services, infrastructure) "
        "mentioned, with versions/constraints where given.",
        result_kind=ResultKind.LIST,
        icon="package",
    ),
    PromptToolSpec(
        id="workflow_explanation",
        name="Workflow Explanation",
        description="Explain a described workflow or process step by step.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="Explain the requested (or most prominent) workflow/process as an ordered "
        "sequence of steps, noting inputs/outputs and decision points.",
        icon="workflow",
    ),
    PromptToolSpec(
        id="configuration_extraction",
        name="Configuration Extraction",
        description="Extract configuration options, environment variables, and settings documented.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="Extract configuration options/environment variables/settings documented: name, "
        "purpose, and default value where given.",
        result_kind=ResultKind.LIST,
        icon="sliders-horizontal",
    ),
    PromptToolSpec(
        id="security_requirement_extraction",
        name="Security Requirement Extraction",
        description="Extract security requirements, constraints, and considerations mentioned.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="Extract security-relevant requirements and considerations: authn/authz "
        "requirements, data handling/encryption requirements, and any explicitly stated threats or "
        "constraints.",
        result_kind=ResultKind.LIST,
        icon="shield",
    ),
    PromptToolSpec(
        id="implementation_checklist",
        name="Generate Implementation Checklist",
        description="Generate a practical checklist for implementing what's described.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="Generate a practical, ordered implementation checklist for someone building "
        "against this documentation -- concrete, actionable items, not vague restatements.",
        result_kind=ResultKind.LIST,
        icon="list-todo",
    ),
    PromptToolSpec(
        id="technical_summary",
        name="Technical Summary",
        description="A dense, technical-audience summary of the document.",
        category="technical_documentation",
        applicable_types=_TD,
        system_prompt="Write a dense summary aimed at a technical reader (engineer/architect) -- assume "
        "domain familiarity, prioritize precision over accessibility.",
        icon="file-code",
    ),
]

for _spec in _SPECS:
    register_tool(PromptTool(_spec))

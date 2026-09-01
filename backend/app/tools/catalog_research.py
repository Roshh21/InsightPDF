"""Research paper mode tools (spec section 6)."""
from __future__ import annotations

from app.core.enums import DocumentType, ResultKind, TaskType
from app.tools.prompt_tool import PromptTool, PromptToolSpec
from app.tools.registry import register_tool

_RP = [DocumentType.RESEARCH_PAPER]

_SPECS = [
    PromptToolSpec(
        id="analyze_dataset",
        name="Analyze Dataset",
        description="Identify the dataset(s) used: source, size, splits, preprocessing, and any known "
        "biases or limitations mentioned.",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="You are a research-methods analyst. Extract everything the paper says about the "
        "dataset(s) it uses -- name/source, size, train/val/test splits, preprocessing/cleaning steps, "
        "and any stated limitations or biases. If multiple datasets are used, cover each separately.",
        icon="database",
    ),
    PromptToolSpec(
        id="analyze_model",
        name="Analyze Model",
        description="Describe the model(s) proposed or used: type, key components, parameters, training "
        "setup.",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="Extract and explain the model(s) central to this paper: architecture family, key "
        "components/modules, parameter count if stated, and training configuration (optimizer, learning "
        "rate, epochs, hardware) where mentioned.",
        icon="cpu",
    ),
    PromptToolSpec(
        id="analyze_architecture",
        name="Analyze Architecture",
        description="Break down the proposed system/model architecture: components, data flow, and how "
        "pieces connect.",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="Describe the architecture in a way a technically literate reader unfamiliar with "
        "the paper could follow: name each component, what it does, and how components connect / the "
        "flow of information between them.",
        icon="layout-grid",
    ),
    PromptToolSpec(
        id="analyze_methodology",
        name="Analyze Methodology",
        description="Summarize the overall research methodology and experimental approach.",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="Summarize the paper's research methodology: research questions/hypotheses, "
        "overall approach, and how the experiments were designed to test the hypotheses.",
        icon="flask-conical",
    ),
    PromptToolSpec(
        id="extract_experimental_setup",
        name="Extract Experimental Setup",
        description="Extract the concrete experimental setup: hardware, hyperparameters, baselines, "
        "evaluation protocol.",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="Extract concrete, reproducibility-relevant details of the experimental setup: "
        "hardware used, key hyperparameters, baseline methods compared against, and the evaluation "
        "protocol/metrics used.",
        icon="settings-2",
    ),
    PromptToolSpec(
        id="analyze_metrics",
        name="Analyze Metrics",
        description="List and explain the evaluation metrics used and why they were chosen.",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="List every evaluation metric used in the paper, a one-line explanation of what "
        "it measures, and (if stated) why the authors chose it.",
        icon="gauge",
    ),
    PromptToolSpec(
        id="extract_results",
        name="Extract Results",
        description="Extract the key quantitative and qualitative results reported.",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="Extract the paper's key results: headline numbers (with the metric and dataset "
        "they belong to), comparisons to baselines, and any notable qualitative findings. Prefer exact "
        "figures over vague statements.",
        top_k=16,
        icon="trending-up",
    ),
    PromptToolSpec(
        id="analyze_limitations",
        name="Analyze Limitations",
        description="Summarize the limitations the authors acknowledge (and note any that seem "
        "unacknowledged but evident from the results).",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="Summarize the limitations of this work. Prioritize limitations the authors "
        "explicitly state. You may separately note limitations that appear evident from the results but "
        "were not explicitly discussed -- label these clearly as your own observation, not the authors'.",
        icon="alert-triangle",
    ),
    PromptToolSpec(
        id="identify_research_gaps",
        name="Identify Research Gaps",
        description="Identify open problems and research gaps this paper points to or leaves unaddressed.",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="Identify research gaps: problems this paper explicitly flags as open/future work, "
        "and questions the work raises but does not answer. Be specific and grounded in the paper's own "
        "framing, not generic AI-research platitudes.",
        icon="search",
    ),
    PromptToolSpec(
        id="compare_papers",
        name="Compare Papers",
        description="Structured side-by-side comparison of the selected papers: dataset, model, "
        "architecture, method, metrics, results, limitations.",
        category="research_paper",
        applicable_types=_RP,
        requires_multi_document=True,
        system_prompt="Produce a structured comparison of the selected research papers using these "
        "attributes (in this order): Dataset, Model, Architecture, Method, Metrics, Results, Limitations. "
        "Every attribute must be filled for every paper (use 'Not specified' if genuinely absent from the "
        "evidence). Close with a short narrative on which paper looks strongest for which use case.",
        result_kind=ResultKind.COMPARISON,
        task_type=TaskType.COMPARISON,
        top_k=18,
        icon="table-2",
    ),
    PromptToolSpec(
        id="find_common_techniques",
        name="Find Common Techniques",
        description="Across the selected papers, find techniques, methods, or components they share.",
        category="research_paper",
        applicable_types=_RP,
        requires_multi_document=True,
        system_prompt="Compare the selected papers and identify techniques, methods, datasets, or "
        "architectural components that multiple papers share. For each shared element, name which papers "
        "use it.",
        result_kind=ResultKind.SECTIONS,
        task_type=TaskType.COMPARISON,
        icon="git-merge",
    ),
    PromptToolSpec(
        id="find_differences",
        name="Find Differences",
        description="Across the selected papers, find the key differences in approach, assumptions, or "
        "results.",
        category="research_paper",
        applicable_types=_RP,
        requires_multi_document=True,
        system_prompt="Compare the selected papers and identify the most important differences: "
        "differing assumptions, methods, datasets, or results. Be specific about which paper does what "
        "differently.",
        task_type=TaskType.COMPARISON,
        icon="git-compare",
    ),
    PromptToolSpec(
        id="generate_literature_review",
        name="Generate Literature Review",
        description="Generate a literature-review-style synthesis across the selected papers.",
        category="research_paper",
        applicable_types=_RP,
        requires_multi_document=True,
        system_prompt="Write a literature-review-style synthesis of the selected papers: group them by "
        "theme/approach, discuss how they relate to and build on (or contradict) each other, and close "
        "with a synthesis of the overall state of this sub-area based only on these papers.",
        result_kind=ResultKind.REPORT,
        task_type=TaskType.REPORT_GENERATION,
        top_k=18,
        max_tokens=4000,
        icon="library",
    ),
    PromptToolSpec(
        id="generate_research_brief",
        name="Generate Research Brief",
        description="Generate a concise research brief: problem, approach, results, significance.",
        category="research_paper",
        applicable_types=_RP,
        system_prompt="Write a concise research brief suitable for someone deciding whether to read the "
        "full paper: the problem being solved, the approach, headline results, and why it matters.",
        result_kind=ResultKind.REPORT,
        task_type=TaskType.REPORT_GENERATION,
        icon="clipboard-list",
    ),
]

for _spec in _SPECS:
    register_tool(PromptTool(_spec))

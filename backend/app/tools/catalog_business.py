"""Business/financial report mode tools (spec section 10)."""
from __future__ import annotations

from app.core.enums import DocumentType, ResultKind, TaskType
from app.tools.prompt_tool import PromptTool, PromptToolSpec
from app.tools.registry import register_tool

_BR = [DocumentType.BUSINESS_REPORT]

_SPECS = [
    PromptToolSpec(
        id="executive_summary",
        name="Executive Summary",
        description="A concise executive summary of the report.",
        category="business_report",
        applicable_types=_BR,
        system_prompt="Write a concise executive summary: what the report covers, headline numbers, and "
        "the main takeaway, suitable for a busy executive.",
        icon="briefcase",
    ),
    PromptToolSpec(
        id="kpi_extraction",
        name="KPI Extraction",
        description="Extract the key performance indicators reported, with values and period.",
        category="business_report",
        applicable_types=_BR,
        system_prompt="Extract every KPI reported: name, value, unit, and the period/quarter it applies "
        "to. Prefer exact figures over descriptions.",
        result_kind=ResultKind.SECTIONS,
        icon="gauge",
    ),
    PromptToolSpec(
        id="metric_extraction",
        name="Metric Extraction",
        description="Extract other quantitative metrics and figures mentioned throughout the report.",
        category="business_report",
        applicable_types=_BR,
        system_prompt="Extract quantitative metrics/figures mentioned (revenue, costs, headcount, "
        "growth rates, market size, etc.) beyond the headline KPIs, with their values and context.",
        result_kind=ResultKind.LIST,
        top_k=16,
        icon="bar-chart-3",
    ),
    PromptToolSpec(
        id="trend_analysis",
        name="Trend Analysis",
        description="Analyze trends over time evident in the report.",
        category="business_report",
        applicable_types=_BR,
        system_prompt="Analyze trends over time evident in the report -- growth/decline patterns, "
        "inflection points, and stated drivers behind them.",
        icon="trending-up",
    ),
    PromptToolSpec(
        id="risk_extraction",
        name="Risk Extraction",
        description="Extract the risks and risk factors disclosed.",
        category="business_report",
        applicable_types=_BR,
        system_prompt="Extract the risks/risk factors disclosed in the report, grouped by category "
        "(e.g. market, operational, regulatory, financial) where the categorization is evident.",
        result_kind=ResultKind.SECTIONS,
        icon="alert-triangle",
    ),
    PromptToolSpec(
        id="action_items",
        name="Action Items",
        description="Extract stated or clearly implied action items / next steps / strategic priorities.",
        category="business_report",
        applicable_types=_BR,
        system_prompt="Extract action items, next steps, or strategic priorities the report states or "
        "clearly implies. Do not invent actions that aren't grounded in the text.",
        result_kind=ResultKind.LIST,
        icon="list-checks",
    ),
    PromptToolSpec(
        id="yoy_comparison",
        name="Year-over-Year Comparison",
        description="Compare this report's figures across the time periods it covers (e.g. this year vs "
        "last year).",
        category="business_report",
        applicable_types=_BR,
        system_prompt="Produce a year-over-year (or period-over-period) comparison of the key metrics "
        "this single report discusses, using the time periods as the comparison attributes. If the "
        "report only covers one period, say so rather than fabricating a prior period.",
        result_kind=ResultKind.COMPARISON,
        task_type=TaskType.COMPARISON,
        top_k=16,
        icon="calendar-range",
    ),
    PromptToolSpec(
        id="multi_report_comparison",
        name="Multi-report Comparison",
        description="Compare KPIs and narrative across multiple selected reports (e.g. different "
        "companies or different quarters).",
        category="business_report",
        applicable_types=_BR,
        requires_multi_document=True,
        system_prompt="Produce a structured comparison across the selected reports: pick the KPIs/metrics "
        "present in most of them as attributes, fill in each report's value, and close with a narrative "
        "on the most notable differences.",
        result_kind=ResultKind.COMPARISON,
        task_type=TaskType.COMPARISON,
        top_k=18,
        icon="table-2",
    ),
    PromptToolSpec(
        id="generate_business_brief",
        name="Generate Business Brief",
        description="Generate a concise business brief: situation, key numbers, risks, recommendation.",
        category="business_report",
        applicable_types=_BR,
        system_prompt="Write a concise business brief: situation summary, key numbers, notable risks, "
        "and a grounded recommendation/outlook if the evidence supports one.",
        result_kind=ResultKind.REPORT,
        task_type=TaskType.REPORT_GENERATION,
        icon="clipboard-list",
    ),
]

for _spec in _SPECS:
    register_tool(PromptTool(_spec))

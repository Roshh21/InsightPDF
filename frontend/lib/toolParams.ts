export interface ParamField {
  key: string;
  label: string;
  type: "text" | "textarea" | "select" | "number" | "checkbox";
  options?: { value: string; label: string }[];
  placeholder?: string;
  default?: string | number | boolean;
}

const SPOILER_FIELD: ParamField = {
  key: "spoiler_level",
  label: "Spoiler level",
  type: "select",
  options: [
    { value: "none", label: "No spoilers" },
    { value: "chapter", label: "Up to current chapter" },
    { value: "full", label: "Full spoilers" },
  ],
  default: "chapter",
};

const QUERY_FIELD = (label: string, placeholder: string): ParamField => ({
  key: "query",
  label,
  type: "textarea",
  placeholder,
});

export const TOOL_PARAM_FIELDS: Record<string, ParamField[]> = {
  ask_question: [QUERY_FIELD("Your question", "What would you like to know about this document?")],
  explain: [QUERY_FIELD("What to explain", "Which passage, section, or concept should be explained?")],
  chapter_summary: [QUERY_FIELD("Chapter", "Which chapter or section? (leave blank for best match)")],
  get_section: [QUERY_FIELD("Section title", "e.g. Introduction")],

  full_summary: [SPOILER_FIELD],
  character_analysis: [SPOILER_FIELD, QUERY_FIELD("Character (optional)", "Focus on a specific character?")],
  character_relationships: [SPOILER_FIELD],
  theme_analysis: [SPOILER_FIELD],
  plot_analysis: [SPOILER_FIELD],
  important_events: [SPOILER_FIELD],
  motifs_symbols: [SPOILER_FIELD],
  review: [SPOILER_FIELD],

  generate_quiz: [
    { key: "num_questions", label: "Number of questions", type: "number", default: 5 },
    {
      key: "difficulty",
      label: "Difficulty",
      type: "select",
      options: [
        { value: "easy", label: "Easy" },
        { value: "medium", label: "Medium" },
        { value: "hard", label: "Hard" },
      ],
      default: "medium",
    },
  ],
  practice_questions: [
    { key: "num_questions", label: "Number of questions", type: "number", default: 8 },
    {
      key: "difficulty",
      label: "Difficulty",
      type: "select",
      options: [
        { value: "easy", label: "Easy" },
        { value: "medium", label: "Medium" },
        { value: "hard", label: "Hard" },
      ],
      default: "medium",
    },
  ],
  flashcards: [{ key: "num_cards", label: "Number of cards", type: "number", default: 12 }],
  question_paper_generator: [
    { key: "total_marks", label: "Total marks", type: "number", default: 30 },
    { key: "num_questions", label: "Number of questions", type: "number", default: 10 },
    {
      key: "difficulty",
      label: "Difficulty",
      type: "select",
      options: [
        { value: "easy", label: "Easy" },
        { value: "medium", label: "Medium" },
        { value: "hard", label: "Hard" },
        { value: "mixed", label: "Mixed" },
      ],
      default: "mixed",
    },
    {
      key: "question_types",
      label: "Question types",
      type: "text",
      placeholder: "e.g. short answer, long answer, MCQ",
    },
    { key: "include_answer_key", label: "Include answer key", type: "checkbox", default: false },
  ],

  web_research: [QUERY_FIELD("Research question", "e.g. What has changed in this field since publication?")],
  search_documents: [QUERY_FIELD("Search query", "What are you looking for?")],
};

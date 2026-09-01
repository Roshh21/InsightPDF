// Mirrors backend/app/schemas/*.py -- kept in sync by hand since this is a
// small, stable surface. If the API grows significantly, consider codegen
// (e.g. openapi-typescript against /openapi.json).

export type DocumentStatus = "UPLOADED" | "PROCESSING" | "READY" | "FAILED";

export type DocumentType =
  | "research_paper"
  | "literature"
  | "study_material"
  | "technical_documentation"
  | "business_report"
  | "generic";

export interface Workspace {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
}

export interface WorkspaceStats {
  total_documents: number;
  by_status: Record<string, number>;
  by_type: Record<string, number>;
}

export interface WorkspaceDetail extends Workspace {
  stats: WorkspaceStats;
}

export interface DocumentSection {
  title: string;
  page_start: number;
  page_end: number | null;
  level: number;
}

export interface DocumentEntity {
  name: string;
  type: string;
  mentions: number;
}

export interface TableInfo {
  page: number;
  caption: string | null;
  n_rows: number | null;
  n_cols: number | null;
}

export interface DocumentProfile {
  type: DocumentType;
  confidence: number;
  title: string | null;
  authors: string[];
  summary_hint: string | null;
  sections: DocumentSection[];
  entities: DocumentEntity[];
  topics: string[];
  tables: TableInfo[];
  key_metadata: Record<string, unknown>;
  capabilities: string[];
}

export interface Document {
  id: string;
  workspace_id: string;
  original_filename: string;
  status: DocumentStatus;
  error_message: string | null;
  document_type: DocumentType | null;
  classification_confidence: number | null;
  page_count: number | null;
  capabilities: string[];
  profile: DocumentProfile | null;
  created_at: string;
  processed_at: string | null;
  retry_count: number;
}

export interface UploadError {
  filename: string;
  message: string;
}

export interface ToolCatalogEntry {
  id: string;
  name: string;
  description: string;
  category: string;
  result_kind: ResultKind;
  requires_multi_document: boolean;
  icon: string;
}

export type ResultKind =
  | "text"
  | "sections"
  | "table"
  | "comparison"
  | "list"
  | "entities"
  | "quiz"
  | "question_paper"
  | "flashcards"
  | "report";

export interface Citation {
  document_id: string;
  document_name: string;
  page: number;
  section: string | null;
  excerpt: string;
  supported: boolean | null;
}

export interface ToolExecuteResponse {
  execution_id: string;
  tool_name: string;
  result_kind: ResultKind;
  title: string | null;
  content: any;
  citations: Citation[];
  warnings: string[];
  latency_ms: number | null;
  model_used: string | null;
  provider_used: string | null;
  fallback_used: boolean;
}

export interface ToolExecutionHistoryItem {
  id: string;
  tool_name: string;
  status: string;
  document_ids: string[];
  latency_ms: number | null;
  model_used: string | null;
  provider_used: string | null;
  fallback_used: boolean;
  error_message: string | null;
  created_at: string;
}

export interface ChatSession {
  id: string;
  workspace_id: string;
  document_ids: string[];
  spoiler_level: string | null;
  title: string | null;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  citations: Citation[];
  tool_used: string | null;
  result_payload: ToolExecuteResponse | null;
  model_used: string | null;
  provider_used: string | null;
  latency_ms: number | null;
  created_at: string;
}

export interface ChatTurnResponse {
  session_id: string;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

export interface EvaluationSummary {
  retrieval_relevance_avg: number | null;
  answer_faithfulness_avg: number | null;
  citation_accuracy_avg: number | null;
  avg_latency_ms: number | null;
  total_runs: number;
  primary_requests: number;
  fallback_requests: number;
  primary_request_pct: number | null;
  fallback_request_pct: number | null;
  provider_breakdown: Record<string, number>;
  failed_runs: number;
  total_tool_executions: number;
  has_data: boolean;
}

export interface ProviderStatus {
  slot: "primary" | "secondary" | "tertiary";
  provider: string;
  configured: boolean;
  available: boolean;
  cooling_down: boolean;
  cooldown_remaining_s: number;
  model_fast: string;
  model_strong: string;
}

export interface ModelStatus {
  providers: ProviderStatus[];
  active_provider: string | null;
  vector_store: string;
  embedding_model: string;
}

export interface LLMSlotConfig {
  slot: "primary" | "secondary" | "tertiary";
  provider: string;
  configured: boolean;
}

export interface PublicConfig {
  app_name: string;
  environment: string;
  vector_store: string;
  embedding_provider: string;
  embedding_model: string;
  llm_slots: LLMSlotConfig[];
  any_llm_configured: boolean;
  web_research_configured: boolean;
  max_upload_mb: number;
}

// --- structured tool-result content shapes (by result_kind) ---------------

export interface ContentSection {
  heading: string;
  content: string;
}
export interface SectionsContent {
  summary: string;
  sections: ContentSection[];
}
export interface TextContent {
  answer: string;
}
export interface ListContent {
  items: string[];
}
export interface EntitiesContent {
  entities: { name: string; type: string; description: string }[];
}
export interface ComparisonRow {
  document_id: string;
  document_name: string;
  values: Record<string, string>;
}
export interface ComparisonContent {
  attributes: string[];
  rows: ComparisonRow[];
  narrative: string;
}
export interface QuizQuestion {
  question: string;
  type: string;
  options: string[];
  correct_answer: string;
  explanation: string;
  difficulty: string;
  marks: number | null;
}
export interface QuizContent {
  title: string;
  questions: QuizQuestion[];
}
export interface QuestionPaperSection {
  section_title: string;
  instructions: string | null;
  marks_per_question: number;
  questions: string[];
}
export interface QuestionPaperContent {
  title: string;
  total_marks: number;
  duration_minutes: number | null;
  sections: QuestionPaperSection[];
  answer_key: string[] | null;
}
export interface Flashcard {
  front: string;
  back: string;
}
export interface FlashcardsContent {
  cards: Flashcard[];
}
export interface WebResearchContent extends SectionsContent {
  web_sources: { title: string; url: string; snippet: string }[];
}

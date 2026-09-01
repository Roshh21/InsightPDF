import type {
  ChatMessage,
  ChatSession,
  ChatTurnResponse,
  Document,
  DocumentType,
  EvaluationSummary,
  ModelStatus,
  PublicConfig,
  ToolCatalogEntry,
  ToolExecuteResponse,
  ToolExecutionHistoryItem,
  UploadError,
  Workspace,
  WorkspaceDetail,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch (err) {
    throw new ApiError(0, "Could not reach the InsightPDF backend. Is it running?");
  }
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      message = body.message || body.detail || message;
    } catch {
      // ignore parse failure, use default message
    }
    throw new ApiError(res.status, message);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  // --- workspaces ---
  listWorkspaces: () => request<Workspace[]>("/workspaces"),
  createWorkspace: (name: string, description?: string) =>
    request<Workspace>("/workspaces", { method: "POST", body: JSON.stringify({ name, description }) }),
  getWorkspace: (id: string) => request<WorkspaceDetail>(`/workspaces/${id}`),
  deleteWorkspace: (id: string) => request<{ deleted: boolean }>(`/workspaces/${id}`, { method: "DELETE" }),

  // --- documents ---
  listDocuments: (workspaceId: string) => request<Document[]>(`/workspaces/${workspaceId}/documents`),
  getDocument: (id: string) => request<Document>(`/documents/${id}`),
  getDocumentStatus: (id: string) =>
    request<{ id: string; status: string; error_message: string | null }>(`/documents/${id}/status`),
  uploadDocuments: (workspaceId: string, files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<{ documents: Document[]; errors: UploadError[] }>(`/workspaces/${workspaceId}/documents`, {
      method: "POST",
      body: form,
    });
  },
  deleteDocument: (id: string) => request<{ deleted: boolean }>(`/documents/${id}`, { method: "DELETE" }),
  retryDocument: (id: string) => request<Document>(`/documents/${id}/retry`, { method: "POST" }),

  // --- tools ---
  getToolCatalog: (documentType?: DocumentType) =>
    request<ToolCatalogEntry[]>(`/tools/catalog${documentType ? `?document_type=${documentType}` : ""}`),
  executeTool: (workspaceId: string, toolId: string, documentIds: string[], params: Record<string, unknown> = {}) =>
    request<ToolExecuteResponse>(`/workspaces/${workspaceId}/tools/execute`, {
      method: "POST",
      body: JSON.stringify({ tool_id: toolId, document_ids: documentIds, params }),
    }),
  listToolExecutions: (workspaceId: string) =>
    request<ToolExecutionHistoryItem[]>(`/workspaces/${workspaceId}/tools/executions`),
  getToolExecution: (executionId: string) => request<ToolExecuteResponse>(`/tools/executions/${executionId}`),

  // --- chat ---
  createChatSession: (workspaceId: string, documentIds: string[] = [], spoilerLevel?: string, title?: string) =>
    request<ChatSession>(`/workspaces/${workspaceId}/chat/sessions`, {
      method: "POST",
      body: JSON.stringify({ document_ids: documentIds, spoiler_level: spoilerLevel, title }),
    }),
  listChatSessions: (workspaceId: string) => request<ChatSession[]>(`/workspaces/${workspaceId}/chat/sessions`),
  getMessages: (sessionId: string) => request<ChatMessage[]>(`/chat/sessions/${sessionId}/messages`),
  sendMessage: (sessionId: string, message: string) =>
    request<ChatTurnResponse>(`/chat/messages`, {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message }),
    }),

  // --- evaluation / models / config ---
  getEvaluationSummary: () => request<EvaluationSummary>("/evaluation/summary"),
  getModelStatus: (refresh = false) => request<ModelStatus>(`/models/status${refresh ? "?refresh=true" : ""}`),
  getPublicConfig: () => request<PublicConfig>("/config/public"),
};

export async function* streamChat(sessionId: string, message: string): AsyncGenerator<any> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.body) throw new ApiError(res.status, "No response stream from server");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const jsonStr = line.slice(5).trim();
      if (!jsonStr) continue;
      try {
        yield JSON.parse(jsonStr);
      } catch {
        // ignore malformed chunk
      }
    }
  }
}

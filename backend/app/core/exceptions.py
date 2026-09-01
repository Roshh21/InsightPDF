"""
Application-wide exception hierarchy.

Keeping these distinct (rather than raising bare Exception everywhere) is
what lets the API layer return useful error messages instead of stack
traces, and lets the model gateway decide *when* it is safe to fail over
to a local model vs. when a failure indicates an application bug that
should simply be logged and surfaced.
"""
from __future__ import annotations


class InsightPDFError(Exception):
    """Base class for all application errors."""

    http_status: int = 500
    user_message: str = "Something went wrong. Please try again."

    def __init__(self, message: str | None = None, *, user_message: str | None = None):
        super().__init__(message or self.user_message)
        if user_message:
            self.user_message = user_message


# --- Document / ingestion errors -----------------------------------------

class DocumentValidationError(InsightPDFError):
    http_status = 422
    user_message = "This file could not be validated as a usable PDF."


class EmptyDocumentError(DocumentValidationError):
    user_message = "This PDF appears to contain no extractable text (it may be empty or purely a scanned image)."


class ScannedDocumentError(DocumentValidationError):
    user_message = (
        "This looks like a scanned/image-only PDF. Text extraction found little or no "
        "selectable text; OCR is not currently enabled for this document."
    )


class DocumentProcessingError(InsightPDFError):
    http_status = 500
    user_message = "Document processing failed."


class DocumentNotFoundError(InsightPDFError):
    http_status = 404
    user_message = "Document not found."


class WorkspaceNotFoundError(InsightPDFError):
    http_status = 404
    user_message = "Workspace not found."


# --- Vector store errors -----------------------------------------------

class VectorStoreError(InsightPDFError):
    http_status = 503
    user_message = "The vector database is currently unavailable."


class EmbeddingError(InsightPDFError):
    http_status = 503
    user_message = "Failed to generate embeddings for this document."


# --- LLM / model gateway errors ------------------------------------------

class ProviderError(InsightPDFError):
    """Base for all LLM-provider related failures."""

    http_status = 502
    user_message = "The language model provider returned an error."


class ProviderUnavailableError(ProviderError):
    """Connection refused / timeout / 5xx -- safe to fail over."""

    user_message = "The language model provider is temporarily unavailable."


class ProviderQuotaError(ProviderError):
    """429 / quota exceeded -- safe to fail over."""

    user_message = "The language model provider's quota or rate limit was hit."


class ProviderAuthError(ProviderError):
    """Missing/invalid API key or misconfiguration -- safe to fail over."""

    user_message = "The language model provider is not configured correctly."


class ProviderBadRequestError(ProviderError):
    """
    Malformed prompt / application bug. This is NOT failed over -- if the
    request itself is invalid, retrying against a different provider will
    fail the same way and would only hide a real bug.
    """

    http_status = 400
    user_message = "The request to the language model was invalid."


class AllProvidersUnavailableError(ProviderError):
    http_status = 503
    user_message = "All configured free LLM providers are temporarily unavailable. Please try again shortly."


class StructuredOutputError(ProviderError):
    """Every provider tried returned output that couldn't be validated
    against the required schema, even after bounded repair attempts."""

    http_status = 502
    user_message = "The model could not produce a valid structured response after several attempts."


# --- Retrieval / agent errors --------------------------------------------

class InsufficientEvidenceError(InsightPDFError):
    http_status = 200  # Not a hard failure -- surfaced as a graceful answer.
    user_message = "Not enough grounded evidence was found in the selected documents to answer confidently."


class ToolNotFoundError(InsightPDFError):
    http_status = 404
    user_message = "Requested tool is not available for this document type."


class ToolExecutionError(InsightPDFError):
    http_status = 500
    user_message = "Tool execution failed."


class WebSearchError(InsightPDFError):
    http_status = 503
    user_message = "Web research is not available right now."

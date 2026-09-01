"""
SQLAlchemy engine/session management.

Postgres is the intended application database (workspaces, documents,
chat, tool executions, model usage, evaluation results). SQLite is also
supported transparently for a zero-setup local trial -- swap DATABASE_URL
in .env to point at Postgres for anything beyond local experimentation.

The vector store (Chroma/Qdrant) is a *separate* system used only for
embeddings/similarity search -- it is never used as the system of record.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

_connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    connect_args=_connect_args,
)

if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite only allows one writer at a time; the agent graph and the
    # model gateway's usage logging can legitimately open independent
    # short-lived sessions while a request's main session is mid-transaction.
    # WAL mode + a busy timeout lets those writers queue briefly instead of
    # failing outright. Postgres (the recommended production database) has
    # proper MVCC and doesn't need this.
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        # Required for ondelete="CASCADE" (set on every child foreign key in
        # app/models) to actually take effect -- SQLite ignores FK
        # constraints entirely unless this is enabled per-connection.
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create tables if they don't exist.

    For a production deployment you would use Alembic migrations instead;
    `create_all` is used here to keep local setup to a single command.
    """
    from app import models  # noqa: F401  (ensure models are registered)

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for use outside of request handlers (e.g. background tasks)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

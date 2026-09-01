from __future__ import annotations

from typing import List

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class Workspace(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "workspaces"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # passive_deletes=True pairs with ondelete="CASCADE" on each child FK
    # (see documents.workspace_id, chat_sessions.workspace_id,
    # tool_executions.workspace_id): the database performs the cascading
    # delete itself in one statement, rather than SQLAlchemy first loading
    # every child row into memory and issuing individual DELETEs. Besides
    # being far more efficient, this avoids a well-known SQLAlchemy warning
    # ("DELETE statement ... expected to delete N row(s); M were matched")
    # that occurs when the ORM's own delete-orphan cascade races against a
    # database-level ON DELETE CASCADE for the same rows.
    documents: Mapped[List["Document"]] = relationship(  # noqa: F821
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )
    chat_sessions: Mapped[List["ChatSession"]] = relationship(  # noqa: F821
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )
    tool_executions: Mapped[List["ToolExecution"]] = relationship(  # noqa: F821
        cascade="all, delete-orphan", passive_deletes=True
    )

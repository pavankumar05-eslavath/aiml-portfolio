"""Async database engine and session management.

The same code path serves PostgreSQL (``postgresql+asyncpg://``) and SQLite
(``sqlite+aiosqlite://``). SQLite exists so the project can be run and evaluated
without a container runtime; PostgreSQL is the intended deployment target and
what ``docker compose`` provisions. Dialect-specific behaviour is confined to
:func:`_configure_sqlite` and the engine keyword arguments below.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import Settings, get_settings
from app.core.exceptions import DatabaseError
from app.core.logging import get_logger
from app.db.base import Base

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _configure_sqlite(engine: AsyncEngine) -> None:
    """Apply SQLite pragmas needed for concurrent reads during indexing.

    WAL lets the API read while the ingestion script writes; without it the
    default rollback journal makes readers fail with "database is locked".
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=10000")
        finally:
            cursor.close()


def _ensure_sqlite_directory(url: str) -> None:
    """Create the parent directory of a SQLite database file if needed."""
    marker = "sqlite+aiosqlite:///"
    if not url.startswith(marker):
        return
    raw = url[len(marker) :].split("?", 1)[0]
    if not raw or raw == ":memory:":
        return
    Path(raw).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    """Build a new async engine for the configured database URL."""
    settings = settings or get_settings()
    url = settings.database_url
    _ensure_sqlite_directory(url)

    kwargs: dict[str, Any] = {"echo": settings.db_echo, "future": True}
    if url.startswith("sqlite"):
        # aiosqlite drives a single file; a pool of connections adds contention
        # rather than throughput, and NullPool avoids cross-event-loop reuse.
        kwargs["poolclass"] = NullPool
    else:
        kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,  # survive Postgres restarts and idle timeouts
            pool_recycle=1800,
        )

    engine = create_async_engine(url, **kwargs)
    if url.startswith("sqlite"):
        _configure_sqlite(engine)
    return engine


def get_engine() -> AsyncEngine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session that rolls back on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except SQLAlchemyError as exc:
            await session.rollback()
            logger.exception("database error during request")
            raise DatabaseError("The metadata database rejected the operation.") from exc
        except Exception:
            await session.rollback()
            raise


async def init_models(engine: AsyncEngine | None = None) -> None:
    """Create tables that do not exist yet.

    Alembic owns schema evolution (``backend/alembic/``). This helper exists for
    first-run bootstrap, tests and the SQLite developer path, where running a
    migration chain for a single-table schema would be ceremony without benefit.
    """
    engine = engine or get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database schema ready", extra={"context": {"dialect": engine.dialect.name}})


async def check_database(engine: AsyncEngine | None = None) -> None:
    """Raise :class:`DatabaseError` unless a trivial query succeeds."""
    engine = engine or get_engine()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Database unreachable: {type(exc).__name__}") from exc


async def dispose_engine() -> None:
    """Dispose of the engine and reset module state (used on shutdown/tests)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None

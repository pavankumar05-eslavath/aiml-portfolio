"""Alembic environment.

Two details worth noting:

* The URL comes from the application settings, so ``alembic upgrade head`` always
  targets the same database as the app and no credentials live in ``alembic.ini``.
* The project uses an **async** driver (``asyncpg`` / ``aiosqlite``). Alembic's
  migration machinery is synchronous, so migrations run through
  ``connection.run_sync``.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection

from alembic import context
from app.core.config import get_settings
from app.db.base import Base

# Importing the models registers them on Base.metadata, which autogenerate needs.
from app.models import product as _product  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the database URL from application settings."""
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (``alembic upgrade --sql``)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        # SQLite cannot ALTER most things in place; batch mode rewrites the table.
        render_as_batch=_database_url().startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    """Configure and run migrations on an established connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live database using the async engine."""
    from app.db.session import create_engine

    engine = create_engine(get_settings())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

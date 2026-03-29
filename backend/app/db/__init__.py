from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from app.core.config import settings


class Base(DeclarativeBase):
    pass


async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
)

async_session_factory = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """Create all tables. Must be called after all models are imported."""
    from app.models import database  # noqa: F401 — triggers model registration

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def check_db_health() -> bool:
    """Returns True if DB is reachable."""
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception:
        return False

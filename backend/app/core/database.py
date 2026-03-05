"""Database connection and session management."""

import sys
from pathlib import Path
from typing import Optional, Generator
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings


class DatabaseManager:
    """Manages database connections and queries."""

    _instance: Optional["DatabaseManager"] = None

    def __new__(cls) -> "DatabaseManager":
        """Singleton pattern for database manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize database connection."""
        if self._initialized:
            return

        self._engine = None
        self._session_factory = None
        self._initialized = True

    def _get_engine(self):
        """Lazy initialization of database engine."""
        if self._engine is None:
            try:
                self._engine = create_engine(
                    settings.database_connection_string,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                    echo=settings.debug,
                )
                self._session_factory = sessionmaker(
                    bind=self._engine,
                    autocommit=False,
                    autoflush=False,
                )
            except Exception as e:
                print(f"Warning: Could not connect to database: {e}")
                self._engine = None
                self._session_factory = None
        return self._engine

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        engine = self._get_engine()
        if engine is None:
            return False
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session context manager."""
        engine = self._get_engine()
        if engine is None or self._session_factory is None:
            raise RuntimeError("Database not connected")

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        engine = self._get_engine()
        if engine is None:
            raise RuntimeError("Database not connected")

        with engine.connect() as conn:
            result = pd.read_sql(text(query), conn, params=params)
        return result

    def get_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get stock data from database."""
        query = "SELECT * FROM stock_prices WHERE ticker = :ticker"
        params = {"ticker": ticker}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp"

        return self.execute_query(query, params)

    def get_available_tickers(self) -> list[str]:
        """Get list of available stock tickers."""
        query = "SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker"
        df = self.execute_query(query)
        return df["ticker"].tolist() if not df.empty else []

    def save_stock_data(self, df: pd.DataFrame, table_name: str = "stock_prices"):
        """Save stock data to database."""
        engine = self._get_engine()
        if engine is None:
            raise RuntimeError("Database not connected")

        df.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            method="multi",
        )


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> DatabaseManager:
    """Dependency for getting database manager."""
    return db_manager

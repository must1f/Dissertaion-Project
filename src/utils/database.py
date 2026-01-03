"""
Database connection and utilities for TimescaleDB
"""

from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from .config import get_config
from .logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manager for database connections and operations"""

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize database manager

        Args:
            config: Optional Config object. If None, uses global config
        """
        self.config = config or get_config()
        self.engine = None
        self.SessionLocal = None
        self._initialize()

    def _initialize(self):
        """Initialize database engine and session factory"""
        try:
            connection_string = self.config.database.connection_string
            logger.info(f"Connecting to database: {self.config.database.host}:{self.config.database.port}/{self.config.database.database}")

            # Create engine with connection pooling
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,  # Verify connections before using
                echo=False,  # Set to True for SQL debugging
                future=True,
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("Database connection successful")

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions

        Usage:
            with db.get_session() as session:
                session.execute(...)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a SQL query and return results as list of dicts

        Args:
            query: SQL query string
            params: Optional parameters for parameterized queries

        Returns:
            List of dictionaries representing rows
        """
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            if result.returns_rows:
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
            return []

    def read_sql(
        self,
        query: str,
        params: Optional[Dict] = None,
        parse_dates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Execute query and return results as pandas DataFrame

        Args:
            query: SQL query string
            params: Optional query parameters
            parse_dates: List of column names to parse as dates

        Returns:
            DataFrame with query results
        """
        return pd.read_sql(
            text(query),
            self.engine,
            params=params or {},
            parse_dates=parse_dates or []
        )

    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str = "finance",
        if_exists: str = "append",
        method: str = "multi"
    ):
        """
        Write DataFrame to database table

        Args:
            df: DataFrame to write
            table_name: Target table name
            schema: Database schema (default: finance)
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            method: Insert method (None, 'multi')
        """
        try:
            df.to_sql(
                table_name,
                self.engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                method=method,
                chunksize=1000,
            )
            logger.info(f"Successfully wrote {len(df)} rows to {schema}.{table_name}")
        except Exception as e:
            logger.error(f"Failed to write to {schema}.{table_name}: {e}")
            raise

    def bulk_insert_stock_prices(self, df: pd.DataFrame):
        """
        Efficiently bulk insert stock prices using COPY

        Args:
            df: DataFrame with columns: time, ticker, open, high, low, close, volume, adjusted_close
        """
        from io import StringIO

        # Prepare data
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        # Use raw connection for COPY
        with self.engine.raw_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.copy_expert(
                    """
                    COPY finance.stock_prices (time, ticker, open, high, low, close, volume, adjusted_close)
                    FROM STDIN WITH CSV
                    """,
                    buffer
                )
                conn.commit()
                logger.info(f"Bulk inserted {len(df)} stock price records")
            except Exception as e:
                conn.rollback()
                logger.error(f"Bulk insert failed: {e}")
                raise
            finally:
                cursor.close()

    def get_stock_prices(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve stock prices from database

        Args:
            tickers: List of ticker symbols (None for all)
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            DataFrame with stock prices
        """
        query = "SELECT * FROM finance.stock_prices WHERE 1=1"
        params = {}

        if tickers:
            query += " AND ticker = ANY(:tickers)"
            params["tickers"] = tickers

        if start_date:
            query += " AND time >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND time <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY ticker, time"

        return self.read_sql(query, params, parse_dates=["time"])

    def get_features(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve processed features from database

        Args:
            tickers: List of ticker symbols (None for all)
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            DataFrame with features
        """
        query = "SELECT * FROM finance.features WHERE 1=1"
        params = {}

        if tickers:
            query += " AND ticker = ANY(:tickers)"
            params["tickers"] = tickers

        if start_date:
            query += " AND time >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND time <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY ticker, time"

        return self.read_sql(query, params, parse_dates=["time"])

    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get global database manager instance (singleton pattern)"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def reset_db():
    """Reset global database manager (useful for testing)"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
    _db_manager = None

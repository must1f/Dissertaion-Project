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

            # Initialize schema if needed
            self.initialize_schema()

        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            logger.info("Will use Parquet files as fallback for data storage")
            self.engine = None
            self.SessionLocal = None

    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.engine is not None and self.SessionLocal is not None

    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions

        Usage:
            with db.get_session() as session:
                session.execute(...)
        """
        if not self.is_connected():
            raise RuntimeError("Database not connected")
        
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
        method: str = "multi",
        upsert: bool = True
    ):
        """
        Write DataFrame to database table with optional upsert support

        Args:
            df: DataFrame to write
            table_name: Target table name
            schema: Database schema (default: finance)
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            method: Insert method (None, 'multi')
            upsert: If True, use ON CONFLICT DO UPDATE to handle duplicates (default: True)
        """
        try:
            if upsert and table_name in ['stock_prices', 'features']:
                # Use custom upsert for tables with primary keys
                self._upsert_dataframe(df, table_name, schema)
            else:
                # Standard insert
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

    def _upsert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str = "finance"
    ):
        """
        Insert or update DataFrame using PostgreSQL ON CONFLICT clause

        Args:
            df: DataFrame to upsert
            table_name: Target table name
            schema: Database schema
        """
        if df.empty:
            return

        # Deduplicate DataFrame based on primary key (time, ticker)
        # Keep the last occurrence of each (time, ticker) pair
        df_dedup = df.drop_duplicates(subset=['time', 'ticker'], keep='last')

        if len(df_dedup) < len(df):
            logger.info(f"Removed {len(df) - len(df_dedup)} duplicate rows before upsert")

        # Prepare values
        from sqlalchemy import MetaData, Table
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        metadata = MetaData()
        metadata.reflect(bind=self.engine, schema=schema, only=[table_name])
        table = Table(table_name, metadata, schema=schema, autoload_with=self.engine)

        # Convert DataFrame to list of dicts
        records = df_dedup.to_dict('records')

        # Batch insert with ON CONFLICT DO UPDATE
        with self.engine.begin() as conn:
            for i in range(0, len(records), 1000):
                batch = records[i:i+1000]

                # Create insert statement with ON CONFLICT
                stmt = pg_insert(table).values(batch)

                # Determine update columns (all except primary key)
                update_cols = {c.name: stmt.excluded[c.name] for c in table.columns if not c.primary_key}

                # Add ON CONFLICT DO UPDATE
                if update_cols:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['time', 'ticker'],  # Primary key columns
                        set_=update_cols
                    )
                else:
                    # If no non-PK columns, just ignore conflicts
                    stmt = stmt.on_conflict_do_nothing()

                conn.execute(stmt)

        logger.info(f"Upserted {len(df_dedup)} rows to {schema}.{table_name}")

    def bulk_insert_stock_prices(self, df: pd.DataFrame):
        """
        Efficiently insert/update stock prices (handles duplicates)

        Args:
            df: DataFrame with columns: time, ticker, open, high, low, close, volume, adjusted_close
        """
        if not self.is_connected():
            logger.warning("Database not connected, skipping bulk insert")
            return

        # Use the upsert method which handles duplicates
        logger.info(f"Upserting {len(df)} stock price records...")
        self._upsert_dataframe(df, 'stock_prices', 'finance')
        logger.info(f"Stock prices upserted successfully")

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
            DataFrame with stock prices (empty if database not connected)
        """
        if not self.is_connected():
            logger.debug("Database not connected, returning empty DataFrame")
            return pd.DataFrame()
        
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

    def initialize_schema(self):
        """
        Initialize database schema if it doesn't exist
        Creates finance schema and required tables
        """
        if not self.is_connected():
            logger.warning("Database not connected, cannot initialize schema")
            return False

        try:
            logger.info("Initializing database schema...")

            with self.get_session() as session:
                # Create finance schema
                session.execute(text("CREATE SCHEMA IF NOT EXISTS finance"))
                logger.info("Schema 'finance' created/verified")

                # Create stock_prices table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS finance.stock_prices (
                        time TIMESTAMPTZ NOT NULL,
                        ticker TEXT NOT NULL,
                        open DOUBLE PRECISION,
                        high DOUBLE PRECISION,
                        low DOUBLE PRECISION,
                        close DOUBLE PRECISION NOT NULL,
                        volume BIGINT,
                        adjusted_close DOUBLE PRECISION,
                        PRIMARY KEY (time, ticker)
                    )
                """))
                logger.info("Table 'stock_prices' created/verified")

                # Try to create hypertable if TimescaleDB is available
                try:
                    session.execute(text("""
                        SELECT create_hypertable(
                            'finance.stock_prices',
                            'time',
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '7 days'
                        )
                    """))
                    logger.info("Hypertable 'stock_prices' created/verified")
                except Exception as e:
                    logger.debug(f"Could not create hypertable (may already exist or TimescaleDB not available): {e}")

                # Create indexes
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker
                        ON finance.stock_prices (ticker, time DESC)
                """))
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_stock_prices_time
                        ON finance.stock_prices (time DESC)
                """))

                # Create features table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS finance.features (
                        time TIMESTAMPTZ NOT NULL,
                        ticker TEXT NOT NULL,
                        close DOUBLE PRECISION,
                        volume BIGINT,
                        log_return DOUBLE PRECISION,
                        simple_return DOUBLE PRECISION,
                        rolling_volatility_5 DOUBLE PRECISION,
                        rolling_volatility_20 DOUBLE PRECISION,
                        rolling_volatility_60 DOUBLE PRECISION,
                        momentum_5 DOUBLE PRECISION,
                        momentum_20 DOUBLE PRECISION,
                        rsi_14 DOUBLE PRECISION,
                        macd DOUBLE PRECISION,
                        macd_signal DOUBLE PRECISION,
                        bollinger_upper DOUBLE PRECISION,
                        bollinger_lower DOUBLE PRECISION,
                        atr_14 DOUBLE PRECISION,
                        PRIMARY KEY (time, ticker)
                    )
                """))
                logger.info("Table 'features' created/verified")

                # Try to create hypertable for features
                try:
                    session.execute(text("""
                        SELECT create_hypertable(
                            'finance.features',
                            'time',
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '7 days'
                        )
                    """))
                    logger.info("Hypertable 'features' created/verified")
                except Exception as e:
                    logger.debug(f"Could not create hypertable for features: {e}")

                # Create predictions table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS finance.predictions (
                        id SERIAL,
                        time TIMESTAMPTZ NOT NULL,
                        ticker TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        prediction_horizon INTEGER NOT NULL,
                        predicted_close DOUBLE PRECISION NOT NULL,
                        actual_close DOUBLE PRECISION,
                        prediction_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        confidence_lower DOUBLE PRECISION,
                        confidence_upper DOUBLE PRECISION,
                        metadata JSONB,
                        PRIMARY KEY (id)
                    )
                """))
                logger.info("Table 'predictions' created/verified")

                # Create model_metrics table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS finance.model_metrics (
                        id SERIAL PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        model_variant TEXT,
                        training_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        test_mse DOUBLE PRECISION,
                        test_mae DOUBLE PRECISION,
                        test_rmse DOUBLE PRECISION,
                        test_r2 DOUBLE PRECISION,
                        test_mape DOUBLE PRECISION,
                        violation_score DOUBLE PRECISION,
                        epochs INTEGER,
                        training_time_seconds INTEGER,
                        data_loss DOUBLE PRECISION,
                        physics_loss DOUBLE PRECISION,
                        hyperparameters JSONB,
                        metadata JSONB
                    )
                """))
                logger.info("Table 'model_metrics' created/verified")

                session.commit()
                logger.info("✓ Database schema initialization complete")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False

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

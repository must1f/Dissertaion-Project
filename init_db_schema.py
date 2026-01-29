#!/usr/bin/env python3
"""
Database Schema Initialization Script

Run this to manually initialize the database schema if tables don't exist.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.database import get_db
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Initialize database schema"""
    logger.info("=" * 80)
    logger.info("DATABASE SCHEMA INITIALIZATION")
    logger.info("=" * 80)

    try:
        # Get database manager
        db = get_db()

        if not db.is_connected():
            logger.error("Database is not connected!")
            logger.info("Please ensure:")
            logger.info("  1. Docker is running")
            logger.info("  2. TimescaleDB container is up: docker-compose up -d timescaledb")
            logger.info("  3. Database credentials in config are correct")
            return 1

        logger.info("Database connection successful")

        # Initialize schema
        logger.info("\nInitializing schema and tables...")
        success = db.initialize_schema()

        if success:
            logger.info("\n" + "=" * 80)
            logger.info("✓ Database schema initialization successful!")
            logger.info("=" * 80)
            logger.info("\nCreated/verified:")
            logger.info("  • Schema: finance")
            logger.info("  • Tables: stock_prices, features, predictions, model_metrics")
            logger.info("  • Indexes for efficient querying")
            logger.info("  • TimescaleDB hypertables (if available)")
            logger.info("\nYou can now run data fetching and training scripts.")
            return 0
        else:
            logger.error("\n✗ Schema initialization failed")
            logger.info("Check the logs above for details")
            return 1

    except Exception as e:
        logger.exception(f"Error during schema initialization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

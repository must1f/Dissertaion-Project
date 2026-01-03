#!/usr/bin/env python3
"""
Main entry point for PINN Financial Forecasting System

This script provides a unified interface to run the complete pipeline:
- Data fetching
- Model training
- Backtesting
- Web interface

Usage:
    python main.py --help
    python main.py fetch-data
    python main.py train --model pinn
    python main.py backtest
    python main.py web
    python main.py full-pipeline
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger, setup_logger
from src.utils.config import get_config
from src.utils.reproducibility import set_seed, log_system_info

logger = get_logger(__name__)


def fetch_data(args):
    """Fetch and store financial data"""
    from src.data.fetcher import DataFetcher
    from src.data.preprocessor import DataPreprocessor

    logger.info("=" * 80)
    logger.info("FETCHING DATA")
    logger.info("=" * 80)

    config = get_config()

    # Fetch data
    fetcher = DataFetcher(config)

    # Use subset or all tickers
    tickers = config.data.tickers[:args.num_tickers] if args.num_tickers else config.data.tickers

    logger.info(f"Fetching data for {len(tickers)} tickers...")

    df = fetcher.fetch_and_store(
        tickers=tickers,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        force_refresh=args.force_refresh
    )

    if df.empty:
        logger.error("No data fetched!")
        return False

    # Preprocess
    preprocessor = DataPreprocessor(config)

    logger.info("Preprocessing data...")
    df_processed = preprocessor.process_and_store(df)

    logger.info(f"✓ Successfully fetched and processed {len(df_processed)} records")
    return True


def train_model(args):
    """Train a model"""
    from src.training.train import main as train_main

    logger.info("=" * 80)
    logger.info(f"TRAINING {args.model.upper()} MODEL")
    logger.info("=" * 80)

    train_main(args)

    logger.info("✓ Training completed")
    return True


def run_backtest(args):
    """Run backtesting"""
    logger.info("=" * 80)
    logger.info("RUNNING BACKTEST")
    logger.info("=" * 80)

    logger.warning("Backtest module needs to be integrated with trained models")
    logger.info("Please use the web interface for backtesting visualization")

    return True


def launch_web(args):
    """Launch web interface"""
    import subprocess

    logger.info("=" * 80)
    logger.info("LAUNCHING WEB INTERFACE")
    logger.info("=" * 80)

    logger.info("Starting Streamlit server...")
    logger.info("Access the interface at: http://localhost:8501")

    try:
        subprocess.run([
            "streamlit", "run",
            "src/web/app.py",
            "--server.port", str(args.port),
            "--server.address", args.host
        ])
    except KeyboardInterrupt:
        logger.info("\nShutting down web server...")

    return True


def full_pipeline(args):
    """Run the complete pipeline"""
    logger.info("=" * 80)
    logger.info("RUNNING FULL PIPELINE")
    logger.info("=" * 80)

    # 1. Fetch data
    logger.info("\n[1/3] Fetching data...")
    if not fetch_data(args):
        logger.error("Data fetching failed!")
        return False

    # 2. Train models
    logger.info("\n[2/3] Training models...")

    models = ['lstm', 'pinn'] if args.quick else ['lstm', 'gru', 'transformer', 'pinn']

    for model in models:
        args.model = model
        logger.info(f"\nTraining {model.upper()}...")

        try:
            train_model(args)
        except Exception as e:
            logger.error(f"Training {model} failed: {e}")
            continue

    # 3. Launch web interface
    logger.info("\n[3/3] Launching web interface...")
    logger.info("Full pipeline completed! Launching dashboard...")

    launch_web(args)

    return True


def main():
    """Main entry point"""

    # Setup logging
    setup_logger(level="INFO")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='PINN Financial Forecasting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch data for 10 tickers
    python main.py fetch-data --num-tickers 10

    # Train PINN model
    python main.py train --model pinn --epochs 50

    # Launch web interface
    python main.py web

    # Run complete pipeline (quick mode)
    python main.py full-pipeline --quick

⚠️  DISCLAIMER: This is for academic research only - NOT financial advice!
        """
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Fetch data
    fetch_parser = subparsers.add_parser('fetch-data', help='Fetch financial data')
    fetch_parser.add_argument('--num-tickers', type=int, help='Number of tickers (default: all)')
    fetch_parser.add_argument('--force-refresh', action='store_true', help='Force data refresh')

    # Train model
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument(
        '--model',
        type=str,
        default='pinn',
        choices=['lstm', 'gru', 'bilstm', 'transformer', 'pinn'],
        help='Model type'
    )
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')

    # Backtest
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')

    # Web interface
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    web_parser.add_argument('--port', type=int, default=8501, help='Port number')

    # Full pipeline
    pipeline_parser = subparsers.add_parser('full-pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--quick', action='store_true', help='Quick mode (fewer models, less data)')
    pipeline_parser.add_argument('--num-tickers', type=int, default=10, help='Number of tickers')
    pipeline_parser.add_argument('--epochs', type=int, default=20, help='Training epochs')

    args = parser.parse_args()

    # Show help if no command
    if not args.command:
        parser.print_help()
        return

    # Initialize config and set seed
    config = get_config()
    set_seed(config.training.random_seed)

    # Log system info
    logger.info("\n" + "=" * 80)
    logger.info("PINN FINANCIAL FORECASTING SYSTEM")
    logger.info("=" * 80)
    logger.info("⚠️  DISCLAIMER: Academic Research Only - NOT Financial Advice!")
    logger.info("=" * 80 + "\n")

    log_system_info()

    # Route to appropriate handler
    try:
        if args.command == 'fetch-data':
            success = fetch_data(args)
        elif args.command == 'train':
            success = train_model(args)
        elif args.command == 'backtest':
            success = run_backtest(args)
        elif args.command == 'web':
            success = launch_web(args)
        elif args.command == 'full-pipeline':
            success = full_pipeline(args)
        else:
            parser.print_help()
            success = False

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

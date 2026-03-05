#!/usr/bin/env python
"""Run the FastAPI application."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn

from backend.app.config import settings


def main():
    """Run the application."""
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Project root: {settings.project_root}")
    print(f"API docs: http://{settings.host}:{settings.port}/docs")

    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )


if __name__ == "__main__":
    main()

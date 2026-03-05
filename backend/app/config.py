"""Application configuration using Pydantic Settings."""

import os
import warnings
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Settings
    app_name: str = "PINN Financial Forecasting API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, alias="DEBUG")

    # Server Settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # CORS Settings
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
        alias="CORS_ORIGINS"
    )

    # Database Settings
    database_url: Optional[str] = Field(
        default=None,
        alias="DATABASE_URL"
    )
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="financial_data", alias="DB_NAME")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")

    # Project Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
    )

    @property
    def src_path(self) -> Path:
        """Path to the src/ directory with ML code."""
        return self.project_root / "src"

    @property
    def models_path(self) -> Path:
        """Path to saved models."""
        return self.project_root / "models"

    @property
    def results_path(self) -> Path:
        """Path to results directory."""
        return self.project_root / "results"

    results_db_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "results" / "experiments.db",
        alias="RESULTS_DB_PATH",
    )

    @property
    def data_path(self) -> Path:
        """Path to data directory."""
        return self.project_root / "data"

    @property
    def database_connection_string(self) -> str:
        """Get database connection string."""
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    # Model Settings
    default_device: str = Field(default="cpu", alias="DEVICE")
    default_input_dim: int = Field(default=5, alias="INPUT_DIM")

    # Runtime Modes
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")

    # Training Settings
    max_epochs: int = Field(default=100, alias="MAX_EPOCHS")
    batch_size: int = Field(default=32, alias="BATCH_SIZE")
    learning_rate: float = Field(default=0.001, alias="LEARNING_RATE")

    # API Keys (optional)
    alpha_vantage_api_key: Optional[str] = Field(default=None, alias="ALPHA_VANTAGE_API_KEY")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @model_validator(mode='after')
    def validate_no_mock_data_policy(self) -> 'Settings':
        """Enforce NO_MOCK_DATA policy - warn about misconfigurations."""
        issues = []

        # Check for demo mode
        if self.demo_mode:
            issues.append(
                "DEMO_MODE is enabled - this will use mock/synthetic data. "
                "Set DEMO_MODE=false for production use."
            )

        # Check for placeholder credentials
        if self.db_password in ('', 'pinn_password_change_me', 'password', 'changeme'):
            issues.append(
                f"Database password appears to be a placeholder: '{self.db_password}'. "
                "Update DB_PASSWORD in .env for production."
            )

        # Check for placeholder API key
        if self.alpha_vantage_api_key and 'your_' in self.alpha_vantage_api_key.lower():
            issues.append(
                "Alpha Vantage API key appears to be a placeholder. "
                "Update ALPHA_VANTAGE_API_KEY in .env or remove it."
            )

        # Emit warnings for all issues
        for issue in issues:
            warnings.warn(f"[NO_MOCK_DATA POLICY] {issue}", UserWarning, stacklevel=2)

        return self


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()

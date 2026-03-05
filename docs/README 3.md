# PINN Financial Forecasting - Documentation

Welcome to the documentation for the Physics-Informed Neural Network (PINN) Financial Forecasting project.

## Table of Contents

1. [Architecture Overview](./architecture.md) - System design and component interactions
2. [Setup Guide](./setup.md) - Installation and configuration instructions
3. [API Reference](./api-reference.md) - Complete API endpoint documentation
4. [Deployment Guide](./deployment.md) - Production deployment instructions
5. [Model Documentation](./models.md) - PINN model variants and configurations

## Quick Links

- [Backend README](../backend/README.md) - FastAPI backend setup
- [Frontend README](../frontend/README.md) - React frontend setup
- [ML Core README](../src/README.md) - Machine learning core library

## Project Overview

This project implements a full-stack web application for financial time series forecasting using Physics-Informed Neural Networks (PINNs). The system combines deep learning with stochastic differential equations (SDEs) to provide interpretable, physics-constrained predictions.

### Key Features

- **Multiple PINN Variants**: GBM, Ornstein-Uhlenbeck, Black-Scholes, combined models
- **Real-time Training**: WebSocket-based live training progress
- **Monte Carlo Simulation**: Risk analysis with uncertainty quantification
- **Backtesting Platform**: Historical strategy performance evaluation
- **Comprehensive Metrics**: Financial (Sharpe, Sortino) and ML (RMSE, MAPE) metrics

### Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Backend | FastAPI, Python 3.11+, Pydantic |
| ML Core | PyTorch, NumPy, Pandas |
| Database | PostgreSQL + TimescaleDB |
| Charts | Recharts |
| State | Zustand, React Query |

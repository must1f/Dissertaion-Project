# Setup Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ with TimescaleDB extension
- Git

## Quick Start

```bash
# Clone repository
git clone https://github.com/username/Dissertaion-Project.git
cd Dissertaion-Project

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install

# Start both servers
# Terminal 1: Backend
cd backend && python run.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

## Detailed Setup

### 1. Database Setup

```bash
# Install PostgreSQL and TimescaleDB
# macOS
brew install postgresql@14 timescaledb

# Ubuntu
sudo apt install postgresql-14 postgresql-14-timescaledb

# Create database
psql -U postgres
CREATE DATABASE pinn_forecasting;
\c pinn_forecasting
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

### 2. Environment Configuration

Create `.env` files:

**backend/.env:**
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/pinn_forecasting

# Paths
MODELS_DIR=../Models
RESULTS_DIR=../results
SRC_DIR=../src

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Optional
CORS_ORIGINS=http://localhost:5173
```

**frontend/.env:**
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install ML core as editable package
pip install -e ../

# Initialize database (if needed)
python ../init_db_schema.py

# Run server
python run.py
```

Verify: Open http://localhost:8000/docs

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Verify: Open http://localhost:5173

### 5. ML Core Setup (Optional)

If you want to use the ML core independently:

```bash
cd src

# Install as package
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Development Tools

### Backend Development

```bash
# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v

# Type checking
mypy app/

# Linting
ruff check app/
```

### Frontend Development

```bash
# Development server
npm run dev

# Type checking
npm run typecheck

# Linting
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## Troubleshooting

### Backend Issues

**ImportError: No module named 'src'**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Dissertaion-Project"
```

**Database connection error**
```bash
# Check PostgreSQL is running
pg_isready

# Verify connection string
psql $DATABASE_URL
```

### Frontend Issues

**Module not found errors**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**API calls failing**
```bash
# Check Vite proxy config
# Verify backend is running on port 8000
curl http://localhost:8000/api/models/
```

### Model Loading Issues

**Model weights not found**
```bash
# Check Models directory exists
ls -la Models/

# Verify model files
ls -la Models/*.pth
```

## IDE Setup

### VS Code

Recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- ESLint (dbaeumer.vscode-eslint)
- Tailwind CSS IntelliSense (bradlc.vscode-tailwindcss)

**.vscode/settings.json:**
```json
{
  "python.defaultInterpreterPath": "./backend/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

### PyCharm

1. Mark `src/` as Sources Root
2. Mark `backend/` as Sources Root
3. Configure Python interpreter from `backend/venv/`

## Next Steps

1. [API Reference](./api-reference.md) - Explore API endpoints
2. [Architecture](./architecture.md) - Understand system design
3. [Deployment](./deployment.md) - Deploy to production

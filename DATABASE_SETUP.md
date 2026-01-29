# Database Setup and Initialization

## Issue Resolved

**Error**: `relation "finance.stock_prices" does not exist`

**Root Cause**: The database container was starting without the schema and tables initialized.

## What Was Fixed

### 1. Created Database Initialization Script
- **File**: `docker/init-db.sql`
- **Purpose**: Automatically creates schema and tables when database container starts
- **Features**:
  - Creates `finance` schema
  - Creates tables: `stock_prices`, `features`, `predictions`, `model_metrics`, `training_history`
  - Converts tables to TimescaleDB hypertables for time-series optimization
  - Creates indexes for efficient querying
  - Sets up continuous aggregates (daily/weekly OHLCV)
  - Configures data retention and compression policies
  - Creates helper functions for common operations

### 2. Added Auto-Initialization to DatabaseManager
- **File**: `src/utils/database.py`
- **Changes**:
  - Added `initialize_schema()` method
  - Automatically calls schema initialization on database connection
  - Creates schema and tables if they don't exist
  - Gracefully handles existing tables

### 3. Enhanced Quick Demo in run.sh
- **Changes**:
  - Now includes Docker database setup as Step 1
  - Automatically recreates database container to ensure init script runs
  - Waits for database initialization (10 seconds)
  - Verifies database is ready before proceeding
  - Falls back to local storage if Docker unavailable

### 4. Created Manual Initialization Script
- **File**: `init_db_schema.py`
- **Purpose**: Manually initialize schema if needed
- **Usage**: `python3 init_db_schema.py` (with venv activated)

## Database Schema

### Tables Created

#### 1. `finance.stock_prices` (Hypertable)
```sql
time TIMESTAMPTZ NOT NULL
ticker TEXT NOT NULL
open DOUBLE PRECISION
high DOUBLE PRECISION
low DOUBLE PRECISION
close DOUBLE PRECISION NOT NULL
volume BIGINT
adjusted_close DOUBLE PRECISION
PRIMARY KEY (time, ticker)
```

#### 2. `finance.features` (Hypertable)
```sql
time TIMESTAMPTZ NOT NULL
ticker TEXT NOT NULL
close DOUBLE PRECISION
volume BIGINT
log_return DOUBLE PRECISION
simple_return DOUBLE PRECISION
rolling_volatility_5 DOUBLE PRECISION
rolling_volatility_20 DOUBLE PRECISION
momentum_5, momentum_20 DOUBLE PRECISION
rsi_14 DOUBLE PRECISION
macd, macd_signal DOUBLE PRECISION
bollinger_upper, bollinger_lower DOUBLE PRECISION
atr_14 DOUBLE PRECISION
PRIMARY KEY (time, ticker)
```

#### 3. `finance.predictions`
```sql
id SERIAL
time TIMESTAMPTZ NOT NULL
ticker TEXT NOT NULL
model_name TEXT NOT NULL
prediction_horizon INTEGER NOT NULL
predicted_close DOUBLE PRECISION NOT NULL
actual_close DOUBLE PRECISION
prediction_date TIMESTAMPTZ NOT NULL DEFAULT NOW()
confidence_lower, confidence_upper DOUBLE PRECISION
metadata JSONB
```

#### 4. `finance.model_metrics`
```sql
id SERIAL PRIMARY KEY
model_name TEXT NOT NULL
model_variant TEXT
training_date TIMESTAMPTZ NOT NULL DEFAULT NOW()
test_mse, test_mae, test_rmse, test_r2, test_mape DOUBLE PRECISION
violation_score DOUBLE PRECISION
epochs INTEGER
training_time_seconds INTEGER
data_loss, physics_loss DOUBLE PRECISION
hyperparameters JSONB
metadata JSONB
```

#### 5. `finance.training_history`
```sql
id SERIAL PRIMARY KEY
model_name TEXT NOT NULL
training_session_id TEXT NOT NULL
epoch INTEGER NOT NULL
train_loss, val_loss DOUBLE PRECISION
train_data_loss, train_physics_loss DOUBLE PRECISION
learning_rate DOUBLE PRECISION
timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
```

### TimescaleDB Features Enabled

1. **Hypertables**: Time-series optimized tables with automatic partitioning
2. **Continuous Aggregates**: Pre-computed daily/weekly OHLCV views
3. **Compression**: Automatic compression of data older than 7 days
4. **Retention Policies**: Automatic deletion of data older than 1-2 years
5. **Indexes**: Optimized for ticker + time queries

## How to Use

### Option 1: Run Quick Demo (Recommended)
```bash
./run.sh
# Select option 1: Quick Demo (DB + Fetch + Train + UI - Everything!)
```

This will:
1. Start/recreate TimescaleDB container with schema
2. Wait for initialization
3. Fetch and store data
4. Train model
5. Launch web interface

### Option 2: Manual Database Setup
```bash
# Start database
docker-compose up -d timescaledb

# Wait 10 seconds for initialization
sleep 10

# Verify schema (optional)
python3 init_db_schema.py
```

### Option 3: Restart Existing Database
If you have an existing database without the schema:
```bash
# Stop and remove container
docker stop pinn-timescaledb
docker rm pinn-timescaledb

# Remove old data (optional - will delete all data!)
# rm -rf postgres-data

# Start fresh
docker-compose up -d timescaledb
sleep 10
```

## Verification

Check if tables exist:
```bash
docker exec -it pinn-timescaledb psql -U pinn_user -d pinn_finance -c "\dt finance.*"
```

Expected output:
```
              List of relations
 Schema  |       Name        | Type  |   Owner
---------+-------------------+-------+-----------
 finance | features          | table | pinn_user
 finance | model_metrics     | table | pinn_user
 finance | predictions       | table | pinn_user
 finance | stock_prices      | table | pinn_user
 finance | training_history  | table | pinn_user
```

Check hypertables:
```bash
docker exec -it pinn-timescaledb psql -U pinn_user -d pinn_finance -c "SELECT * FROM timescaledb_information.hypertables;"
```

## Troubleshooting

### Error: "relation does not exist"
**Solution**: Run `python3 init_db_schema.py` to initialize schema

### Error: "Database connection failed"
**Check**:
1. Docker is running: `docker ps | grep timescale`
2. Container is healthy: `docker inspect pinn-timescaledb | grep Health`
3. Port 5432 is not in use: `lsof -i :5432`

### Error: "permission denied"
**Solution**: Check database credentials in config match docker-compose.yml

### Database is slow
**Check**: Compression and retention policies are working
```sql
SELECT * FROM timescaledb_information.compression_settings;
SELECT * FROM timescaledb_information.jobs;
```

## Benefits of This Setup

1. **Automatic Initialization**: Schema created on first container start
2. **Time-Series Optimization**: TimescaleDB hypertables for better performance
3. **Data Management**: Automatic compression and retention
4. **Scalability**: Efficient handling of large time-series datasets
5. **Analytics**: Pre-computed aggregates for faster queries
6. **Fallback**: System works without database (local Parquet files)

## Next Steps

After database is initialized:
1. Run data fetching: `./run.sh` → Option 2
2. Train models: `./run.sh` → Option 4
3. View predictions: `./run.sh` → Option 5

---

**Note**: The database initialization happens automatically when using the Quick Demo (Option 1) or when connecting to the database via Python. No manual intervention needed for normal usage.

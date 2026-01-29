# Database Issues Fixed

## Issues Resolved

### Issue 1: '_ConnectionFairy' object does not support the context manager protocol

**Error Message:**
```
Failed to store in database: '_ConnectionFairy' object does not support the context manager protocol
```

**Root Cause:**
The `bulk_insert_stock_prices()` method in `database.py` was using `with self.engine.raw_connection() as conn:` but SQLAlchemy's raw_connection() doesn't return a proper context manager.

**Fix Applied:**
Updated `src/utils/database.py` line 189-206 to:
```python
# Use raw connection for COPY
conn = self.engine.raw_connection()
cursor = conn.cursor()
try:
    cursor.copy_expert(...)
    conn.commit()
except Exception as e:
    conn.rollback()
    raise
finally:
    cursor.close()
    conn.close()  # Explicitly close connection
```

### Issue 2: column "rolling_volatility_60" does not exist

**Error Message:**
```
(psycopg2.errors.UndefinedColumn) column "rolling_volatility_60" of relation "features" does not exist
```

**Root Cause:**
The preprocessor calculates `rolling_volatility_60` but the database schema didn't include this column in the `finance.features` table.

**Fixes Applied:**

1. **Added column to SQL initialization script** (`docker/init-db.sql`):
   ```sql
   CREATE TABLE IF NOT EXISTS finance.features (
       ...
       rolling_volatility_5 DOUBLE PRECISION,
       rolling_volatility_20 DOUBLE PRECISION,
       rolling_volatility_60 DOUBLE PRECISION,  -- Added this line
       ...
   );
   ```

2. **Updated Python schema initialization** (`src/utils/database.py`):
   ```python
   rolling_volatility_5 DOUBLE PRECISION,
   rolling_volatility_20 DOUBLE PRECISION,
   rolling_volatility_60 DOUBLE PRECISION,  -- Added
   ```

3. **Added column to existing database**:
   ```sql
   ALTER TABLE finance.features
   ADD COLUMN IF NOT EXISTS rolling_volatility_60 DOUBLE PRECISION;
   ```

## Verification

Database schema now includes the `rolling_volatility_60` column:

```
\d finance.features

 rolling_volatility_5  | double precision
 rolling_volatility_20 | double precision
 rolling_volatility_60 | double precision  ✓ ADDED
 momentum_5            | double precision
```

## Files Modified

1. `src/utils/database.py`:
   - Fixed raw_connection context manager usage
   - Added rolling_volatility_60 to schema initialization

2. `docker/init-db.sql`:
   - Added rolling_volatility_60 column definition

3. Database:
   - Added missing column via ALTER TABLE

## Testing

The fixes can be verified by:

1. **Test database connection**:
   ```bash
   source venv/bin/activate
   python3 init_db_schema.py
   ```

2. **Test data insertion**:
   ```bash
   python3 -m src.training.train --model pinn --epochs 20
   ```

3. **Verify table structure**:
   ```bash
   docker exec pinn-timescaledb psql -U pinn_user -d pinn_finance -c "\d finance.features"
   ```

## Status

✅ Both issues resolved
✅ Database schema updated
✅ Column added to existing table
✅ Raw connection handling fixed

The system should now be able to:
- Store stock prices in database without connection errors
- Store processed features including rolling_volatility_60
- Handle database operations correctly

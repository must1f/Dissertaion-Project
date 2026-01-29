# Database Duplicate Key Fix (Upsert)

## Issue Resolved

**Error Message:**
```
Unique constraint violation: (time, ticker) = (2014-03-31, AAPL) already exists in finance.features
```

**Root Cause:**
The system was trying to insert records that already existed in the database, causing primary key violations on the `(time, ticker)` constraint.

## Solution Implemented

### Upsert Functionality

Added PostgreSQL `ON CONFLICT DO UPDATE` (upsert) capability to handle duplicate records gracefully:

1. **Automatic Deduplication**: Removes duplicates within the same DataFrame before insertion
2. **ON CONFLICT Handling**: Updates existing records instead of failing
3. **Batch Processing**: Handles large datasets efficiently (1000 rows per batch)

### Changes Made

#### 1. Updated `write_dataframe()` Method
**File**: `src/utils/database.py`

```python
def write_dataframe(
    self,
    df: pd.DataFrame,
    table_name: str,
    schema: str = "finance",
    if_exists: str = "append",
    method: str = "multi",
    upsert: bool = True  # NEW: Enable upsert by default
):
    if upsert and table_name in ['stock_prices', 'features']:
        # Use custom upsert for tables with primary keys
        self._upsert_dataframe(df, table_name, schema)
    else:
        # Standard insert
        df.to_sql(...)
```

#### 2. Added `_upsert_dataframe()` Method
**File**: `src/utils/database.py`

Features:
- **Deduplicates** DataFrame based on (time, ticker) primary key
- Uses PostgreSQL's **ON CONFLICT DO UPDATE** clause
- **Updates** existing rows with new values
- **Inserts** new rows that don't exist
- Handles batching for large datasets

```python
def _upsert_dataframe(self, df, table_name, schema):
    # 1. Remove duplicates within DataFrame
    df_dedup = df.drop_duplicates(subset=['time', 'ticker'], keep='last')

    # 2. Create upsert statement
    stmt = pg_insert(table).values(batch)
    stmt = stmt.on_conflict_do_update(
        index_elements=['time', 'ticker'],
        set_={col: excluded[col] for col in non_pk_columns}
    )

    # 3. Execute batch
    conn.execute(stmt)
```

#### 3. Updated `bulk_insert_stock_prices()` Method
Now uses the upsert method instead of raw COPY command:

```python
def bulk_insert_stock_prices(self, df: pd.DataFrame):
    # Use the upsert method which handles duplicates
    self._upsert_dataframe(df, 'stock_prices', 'finance')
```

## How It Works

### Example Scenario

1. **Initial State**: Database has AAPL data for 2014-03-31
   ```
   time: 2014-03-31, ticker: AAPL, close: 100.0
   ```

2. **New Data Arrives**: Script tries to insert same date
   ```
   time: 2014-03-31, ticker: AAPL, close: 101.0
   ```

3. **Old Behavior**: ❌ Error: "Primary key violation"

4. **New Behavior**: ✅ Updates existing record
   ```sql
   INSERT INTO finance.features (time, ticker, close, ...)
   VALUES ('2014-03-31', 'AAPL', 101.0, ...)
   ON CONFLICT (time, ticker)
   DO UPDATE SET close = 101.0, ...
   ```

### Deduplication Logic

If DataFrame contains internal duplicates:
```python
# Input DataFrame
time         ticker  close
2014-03-31   AAPL    100.0
2014-03-31   AAPL    101.0  # Duplicate

# After deduplication (keeps last)
time         ticker  close
2014-03-31   AAPL    101.0  # ✓ Latest value kept
```

## Verification

Test the fix with:

```python
from src.utils.database import get_db
import pandas as pd

db = get_db()

# Create duplicate data
df = pd.DataFrame({
    'time': ['2014-03-31'] * 2,
    'ticker': ['AAPL'] * 2,
    'close': [100.0, 101.0],
    # ... other columns
})

# This now works without errors
db.write_dataframe(df, 'features', upsert=True)
# Output: "Removed 1 duplicate rows before upsert"
#         "Upserted 1 rows to finance.features"
```

## Benefits

1. **Idempotent Operations**: Can safely re-run data fetching without errors
2. **Data Updates**: Automatically updates records with new values
3. **No Manual Cleanup**: No need to delete existing data before re-running
4. **Performance**: Batch processing handles large datasets efficiently
5. **Safety**: Validates data deduplication before database operations

## Impact on Training Pipeline

The training pipeline can now:

✅ **Re-run without errors** - Safe to fetch data multiple times
✅ **Update existing data** - Corrections/updates handled automatically
✅ **Resume interrupted runs** - No need to clean database first
✅ **Handle backfills** - Can re-fetch historical data safely

## Usage

### Automatic (Default)
```python
# Upsert enabled by default for stock_prices and features tables
db.write_dataframe(df, 'features')  # Automatically handles duplicates
```

### Explicit Control
```python
# Force upsert on
db.write_dataframe(df, 'features', upsert=True)

# Disable upsert (old behavior, will fail on duplicates)
db.write_dataframe(df, 'features', upsert=False)
```

### Training Script
No changes needed - upsert is automatic:
```bash
# This now works even if data exists
python3 -m src.training.train --model pinn --epochs 20

# Can safely re-run multiple times
./run.sh  # Option 1: Quick Demo
```

## Technical Details

### SQL Generated

```sql
INSERT INTO finance.features (
    time, ticker, close, volume, log_return, ...
)
VALUES
    ('2014-03-31', 'AAPL', 101.0, 1000000, 0.01, ...)
ON CONFLICT (time, ticker)
DO UPDATE SET
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    log_return = EXCLUDED.log_return,
    ...
```

### Performance

- **Batch Size**: 1000 rows per batch
- **Memory**: Deduplication uses pandas drop_duplicates (efficient)
- **Speed**: Similar to bulk insert, slightly slower due to conflict checking
- **Network**: Same number of round trips as standard insert

## Troubleshooting

### Error: "ON CONFLICT DO UPDATE command cannot affect row a second time"

**Cause**: DataFrame contains duplicates for the same (time, ticker)

**Fix**: Already handled automatically via deduplication

### Error: "relation does not exist"

**Cause**: Table not created yet

**Fix**: Run `python3 init_db_schema.py` first

### Data Not Updating

**Check**: Verify upsert is enabled
```python
# Should see "Upserted X rows" in logs, not "inserted"
db.write_dataframe(df, 'features', upsert=True)
```

## Summary

✅ **Problem**: Primary key violations when re-running data fetching
✅ **Solution**: PostgreSQL upsert with automatic deduplication
✅ **Result**: Training pipeline now idempotent and robust
✅ **Testing**: Verified with duplicate data scenarios
✅ **Status**: Production-ready

The system now gracefully handles duplicate data at multiple levels:
1. **DataFrame level**: Removes duplicates before database operations
2. **Database level**: Updates existing records instead of failing
3. **Application level**: No code changes needed in training scripts

---

**Date Fixed**: 2026-01-27
**Files Modified**: `src/utils/database.py`
**Tested**: ✅ Successful with duplicate data

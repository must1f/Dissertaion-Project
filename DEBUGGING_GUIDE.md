# Debugging Guide for PINN Financial Forecasting

This guide explains how to use the enhanced debugging features in the setup and run scripts.

## Features Added

### 1. Automatic Logging
- Every run creates a timestamped log file
- **Setup logs**: `setup_YYYYMMDD_HHMMSS.log`
- **Run logs**: `run_YYYYMMDD_HHMMSS.log`
- All output (stdout and stderr) is captured

### 2. Timestamped Output
Every message includes a timestamp:
```
[2026-01-26 14:30:45] [DEBUG] Starting data fetch...
[2026-01-26 14:31:12] ✓ Data fetched successfully (27s)
```

### 3. Debug Mode
Enable verbose debugging with the `DEBUG` environment variable:

```bash
# Enable debug mode for setup
DEBUG=1 ./setup.sh

# Enable debug mode for running
DEBUG=1 ./run.sh
```

In debug mode, you'll see:
- Every command before it executes
- Full command output
- Internal script operations
- Variable values
- File system operations

### 4. Progress Tracking

#### Setup Script Stages
1. **Stage 1/7**: Checking Python Installation
2. **Stage 2/7**: Setting Up Virtual Environment
3. **Stage 3/7**: Upgrading pip
4. **Stage 4/7**: Installing Python Dependencies
5. **Stage 5/7**: Configuring Environment Variables
6. **Stage 6/7**: Checking Docker Installation
7. **Stage 7/7**: Creating Project Directories

#### Run Script Operations
- Real-time progress for each operation
- Time tracking for long-running tasks
- Step-by-step execution flow

### 5. Enhanced Error Messages
Errors now include:
- Timestamp of failure
- Detailed context about what failed
- File paths and system information
- Helpful debugging hints

### 6. Execution Timing
Automatically tracks and reports:
- Time per operation
- Total execution time
- Average time per model (when training multiple)

## Usage Examples

### Basic Usage (Standard Logging)
```bash
# Setup with automatic logging
./setup.sh

# Run with automatic logging
./run.sh
```

### Debug Mode (Verbose Output)
```bash
# See every command and internal operation
DEBUG=1 ./setup.sh
DEBUG=1 ./run.sh
```

### Review Logs Later
```bash
# View the most recent setup log
tail -f setup_*.log | tail -1

# View the most recent run log
tail -f run_*.log | tail -1

# Search logs for errors
grep -i error *.log

# Search logs for specific stage
grep "Stage 4/7" setup_*.log
```

## Debug Output Examples

### Normal Mode
```
========================================
[2026-01-26 14:30:00] STAGE 1/7: Checking Python Installation
========================================

Checking Python version...
[2026-01-26 14:30:00] ✓ Python 3.10 detected
```

### Debug Mode
```
========================================
[2026-01-26 14:30:00] STAGE 1/7: Checking Python Installation
========================================

[2026-01-26 14:30:00] [DEBUG] Searching for Python 3 executable...
Checking Python version...
[2026-01-26 14:30:00] [DEBUG] Python 3 found at: /usr/local/bin/python3
[2026-01-26 14:30:00] [DEBUG] Detected Python version: 3.10
[2026-01-26 14:30:00] [DEBUG] Full version info: Python 3.10.12
[2026-01-26 14:30:00] [DEBUG] Required minimum version: 3.10
[2026-01-26 14:30:00] ✓ Python 3.10 detected
[2026-01-26 14:30:00] [DEBUG] Python version check passed
```

## Debugging Common Issues

### Issue: Setup hangs during pip install
**Solution**: Check the log file to see which package is being installed:
```bash
# In another terminal
tail -f setup_*.log | grep "Collecting\|Installing"
```

### Issue: Training appears stuck
**Solution**: Enable debug mode to see epoch-by-epoch progress:
```bash
DEBUG=1 ./run.sh
# Select option 3 or 4
```

### Issue: Need to understand script flow
**Solution**: Review the debug messages in the log:
```bash
grep "\[DEBUG\]" run_*.log
```

## Log File Management

### Automatic Cleanup (Recommended)
Add to your crontab or run manually:
```bash
# Keep only logs from last 7 days
find . -name "setup_*.log" -mtime +7 -delete
find . -name "run_*.log" -mtime +7 -delete
```

### Manual Cleanup
```bash
# Remove all old logs
rm setup_*.log run_*.log

# Keep only the 5 most recent logs
ls -t setup_*.log | tail -n +6 | xargs rm
ls -t run_*.log | tail -n +6 | xargs rm
```

## Performance Monitoring

### Track Script Performance
```bash
# See execution times for each stage
grep "completed in\|successfully (" setup_*.log

# See model training times
grep "trained successfully" run_*.log
```

### Example Output
```
[2026-01-26 14:35:12] ✓ Data fetched successfully (27s)
[2026-01-26 14:42:18] ✓ PINN model trained (426s)
[2026-01-26 14:42:20] Total demo preparation time: 453 seconds
```

## Tips

1. **Always check logs first** when debugging issues
2. **Use DEBUG=1** when reporting bugs or issues
3. **Keep recent logs** for troubleshooting
4. **Monitor disk space** if running many experiments
5. **Grep logs** to find specific events quickly

## Color Coding

The scripts use color-coded output:
- 🟢 **Green**: Success messages
- 🟡 **Yellow**: Warnings and progress
- 🔴 **Red**: Errors
- 🔵 **Blue**: Information
- 🔷 **Cyan**: Debug messages
- 🟣 **Magenta**: Section headers

## Advanced Usage

### Redirect Only Errors
```bash
./setup.sh 2> setup_errors.log
```

### Run in Background
```bash
# Run and continue using terminal
./run.sh > output.log 2>&1 &

# Monitor progress
tail -f output.log
```

### Combine Multiple Operations
```bash
# Setup, then immediately run demo
./setup.sh && ./run.sh <<< "1"
```

## Troubleshooting

If you encounter issues with the enhanced scripts:

1. **Check file permissions**: `chmod +x setup.sh run.sh`
2. **Verify log file creation**: `ls -la *.log`
3. **Test debug mode**: `DEBUG=1 ./setup.sh`
4. **Review timestamps**: Ensure system clock is correct
5. **Check disk space**: `df -h .`

## Support

For issues or questions:
1. Check the log files first
2. Run with DEBUG=1 to get detailed output
3. Review this guide
4. Open an issue with log excerpts (remove sensitive data)

#!/bin/bash

# Physics-Informed Neural Network (PINN) Financial Forecasting - Run Script
# ==========================================================================

set -e  # Exit on error

# Enable debug mode if DEBUG=1 is set
if [ "${DEBUG}" = "1" ]; then
    set -x  # Print commands as they execute
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Logging setup
LOGFILE="run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE")
exec 2>&1

# Helper function for timestamps
timestamp() {
    date "+[%Y-%m-%d %H:%M:%S]"
}

# Helper function for debug messages
debug_msg() {
    echo -e "${CYAN}$(timestamp) [DEBUG] $1${NC}"
}

# Helper function for section headers
section_header() {
    echo ""
    echo "========================================"
    echo -e "${MAGENTA}$(timestamp) $1${NC}"
    echo "========================================"
    echo ""
}

# Normalize/sync trained model artifacts so the web API can discover them
sync_models_for_webapp() {
    section_header "Syncing Trained Models for Web App"

    # Normalize directory name (some runs created capitalized 'Models')
    if [ -d "Models" ] && [ ! -d "models" ]; then
        debug_msg "Renaming Models -> models for backend compatibility"
        mv Models models
    elif [ -d "Models" ] && [ -d "models" ]; then
        debug_msg "Merging Models into models/"
        rsync -a Models/ models/ 2>/dev/null || cp -R Models/. models/
    fi

    mkdir -p models models/stacked_pinn

    # Copy checkpoints/histories from checkpoints/ (used by some scripts)
    if [ -d "checkpoints" ]; then
        find checkpoints -maxdepth 2 -type f \( -name "*_best.pt" -o -name "*_best.pth" -o -name "*_history.json" \) -print0 \
        | while IFS= read -r -d '' file; do
            dest="models/$(basename "$file")"
            if [ ! -f "$dest" ]; then
                debug_msg "Syncing $(basename "$file") from checkpoints to models/"
                cp "$file" "$dest"
            fi
        done
    fi

    # Ensure stacked/residual variants land in models/stacked_pinn
    if [ -d "checkpoints/stacked_pinn" ]; then
        cp checkpoints/stacked_pinn/*_best.pt models/stacked_pinn/ 2>/dev/null || true
        cp checkpoints/stacked_pinn/*_history.json models/stacked_pinn/ 2>/dev/null || true
    fi

    MODEL_COUNT=$(find models -maxdepth 2 -type f \( -name "*_best.pt" -o -name "*_best.pth" \) | wc -l | tr -d ' ')
    echo -e "${GREEN}$(timestamp) ✓ Synced ${MODEL_COUNT} checkpoint(s) into models/ for the web app${NC}"
    debug_msg "Model sync complete; models directory=$(pwd)/models"
}

# Normalize/sync trained model artifacts so the web API can discover them
sync_models_for_webapp() {
    section_header "Syncing Trained Models for Web App"

    # Normalize directory name (some runs created capitalized 'Models')
    if [ -d "Models" ] && [ ! -d "models" ]; then
        debug_msg "Renaming Models -> models for backend compatibility"
        mv Models models
    elif [ -d "Models" ] && [ -d "models" ]; then
        debug_msg "Merging Models into models/"
        rsync -a Models/ models/ 2>/dev/null || cp -R Models/. models/
    fi

    mkdir -p models models/stacked_pinn

    # Copy checkpoints/histories from checkpoints/ (used by some scripts)
    if [ -d "checkpoints" ]; then
        find checkpoints -maxdepth 2 -type f \( -name "*_best.pt" -o -name "*_best.pth" -o -name "*_history.json" \) -print0 \
        | while IFS= read -r -d '' file; do
            dest="models/$(basename "$file")"
            if [ ! -f "$dest" ]; then
                debug_msg "Syncing $(basename "$file") from checkpoints to models/"
                cp "$file" "$dest"
            fi
        done
    fi

    # Ensure stacked/residual variants land in models/stacked_pinn
    if [ -d "checkpoints/stacked_pinn" ]; then
        cp checkpoints/stacked_pinn/*_best.pt models/stacked_pinn/ 2>/dev/null || true
        cp checkpoints/stacked_pinn/*_history.json models/stacked_pinn/ 2>/dev/null || true
    fi

    MODEL_COUNT=$(find models -maxdepth 2 -type f \( -name "*_best.pt" -o -name "*_best.pth" \) | wc -l | tr -d ' ')
    echo -e "${GREEN}$(timestamp) ✓ Synced ${MODEL_COUNT} checkpoint(s) into models/ for the web app${NC}"
    debug_msg "Model sync complete; models directory=$(pwd)/models"
}

# Initial setup
debug_msg "Run script started"
debug_msg "Logging to: $LOGFILE"
debug_msg "Current directory: $(pwd)"
debug_msg "User: $(whoami)"

# Check if virtual environment is activated
debug_msg "Checking virtual environment..."
if [ -z "$VIRTUAL_ENV" ]; then
    debug_msg "Virtual environment not activated"
    if [ -d "venv" ]; then
        echo -e "${YELLOW}$(timestamp) Activating virtual environment...${NC}"
        debug_msg "Found venv directory, activating..."
        source venv/bin/activate
        debug_msg "Virtual environment activated: $VIRTUAL_ENV"
    else
        echo -e "${RED}$(timestamp) [ERROR] Virtual environment not found. Run ./setup.sh first${NC}"
        debug_msg "No venv directory found at: $(pwd)/venv"
        exit 1
    fi
else
    debug_msg "Virtual environment already activated: $VIRTUAL_ENV"
fi

debug_msg "Python executable: $(which python3)"
debug_msg "Python version: $(python3 --version)"

# Function to display menu
show_menu() {
    echo ""
    echo "========================================"
    echo "PINN Financial Forecasting System"
    echo "========================================"
    echo ""
    echo "Select an option:"
    echo ""
    echo "1) Quick Demo (DB + Fetch + Train + UI - Everything!)"
    echo "2) Fetch Financial Data Only"
    echo "3) Train PINN Model"
    echo "4) Train All Models (LSTM, GRU, Transformer, PINN)"
    echo "5) Launch Web Interface"
    echo "6) Run Tests"
    echo "7) Start Database (Docker)"
    echo "8) Full Docker Stack"
    echo "9) Complete Pipeline (DB + Fetch + Train All + Store)"
    echo "10) Systematic PINN Physics Comparison (6 variants)"
    echo "11) Full Model Pipeline (All 13 Models + Evaluation + Dashboard)"
    echo "12) Train Baseline Models (LSTM, GRU, BiLSTM, Attention, Transformer)"
    echo "13) Compute Financial Metrics (All Models)"
    echo "14) View Metrics in Terminal"
    echo "15) Evaluate Stacked/Residual PINN (Generate Metrics)"
    echo "16) Launch Backtesting Dashboard"
    echo "17) Recompute All Metrics (Fix Consistency)"
    echo "18) Launch Training Visualizations Dashboard"
    echo "19) Generate Dissertation Figures"
    echo "20) Research-Grade Training (Standalone Script)"
    echo "21) Launch Full Web App (Backend + Frontend)"
    echo "22) Exit"
    echo ""
}

# Function to run quick demo
quick_demo() {
    section_header "Quick Demo Mode"
    debug_msg "Starting quick demo workflow"
    DEMO_START=$SECONDS
    echo -e "${BLUE}$(timestamp) Starting Quick Demo...${NC}"
    echo ""
    echo "This demo will:"
    echo "  1. Start TimescaleDB (Docker)"
    echo "  2. Fetch financial data and store in database"
    echo "  3. Train PINN model (20 epochs)"
    echo "  4. Launch web interface"
    echo ""

    # Step 1: Start Database
    section_header "Step 1/4: Starting Database"
    debug_msg "Checking Docker and database..."

    if command -v docker &> /dev/null; then
        debug_msg "Docker found: $(which docker)"

        # Check if TimescaleDB container exists (running or stopped)
        if docker ps -a | grep -q timescale; then
            echo -e "${YELLOW}$(timestamp) Database container exists, recreating to ensure schema...${NC}"
            debug_msg "Removing existing TimescaleDB container to apply init script"

            # Stop and remove existing container
            docker stop pinn-timescaledb 2>/dev/null || true
            docker rm pinn-timescaledb 2>/dev/null || true
            debug_msg "Old container removed"
        fi

        # Start fresh database container
        echo -e "${YELLOW}$(timestamp) Starting TimescaleDB with schema initialization...${NC}"
        debug_msg "Starting TimescaleDB container..."

        if [ ! -f "docker-compose.yml" ]; then
            echo -e "${RED}$(timestamp) [ERROR] docker-compose.yml not found${NC}"
            debug_msg "Docker Compose file missing"
            echo -e "${YELLOW}Continuing without database (data will be stored locally)...${NC}"
        else
            debug_msg "Command: docker-compose up -d timescaledb"
            STEP_START=$SECONDS

            if docker-compose up -d timescaledb 2>&1 | while IFS= read -r line; do
                echo -e "${CYAN}  $line${NC}"
                debug_msg "docker: $line"
            done; then
                STEP_TIME=$((SECONDS - STEP_START))
                echo -e "${GREEN}$(timestamp) ✓ Database started successfully (${STEP_TIME}s)${NC}"
                debug_msg "TimescaleDB started in $STEP_TIME seconds"

                # Wait for database to be ready and schema to be initialized
                echo -e "${YELLOW}$(timestamp) Waiting for database initialization and schema creation...${NC}"
                debug_msg "Waiting 10 seconds for database and schema initialization"
                sleep 10

                # Verify database is responding
                if docker ps | grep -q timescale; then
                    echo -e "${GREEN}$(timestamp) ✓ Database is ready with schema initialized${NC}"
                    debug_msg "Database container is running and ready"
                else
                    echo -e "${YELLOW}$(timestamp) Warning: Could not verify database status${NC}"
                    debug_msg "Database status unclear, continuing anyway"
                fi
            else
                echo -e "${YELLOW}$(timestamp) Warning: Failed to start database${NC}"
                echo -e "${YELLOW}Continuing without database (data will be stored locally)...${NC}"
                debug_msg "Database startup failed, proceeding without it"
            fi
        fi
    else
        echo -e "${YELLOW}$(timestamp) Docker not found - skipping database setup${NC}"
        echo -e "${CYAN}Data will be stored locally instead${NC}"
        debug_msg "Docker not installed, skipping database"
    fi
    echo ""

    # Step 2: Fetch Data
    section_header "Step 2/4: Fetching Financial Data"
    debug_msg "Initiating data fetch..."
    debug_msg "Command: python3 -m src.data.fetcher"
    STEP_START=$SECONDS

    echo -e "${YELLOW}$(timestamp) Fetching financial data...${NC}"

    if python3 -m src.data.fetcher 2>&1 | while IFS= read -r line; do
        echo "$line"
        debug_msg "fetcher: $line"
    done; then
        STEP_TIME=$((SECONDS - STEP_START))
        echo -e "${GREEN}$(timestamp) ✓ Data fetched successfully (${STEP_TIME}s)${NC}"
        debug_msg "Data fetch completed in $STEP_TIME seconds"
    else
        echo -e "${RED}$(timestamp) [ERROR] Data fetch failed${NC}"
        debug_msg "Data fetch failed, aborting demo"
        return 1
    fi
    echo ""

    # Step 3: Train Model
    section_header "Step 3/4: Training PINN Model (20 epochs)"
    debug_msg "Starting PINN training..."
    debug_msg "Command: python3 -m src.training.train --model pinn --epochs 20"
    STEP_START=$SECONDS

    echo -e "${YELLOW}$(timestamp) Training model...${NC}"

    if python3 -m src.training.train --model pinn --epochs 20 2>&1 | while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "Epoch" ]] || [[ "$line" =~ "Loss" ]] || [[ "$line" =~ "Saving" ]]; then
            debug_msg "training: $line"
        fi
    done; then
        STEP_TIME=$((SECONDS - STEP_START))
        echo -e "${GREEN}$(timestamp) ✓ Model trained successfully (${STEP_TIME}s)${NC}"
        debug_msg "Training completed in $STEP_TIME seconds"
        sync_models_for_webapp
    else
        echo -e "${RED}$(timestamp) [ERROR] Training failed${NC}"
        debug_msg "Training failed, aborting demo"
        return 1
    fi
    echo ""

    # Step 4: Launch UI
    section_header "Step 4/4: Launching Web Interface"
    debug_msg "Starting Streamlit application..."
    DEMO_TIME=$((SECONDS - DEMO_START))
    debug_msg "Total demo preparation time: $DEMO_TIME seconds"

    echo -e "${GREEN}$(timestamp) ✓ Demo setup complete!${NC}"
    echo ""
    echo "Summary:"
    echo -e "${CYAN}  • Database: $(docker ps | grep -q timescale && echo "Running" || echo "Local mode")${NC}"
    echo -e "${CYAN}  • Data: Fetched and stored${NC}"
    echo -e "${CYAN}  • Model: Trained (20 epochs)${NC}"
    echo -e "${CYAN}  • Total time: $((DEMO_TIME / 60))m $((DEMO_TIME % 60))s${NC}"
    echo ""
    echo -e "${GREEN}$(timestamp) Starting Streamlit app at http://localhost:8501${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the web interface${NC}"
    echo ""
    debug_msg "Command: streamlit run src/web/app.py"
    debug_msg "Launching Streamlit..."

    streamlit run src/web/app.py
}

# Function to fetch data
fetch_data() {
    section_header "Fetching Financial Data"
    debug_msg "Starting data fetch operation..."
    debug_msg "Command: python3 -m src.data.fetcher"
    FETCH_START=$SECONDS

    echo -e "${YELLOW}$(timestamp) Fetching financial data...${NC}"

    if python3 -m src.data.fetcher 2>&1 | while IFS= read -r line; do
        echo "$line"
        debug_msg "fetcher: $line"
    done; then
        FETCH_TIME=$((SECONDS - FETCH_START))
        echo -e "${GREEN}$(timestamp) ✓ Data fetched successfully (${FETCH_TIME}s)${NC}"
        debug_msg "Data fetch completed in $FETCH_TIME seconds"
    else
        echo -e "${RED}$(timestamp) [ERROR] Data fetch failed${NC}"
        debug_msg "Data fetch operation failed"
        return 1
    fi
}

# Function to train PINN
train_pinn() {
    section_header "Training PINN Model"
    debug_msg "Preparing PINN training..."

    echo -e "${YELLOW}$(timestamp) Training PINN model...${NC}"
    read -p "Enter number of epochs (default: 20): " epochs
    epochs=${epochs:-20}

    debug_msg "User selected $epochs epochs"
    debug_msg "Command: python3 -m src.training.train --model pinn --epochs $epochs"
    TRAIN_START=$SECONDS

    if python3 -m src.training.train --model pinn --epochs $epochs 2>&1 | while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "Epoch" ]] || [[ "$line" =~ "Loss" ]] || [[ "$line" =~ "Saving" ]]; then
            debug_msg "training: $line"
        fi
    done; then
        TRAIN_TIME=$((SECONDS - TRAIN_START))
        echo -e "${GREEN}$(timestamp) ✓ PINN model trained (${TRAIN_TIME}s)${NC}"
        debug_msg "PINN training completed in $TRAIN_TIME seconds"
        sync_models_for_webapp
    else
        echo -e "${RED}$(timestamp) [ERROR] Training failed${NC}"
        debug_msg "PINN training failed"
        return 1
    fi
}

# Function to train all models
train_all() {
    section_header "Training All Models"
    debug_msg "Starting batch training for all models..."

    echo -e "${YELLOW}$(timestamp) Training all models...${NC}"
    read -p "Enter number of epochs (default: 100): " epochs
    epochs=${epochs:-100}

    debug_msg "User selected $epochs epochs for all models"
    TOTAL_START=$SECONDS
    MODELS=("lstm" "gru" "transformer" "pinn")
    COMPLETED=0
    TOTAL_MODELS=${#MODELS[@]}

    for model in "${MODELS[@]}"; do
        COMPLETED=$((COMPLETED + 1))
        MODEL_UPPER=$(echo "$model" | tr '[:lower:]' '[:upper:]')
        section_header "Training Model $COMPLETED/$TOTAL_MODELS: $MODEL_UPPER"
        debug_msg "Starting $model training..."
        debug_msg "Command: python3 -m src.training.train --model $model --epochs $epochs"
        MODEL_START=$SECONDS

        echo -e "${BLUE}$(timestamp) Training $model...${NC}"

        if python3 -m src.training.train --model $model --epochs $epochs 2>&1 | while IFS= read -r line; do
            echo "$line"
            if [[ "$line" =~ "Epoch" ]] || [[ "$line" =~ "Loss" ]] || [[ "$line" =~ "Saving" ]]; then
                debug_msg "$model: $line"
            fi
        done; then
            MODEL_TIME=$((SECONDS - MODEL_START))
            echo -e "${GREEN}$(timestamp) ✓ $model trained successfully (${MODEL_TIME}s)${NC}"
            debug_msg "$model training completed in $MODEL_TIME seconds"
            sync_models_for_webapp
        else
            echo -e "${RED}$(timestamp) [ERROR] $model training failed${NC}"
            debug_msg "$model training failed"
            return 1
        fi
        echo ""
    done

    TOTAL_TIME=$((SECONDS - TOTAL_START))
    echo -e "${GREEN}$(timestamp) ✓ All models trained successfully!${NC}"
    debug_msg "Total training time for all models: $TOTAL_TIME seconds"
    debug_msg "Average time per model: $((TOTAL_TIME / TOTAL_MODELS)) seconds"
}

# Function to launch web UI
launch_web() {
    section_header "Launching Web Interface"
    debug_msg "Starting Streamlit application..."

    # Check if app.py exists
    if [ ! -f "src/web/app.py" ]; then
        echo -e "${RED}$(timestamp) [ERROR] src/web/app.py not found${NC}"
        debug_msg "Web app file missing"
        return 1
    fi

    debug_msg "Web app file found: $(pwd)/src/web/app.py"
    debug_msg "File size: $(wc -c < src/web/app.py) bytes"

    echo -e "${YELLOW}$(timestamp) Launching web interface...${NC}"
    echo -e "${GREEN}$(timestamp) ✓ Starting Streamlit app at http://localhost:8501${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    debug_msg "Command: streamlit run src/web/app.py"

    streamlit run src/web/app.py
}

# Function to run tests
run_tests() {
    section_header "Running Test Suite"
    debug_msg "Checking for test directory..."

    echo -e "${YELLOW}$(timestamp) Running tests...${NC}"

    if [ -d "tests" ]; then
        debug_msg "Tests directory found: $(pwd)/tests"
        TEST_COUNT=$(find tests -name "test_*.py" -o -name "*_test.py" | wc -l | tr -d ' ')
        debug_msg "Found $TEST_COUNT test files"
        debug_msg "Command: pytest tests/ -v"
        TEST_START=$SECONDS

        if pytest tests/ -v 2>&1 | while IFS= read -r line; do
            echo "$line"
            if [[ "$line" =~ "PASSED" ]] || [[ "$line" =~ "FAILED" ]] || [[ "$line" =~ "ERROR" ]]; then
                debug_msg "test: $line"
            fi
        done; then
            TEST_TIME=$((SECONDS - TEST_START))
            echo -e "${GREEN}$(timestamp) ✓ Tests completed (${TEST_TIME}s)${NC}"
            debug_msg "Test suite completed in $TEST_TIME seconds"
        else
            echo -e "${RED}$(timestamp) [ERROR] Some tests failed${NC}"
            debug_msg "Test suite had failures"
            return 1
        fi
    else
        echo -e "${RED}$(timestamp) [ERROR] No tests directory found${NC}"
        debug_msg "Tests directory not found at: $(pwd)/tests"
        return 1
    fi
}

# Function to start database
start_database() {
    section_header "Starting TimescaleDB"
    debug_msg "Checking Docker availability..."

    echo -e "${YELLOW}$(timestamp) Starting TimescaleDB with Docker...${NC}"

    if command -v docker &> /dev/null; then
        debug_msg "Docker found: $(which docker)"
        debug_msg "Docker version: $(docker --version)"

        # Check if docker-compose.yml exists
        if [ ! -f "docker-compose.yml" ]; then
            echo -e "${RED}$(timestamp) [ERROR] docker-compose.yml not found${NC}"
            debug_msg "Docker Compose file missing"
            return 1
        fi

        debug_msg "Docker Compose file found"
        debug_msg "Command: docker-compose up -d timescaledb"

        if docker-compose up -d timescaledb 2>&1 | while IFS= read -r line; do
            echo -e "${CYAN}  $line${NC}"
            debug_msg "docker: $line"
        done; then
            echo -e "${GREEN}$(timestamp) ✓ Database started at localhost:5432${NC}"
            debug_msg "TimescaleDB container started"

            # Check container status
            sleep 2
            CONTAINER_STATUS=$(docker ps --filter "name=timescale" --format "{{.Status}}")
            debug_msg "Container status: $CONTAINER_STATUS"
        else
            echo -e "${RED}$(timestamp) [ERROR] Failed to start database${NC}"
            debug_msg "Docker Compose failed"
            return 1
        fi
    else
        echo -e "${RED}$(timestamp) [ERROR] Docker not found${NC}"
        debug_msg "Docker not installed"
        exit 1
    fi
}

# Function to start full Docker stack
start_docker_stack() {
    section_header "Starting Full Docker Stack"
    debug_msg "Initializing full Docker stack..."

    echo -e "${YELLOW}$(timestamp) Starting full Docker stack...${NC}"

    if command -v docker &> /dev/null; then
        debug_msg "Docker found: $(which docker)"

        if [ ! -f "docker-compose.yml" ]; then
            echo -e "${RED}$(timestamp) [ERROR] docker-compose.yml not found${NC}"
            debug_msg "Docker Compose file missing"
            return 1
        fi

        debug_msg "Command: docker-compose up --build"
        echo -e "${CYAN}$(timestamp) Building and starting all containers...${NC}"

        docker-compose up --build 2>&1 | while IFS= read -r line; do
            echo "$line"
            debug_msg "docker: $line"
        done

        echo -e "${GREEN}$(timestamp) ✓ Full stack running${NC}"
        debug_msg "Docker stack started"
    else
        echo -e "${RED}$(timestamp) [ERROR] Docker not found${NC}"
        debug_msg "Docker not installed"
        exit 1
    fi
}

# Function to run complete pipeline
run_complete_pipeline() {
    section_header "Complete Pipeline: End-to-End Execution"
    debug_msg "Starting complete pipeline execution..."

    PIPELINE_START=$SECONDS
    echo -e "${YELLOW}$(timestamp) Running complete end-to-end pipeline...${NC}"
    echo ""
    echo "This will:"
    echo "  1. Start TimescaleDB (if not running)"
    echo "  2. Fetch all financial data"
    echo "  3. Store data in database"
    echo "  4. Train all models (LSTM, GRU, Transformer, PINN)"
    echo "  5. Store trained models and results"
    echo ""

    read -p "Enter number of epochs for training (default: 100): " epochs
    epochs=${epochs:-100}
    debug_msg "User selected $epochs epochs for complete pipeline"
    echo ""

    # Step 1: Start Database
    section_header "Pipeline Step 1/4: Starting Database"
    debug_msg "Checking if database is running..."

    if command -v docker &> /dev/null; then
        # Check if TimescaleDB container is already running
        if docker ps | grep -q timescale; then
            echo -e "${GREEN}$(timestamp) ✓ Database already running${NC}"
            debug_msg "TimescaleDB container already active"
        else
            echo -e "${YELLOW}$(timestamp) Starting TimescaleDB...${NC}"
            debug_msg "Starting TimescaleDB container..."

            if [ ! -f "docker-compose.yml" ]; then
                echo -e "${RED}$(timestamp) [ERROR] docker-compose.yml not found${NC}"
                debug_msg "Docker Compose file missing, aborting pipeline"
                return 1
            fi

            debug_msg "Command: docker-compose up -d timescaledb"
            if docker-compose up -d timescaledb 2>&1 | while IFS= read -r line; do
                echo -e "${CYAN}  $line${NC}"
                debug_msg "docker: $line"
            done; then
                echo -e "${GREEN}$(timestamp) ✓ Database started successfully${NC}"
                debug_msg "TimescaleDB started, waiting for initialization..."
                echo -e "${YELLOW}$(timestamp) Waiting 5 seconds for database initialization...${NC}"
                sleep 5
            else
                echo -e "${RED}$(timestamp) [ERROR] Failed to start database${NC}"
                debug_msg "Database startup failed, aborting pipeline"
                return 1
            fi
        fi
    else
        echo -e "${RED}$(timestamp) [ERROR] Docker not found${NC}"
        debug_msg "Docker not installed, cannot start database"
        return 1
    fi
    echo ""

    # Step 2: Fetch Financial Data
    section_header "Pipeline Step 2/4: Fetching All Financial Data"
    debug_msg "Starting data fetch operation..."
    debug_msg "Command: python3 -m src.data.fetcher"
    STEP_START=$SECONDS

    echo -e "${YELLOW}$(timestamp) Fetching financial data from all sources...${NC}"

    if python3 -m src.data.fetcher 2>&1 | while IFS= read -r line; do
        echo "$line"
        debug_msg "fetcher: $line"
    done; then
        STEP_TIME=$((SECONDS - STEP_START))
        echo -e "${GREEN}$(timestamp) ✓ Data fetched and stored successfully (${STEP_TIME}s)${NC}"
        debug_msg "Data fetch completed in $STEP_TIME seconds"
    else
        echo -e "${RED}$(timestamp) [ERROR] Data fetch failed${NC}"
        debug_msg "Data fetch failed, aborting pipeline"
        return 1
    fi
    echo ""

    # Step 3: Train All Models
    section_header "Pipeline Step 3/4: Training All Models"
    debug_msg "Starting batch training for all models..."

    TOTAL_START=$SECONDS
    MODELS=("lstm" "gru" "transformer" "pinn")
    COMPLETED=0
    TOTAL_MODELS=${#MODELS[@]}

    echo -e "${YELLOW}$(timestamp) Training all models with $epochs epochs...${NC}"
    echo ""

    for model in "${MODELS[@]}"; do
        COMPLETED=$((COMPLETED + 1))
        MODEL_UPPER=$(echo "$model" | tr '[:lower:]' '[:upper:]')
        section_header "Training Model $COMPLETED/$TOTAL_MODELS: $MODEL_UPPER"
        debug_msg "Starting $model training..."
        debug_msg "Command: python3 -m src.training.train --model $model --epochs $epochs"
        MODEL_START=$SECONDS

        echo -e "${BLUE}$(timestamp) Training $model model...${NC}"

        if python3 -m src.training.train --model $model --epochs $epochs 2>&1 | while IFS= read -r line; do
            echo "$line"
            if [[ "$line" =~ "Epoch" ]] || [[ "$line" =~ "Loss" ]] || [[ "$line" =~ "Saving" ]] || [[ "$line" =~ "Stored" ]]; then
                debug_msg "$model: $line"
            fi
        done; then
            MODEL_TIME=$((SECONDS - MODEL_START))
            echo -e "${GREEN}$(timestamp) ✓ $model trained and stored successfully (${MODEL_TIME}s)${NC}"
            debug_msg "$model training completed in $MODEL_TIME seconds"
        else
            echo -e "${RED}$(timestamp) [ERROR] $model training failed${NC}"
            debug_msg "$model training failed, continuing with remaining models..."
            echo -e "${YELLOW}$(timestamp) Continuing with remaining models...${NC}"
        fi
        echo ""
    done

    TRAINING_TIME=$((SECONDS - TOTAL_START))
    echo -e "${GREEN}$(timestamp) ✓ All models training completed!${NC}"
    debug_msg "Total training time: $TRAINING_TIME seconds"
    debug_msg "Average time per model: $((TRAINING_TIME / TOTAL_MODELS)) seconds"
    echo ""

    # Step 4: Verification
    section_header "Pipeline Step 4/4: Verification"
    debug_msg "Running verification checks..."

    echo -e "${YELLOW}$(timestamp) Verifying pipeline execution...${NC}"
    echo ""

    # Check for saved models
    debug_msg "Checking for saved model files..."
    if [ -d "models" ]; then
        MODEL_COUNT=$(find models -name "*.pth" -o -name "*.h5" -o -name "*.pt" | wc -l | tr -d ' ')
        echo -e "${CYAN}  • Found $MODEL_COUNT model file(s) in models/ directory${NC}"
        debug_msg "Found $MODEL_COUNT model files"
    else
        echo -e "${YELLOW}  • No models directory found${NC}"
        debug_msg "Models directory not found"
    fi

    # Check database connection
    debug_msg "Verifying database connectivity..."
    if docker ps | grep -q timescale; then
        echo -e "${CYAN}  • TimescaleDB container is running${NC}"
        debug_msg "Database container verified"
    else
        echo -e "${YELLOW}  • TimescaleDB container not found${NC}"
        debug_msg "Database container not running"
    fi

    # Summary
    PIPELINE_TIME=$((SECONDS - PIPELINE_START))
    echo ""
    section_header "Pipeline Complete"
    echo -e "${GREEN}$(timestamp) ✓ Complete pipeline executed successfully!${NC}"
    echo ""
    echo "Summary:"
    echo -e "${CYAN}  • Total execution time: ${PIPELINE_TIME}s ($((PIPELINE_TIME / 60))m $((PIPELINE_TIME % 60))s)${NC}"
    echo -e "${CYAN}  • Models trained: ${TOTAL_MODELS}${NC}"
    echo -e "${CYAN}  • Epochs per model: ${epochs}${NC}"
    echo -e "${CYAN}  • Average training time: $((TRAINING_TIME / TOTAL_MODELS))s per model${NC}"
    echo ""
    debug_msg "Complete pipeline finished in $PIPELINE_TIME seconds"
    debug_msg "Pipeline summary: $TOTAL_MODELS models trained with $epochs epochs each"

    sync_models_for_webapp

    echo -e "${YELLOW}Next steps:${NC}"
    echo "  • Run option 5 to launch the web interface"
    echo "  • Run option 6 to execute tests"
    echo "  • Check logs in: $LOGFILE"
    echo ""
}

# Function to run systematic PINN comparison
run_pinn_comparison() {
    section_header "Systematic PINN Physics Comparison"
    debug_msg "Starting systematic PINN configuration comparison..."

    COMPARISON_START=$SECONDS
    echo -e "${YELLOW}$(timestamp) Running systematic PINN physics comparison...${NC}"
    echo ""
    echo "This will train and compare 6 PINN variants:"
    echo "  1. Baseline (Data-only) - No physics constraints"
    echo "  2. Pure GBM (Trend) - Geometric Brownian Motion"
    echo "  3. Pure OU (Mean-Reversion) - Ornstein-Uhlenbeck"
    echo "  4. Pure Black-Scholes - No-arbitrage PDE"
    echo "  5. GBM+OU Hybrid - Combined trend & mean-reversion"
    echo "  6. Global Constraint - All equations combined"
    echo ""
    echo "Outputs:"
    echo "  • Individual model checkpoints (e.g., pinn_black_scholes.pt)"
    echo "  • Per-variant training logs with data_loss and physics_loss"
    echo "  • Violation scores (physics_loss / data_loss ratio)"
    echo "  • comparison_report.csv ranking all variants"
    echo "  • README_theory.md with financial justifications"
    echo ""

    read -p "Enter number of epochs per variant (default: 100): " epochs
    epochs=${epochs:-100}
    debug_msg "User selected $epochs epochs for PINN comparison"
    echo ""

    # Ask which variants to train
    echo -e "${CYAN}Train all 6 variants or select specific ones?${NC}"
    echo "1) Train all variants"
    echo "2) Select specific variants"
    read -p "Enter choice [1-2]: " variant_choice

    VARIANT_ARGS=""
    if [ "$variant_choice" = "2" ]; then
        echo ""
        echo "Available variants:"
        echo "  baseline, gbm, ou, black_scholes, gbm_ou, global"
        echo ""
        read -p "Enter variants (space-separated): " selected_variants
        if [ -n "$selected_variants" ]; then
            VARIANT_ARGS="--variants $selected_variants"
            debug_msg "User selected variants: $selected_variants"
        fi
    else
        debug_msg "Training all variants"
    fi
    echo ""

    # Run comparison
    section_header "Running PINN Comparison"
    debug_msg "Starting training script..."
    debug_msg "Command: python3 -m src.training.train_pinn_variants --epochs $epochs $VARIANT_ARGS"
    STEP_START=$SECONDS

    echo -e "${YELLOW}$(timestamp) Training PINN variants...${NC}"
    echo -e "${CYAN}This may take a while depending on the number of epochs and variants...${NC}"
    echo ""

    if python3 -m src.training.train_pinn_variants --epochs $epochs $VARIANT_ARGS 2>&1 | while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "VARIANT" ]] || [[ "$line" =~ "Training" ]] || [[ "$line" =~ "Violation Score" ]] || [[ "$line" =~ "Best model" ]]; then
            debug_msg "comparison: $line"
        fi
    done; then
        STEP_TIME=$((SECONDS - STEP_START))
        echo -e "${GREEN}$(timestamp) ✓ PINN comparison completed successfully (${STEP_TIME}s)${NC}"
        debug_msg "PINN comparison completed in $STEP_TIME seconds"
        sync_models_for_webapp
    else
        echo -e "${RED}$(timestamp) [ERROR] PINN comparison failed${NC}"
        debug_msg "PINN comparison failed"
        return 1
    fi
    echo ""

    # Summary
    COMPARISON_TIME=$((SECONDS - COMPARISON_START))
    section_header "PINN Comparison Complete"
    echo -e "${GREEN}$(timestamp) ✓ Systematic PINN comparison finished!${NC}"
    echo ""
    echo "Summary:"
    echo -e "${CYAN}  • Total execution time: ${COMPARISON_TIME}s ($((COMPARISON_TIME / 60))m $((COMPARISON_TIME % 60))s)${NC}"
    echo -e "${CYAN}  • Epochs per variant: ${epochs}${NC}"
    echo ""
    debug_msg "PINN comparison finished in $COMPARISON_TIME seconds"

    echo -e "${YELLOW}Results saved to: results/pinn_comparison/${NC}"
    echo "  • README_theory.md - Financial theory & justifications"
    echo "  • comparison_report.csv - Ranked comparison of all variants"
    echo "  • detailed_results.json - Complete training history"
    echo "  • models/pinn_*_best.pt - Individual model checkpoints"
    echo ""

    echo -e "${CYAN}Next steps:${NC}"
    echo "  • Review comparison_report.csv to see which physics constraints work best"
    echo "  • Read README_theory.md to understand the financial theory"
    echo "  • Use the best model for forecasting (lowest MSE + violation score)"
    echo ""
}

# Function to run full model pipeline (all 14 models)
run_full_model_pipeline() {
    section_header "Full Model Pipeline: All 14 Neural Network Models"
    debug_msg "Starting comprehensive full model pipeline..."

    FULL_PIPELINE_START=$SECONDS
    echo -e "${YELLOW}$(timestamp) Running comprehensive full model pipeline...${NC}"
    echo ""
    echo "This will execute the complete training and evaluation pipeline:"
    echo ""
    echo "  Phase 1: Infrastructure Setup"
    echo "    • Start TimescaleDB container"
    echo "    • Fetch all financial data"
    echo ""
    echo "  Phase 2: Baseline Models (5 models)"
    echo "    • LSTM - Long Short-Term Memory"
    echo "    • GRU - Gated Recurrent Unit"
    echo "    • BiLSTM - Bidirectional LSTM"
    echo "    • Attention LSTM - LSTM with attention"
    echo "    • Transformer - Multi-head self-attention"
    echo ""
    echo "  Phase 3: PINN Variants (6 models)"
    echo "    • Baseline (Data-only)"
    echo "    • Pure GBM (Trend)"
    echo "    • Pure OU (Mean-Reversion)"
    echo "    • Pure Black-Scholes (No-arbitrage)"
    echo "    • GBM+OU Hybrid"
    echo "    • Global Constraint (All physics)"
    echo ""
    echo "  Phase 4: Advanced PINN Architectures (2 models)"
    echo "    • StackedPINN - Physics encoder + parallel LSTM/GRU"
    echo "    • ResidualPINN - Base model + physics correction"
    echo ""
    echo "  Phase 5: Unified Evaluation & Dashboard"
    echo "    • Comprehensive financial metrics for all models"
    echo "    • Launch interactive web dashboard"
    echo ""
    echo "  Total: 14 models with unified evaluation"
    echo ""

    read -p "Enter number of epochs for baseline/PINN models (default: 100): " epochs
    epochs=${epochs:-100}

    read -p "Enter number of epochs for advanced PINN models (default: 100): " advanced_epochs
    advanced_epochs=${advanced_epochs:-100}

    debug_msg "User selected $epochs epochs for baseline/PINN, $advanced_epochs for advanced PINN"
    echo ""

    # ========== PHASE 1: INFRASTRUCTURE SETUP ==========
    section_header "PHASE 1/5: Infrastructure Setup"

    # Step 1.1: Start Database
    echo -e "${BLUE}$(timestamp) Step 1.1: Starting Database${NC}"
    debug_msg "Checking database status..."

    if command -v docker &> /dev/null; then
        if docker ps | grep -q timescale; then
            echo -e "${GREEN}$(timestamp) ✓ Database already running${NC}"
            debug_msg "TimescaleDB container already active"
        else
            echo -e "${YELLOW}$(timestamp) Starting TimescaleDB...${NC}"
            debug_msg "Starting TimescaleDB container..."

            if [ -f "docker-compose.yml" ]; then
                debug_msg "Command: docker-compose up -d timescaledb"
                if docker-compose up -d timescaledb 2>&1 | while IFS= read -r line; do
                    echo -e "${CYAN}  $line${NC}"
                    debug_msg "docker: $line"
                done; then
                    echo -e "${GREEN}$(timestamp) ✓ Database started successfully${NC}"
                    debug_msg "Waiting for database initialization..."
                    sleep 5
                else
                    echo -e "${YELLOW}$(timestamp) Warning: Database startup issue, continuing...${NC}"
                fi
            else
                echo -e "${YELLOW}$(timestamp) docker-compose.yml not found, skipping database${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}$(timestamp) Docker not found, skipping database${NC}"
    fi
    echo ""

    # Step 1.2: Fetch Financial Data
    echo -e "${BLUE}$(timestamp) Step 1.2: Fetching Financial Data${NC}"
    debug_msg "Starting data fetch..."
    debug_msg "Command: python3 -m src.data.fetcher"
    STEP_START=$SECONDS

    if python3 -m src.data.fetcher 2>&1 | while IFS= read -r line; do
        echo "$line"
        debug_msg "fetcher: $line"
    done; then
        STEP_TIME=$((SECONDS - STEP_START))
        echo -e "${GREEN}$(timestamp) ✓ Data fetched successfully (${STEP_TIME}s)${NC}"
        debug_msg "Data fetch completed in $STEP_TIME seconds"
    else
        echo -e "${RED}$(timestamp) [ERROR] Data fetch failed${NC}"
        debug_msg "Data fetch failed, aborting pipeline"
        return 1
    fi
    echo ""

    # ========== PHASE 2: BASELINE MODELS ==========
    section_header "PHASE 2/5: Training Baseline Models (5 models)"
    debug_msg "Starting baseline model training phase..."

    BASELINE_START=$SECONDS
    BASELINE_MODELS=("lstm" "gru" "bilstm" "attention_lstm" "transformer")
    BASELINE_COMPLETED=0
    BASELINE_FAILED=0

    for model in "${BASELINE_MODELS[@]}"; do
        BASELINE_COMPLETED=$((BASELINE_COMPLETED + 1))
        MODEL_UPPER=$(echo "$model" | tr '[:lower:]' '[:upper:]')
        echo ""
        echo -e "${CYAN}========================================${NC}"
        echo -e "${CYAN}Baseline Model $BASELINE_COMPLETED/5: $MODEL_UPPER${NC}"
        echo -e "${CYAN}========================================${NC}"
        debug_msg "Starting $model training..."
        debug_msg "Command: python3 -m src.training.train --model $model --epochs $epochs"
        MODEL_START=$SECONDS

        if python3 -m src.training.train --model $model --epochs $epochs 2>&1 | while IFS= read -r line; do
            echo "$line"
            if [[ "$line" =~ "Epoch" ]] || [[ "$line" =~ "Loss" ]] || [[ "$line" =~ "Saving" ]]; then
                debug_msg "$model: $line"
            fi
        done; then
            MODEL_TIME=$((SECONDS - MODEL_START))
            echo -e "${GREEN}$(timestamp) ✓ $model trained successfully (${MODEL_TIME}s)${NC}"
            debug_msg "$model training completed in $MODEL_TIME seconds"
            sync_models_for_webapp
        else
            BASELINE_FAILED=$((BASELINE_FAILED + 1))
            echo -e "${YELLOW}$(timestamp) Warning: $model training failed, continuing...${NC}"
            debug_msg "$model training failed"
        fi
    done

    BASELINE_TIME=$((SECONDS - BASELINE_START))
    echo ""
    echo -e "${GREEN}$(timestamp) ✓ Baseline phase complete: $((5 - BASELINE_FAILED))/5 models trained (${BASELINE_TIME}s)${NC}"
    debug_msg "Baseline phase: $((5 - BASELINE_FAILED)) succeeded, $BASELINE_FAILED failed"
    echo ""
    sync_models_for_webapp

    # ========== PHASE 3: PINN VARIANTS ==========
    section_header "PHASE 3/5: Training PINN Variants (6 models)"
    debug_msg "Starting PINN variants training phase..."

    echo -e "${BLUE}$(timestamp) Training all 6 PINN variants with systematic physics comparison...${NC}"
    echo ""

    PINN_START=$SECONDS
    debug_msg "Command: python3 -m src.training.train_pinn_variants --epochs $epochs"

    if python3 -m src.training.train_pinn_variants --epochs $epochs 2>&1 | while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "VARIANT" ]] || [[ "$line" =~ "Training" ]] || [[ "$line" =~ "Violation" ]]; then
            debug_msg "pinn: $line"
        fi
    done; then
        PINN_TIME=$((SECONDS - PINN_START))
        echo -e "${GREEN}$(timestamp) ✓ All 6 PINN variants trained successfully (${PINN_TIME}s)${NC}"
        debug_msg "PINN variants phase completed in $PINN_TIME seconds"
        sync_models_for_webapp
    else
        echo -e "${YELLOW}$(timestamp) Warning: PINN variants training had issues, continuing...${NC}"
        debug_msg "PINN variants training had failures"
    fi
    echo ""

    # ========== PHASE 4: ADVANCED PINN ARCHITECTURES ==========
    section_header "PHASE 4/5: Training Advanced PINN Architectures (2 models)"
    debug_msg "Starting advanced PINN training phase..."

    ADVANCED_START=$SECONDS
    ADVANCED_MODELS=("stacked" "residual")
    ADVANCED_COMPLETED=0
    ADVANCED_FAILED=0

    for model_type in "${ADVANCED_MODELS[@]}"; do
        ADVANCED_COMPLETED=$((ADVANCED_COMPLETED + 1))
        MODEL_UPPER=$(echo "$model_type" | tr '[:lower:]' '[:upper:]')
        echo ""
        echo -e "${CYAN}========================================${NC}"
        echo -e "${CYAN}Advanced PINN $ADVANCED_COMPLETED/2: ${MODEL_UPPER}PINN${NC}"
        echo -e "${CYAN}========================================${NC}"
        debug_msg "Starting $model_type PINN training..."
        debug_msg "Command: python3 -m src.training.train_stacked_pinn --model-type $model_type --epochs $advanced_epochs"
        MODEL_START=$SECONDS

        if python3 -m src.training.train_stacked_pinn --model-type $model_type --epochs $advanced_epochs 2>&1 | while IFS= read -r line; do
            echo "$line"
            if [[ "$line" =~ "Epoch" ]] || [[ "$line" =~ "Loss" ]] || [[ "$line" =~ "Saving" ]] || [[ "$line" =~ "Phase" ]]; then
                debug_msg "$model_type: $line"
            fi
        done; then
            MODEL_TIME=$((SECONDS - MODEL_START))
            echo -e "${GREEN}$(timestamp) ✓ ${MODEL_UPPER}PINN trained successfully (${MODEL_TIME}s)${NC}"
            debug_msg "$model_type PINN training completed in $MODEL_TIME seconds"
            sync_models_for_webapp
        else
            ADVANCED_FAILED=$((ADVANCED_FAILED + 1))
            echo -e "${YELLOW}$(timestamp) Warning: ${MODEL_UPPER}PINN training failed, continuing...${NC}"
            debug_msg "$model_type PINN training failed"
        fi
    done

    ADVANCED_TIME=$((SECONDS - ADVANCED_START))
    echo ""
    echo -e "${GREEN}$(timestamp) ✓ Advanced PINN phase complete: $((2 - ADVANCED_FAILED))/2 models trained (${ADVANCED_TIME}s)${NC}"
    debug_msg "Advanced PINN phase: $((2 - ADVANCED_FAILED)) succeeded, $ADVANCED_FAILED failed"
    echo ""
    sync_models_for_webapp

    # ========== PHASE 5: UNIFIED EVALUATION & DASHBOARD ==========
    section_header "PHASE 5/5: Unified Evaluation & Dashboard"
    debug_msg "Running unified evaluation and launching dashboard..."

    echo -e "${BLUE}$(timestamp) Computing comprehensive financial metrics for all trained models...${NC}"
    echo ""

    # Check which models were trained
    echo -e "${CYAN}Scanning for trained models...${NC}"
    TRAINED_COUNT=0
    if [ -d "models" ]; then
        TRAINED_COUNT=$(find models -name "*_best.pt" -o -name "*_best.pth" | wc -l | tr -d ' ')
        echo -e "${GREEN}$(timestamp) ✓ Found $TRAINED_COUNT trained model checkpoints${NC}"
        debug_msg "Found $TRAINED_COUNT trained models"
    else
        echo -e "${YELLOW}$(timestamp) Warning: models directory not found${NC}"
        debug_msg "Models directory not found"
    fi
    echo ""

    # Evaluate Stacked/Residual PINN models first (generates results in dashboard format)
    if [ -f "evaluate_stacked_pinn.py" ]; then
        echo -e "${BLUE}$(timestamp) Evaluating Stacked/Residual PINN models...${NC}"
        debug_msg "Command: python3 evaluate_stacked_pinn.py"
        STACKED_EVAL_START=$SECONDS

        if python3 evaluate_stacked_pinn.py 2>&1 | while IFS= read -r line; do
            echo "$line"
            if [[ "$line" =~ "Evaluating" ]] || [[ "$line" =~ "✓" ]] || [[ "$line" =~ "Saved" ]]; then
                debug_msg "stacked_eval: $line"
            fi
        done; then
            STACKED_EVAL_TIME=$((SECONDS - STACKED_EVAL_START))
            echo ""
            echo -e "${GREEN}$(timestamp) ✓ Stacked/Residual PINN evaluation completed (${STACKED_EVAL_TIME}s)${NC}"
            debug_msg "Stacked PINN evaluation completed in $STACKED_EVAL_TIME seconds"
        else
            echo -e "${YELLOW}$(timestamp) Warning: Stacked/Residual PINN evaluation had issues${NC}"
            debug_msg "Stacked PINN evaluation failed"
        fi
        echo ""
    fi

    # Compute financial metrics for all models (including stacked/residual)
    if [ -f "compute_all_financial_metrics.py" ]; then
        echo -e "${BLUE}$(timestamp) Computing comprehensive financial metrics (Sharpe, Sortino, etc.)...${NC}"
        debug_msg "Command: python3 compute_all_financial_metrics.py"
        EVAL_START=$SECONDS

        if python3 compute_all_financial_metrics.py 2>&1 | while IFS= read -r line; do
            echo "$line"
            if [[ "$line" =~ "EVALUATING" ]] || [[ "$line" =~ "✓" ]] || [[ "$line" =~ "Sharpe" ]]; then
                debug_msg "eval: $line"
            fi
        done; then
            EVAL_TIME=$((SECONDS - EVAL_START))
            echo ""
            echo -e "${GREEN}$(timestamp) ✓ Financial metrics computed successfully (${EVAL_TIME}s)${NC}"
            debug_msg "Financial metrics evaluation completed in $EVAL_TIME seconds"
        else
            echo -e "${YELLOW}$(timestamp) Warning: Financial metrics computation had issues${NC}"
            debug_msg "Financial metrics computation failed"
        fi
        echo ""
    else
        echo -e "${YELLOW}$(timestamp) Warning: compute_all_financial_metrics.py not found${NC}"
        echo -e "${CYAN}Skipping financial metrics computation${NC}"
        debug_msg "Financial metrics script not found, skipping evaluation"
        echo ""
    fi

    # Display pipeline summary
    TOTAL_PIPELINE_TIME=$((SECONDS - FULL_PIPELINE_START))
    section_header "Full Pipeline Complete"
    echo -e "${GREEN}$(timestamp) ✓✓✓ FULL MODEL PIPELINE EXECUTED SUCCESSFULLY ✓✓✓${NC}"
    echo ""
    echo "========================================"
    echo "COMPREHENSIVE TRAINING SUMMARY"
    echo "========================================"
    echo ""
    echo "Infrastructure:"
    echo -e "${CYAN}  • Database: $(docker ps | grep -q timescale && echo "Running ✓" || echo "Skipped")${NC}"
    echo -e "${CYAN}  • Data: Fetched and stored ✓${NC}"
    echo ""
    echo "Phase 2 - Baseline Models (5 total):"
    echo -e "${CYAN}  • LSTM, GRU, BiLSTM, Attention LSTM, Transformer${NC}"
    echo -e "${CYAN}  • Succeeded: $((5 - BASELINE_FAILED))/5${NC}"
    echo -e "${CYAN}  • Training time: ${BASELINE_TIME}s ($((BASELINE_TIME / 60))m)${NC}"
    echo ""
    echo "Phase 3 - PINN Variants (6 total):"
    echo -e "${CYAN}  • Baseline, GBM, OU, Black-Scholes, GBM+OU, Global${NC}"
    echo -e "${CYAN}  • All variants trained with physics constraints${NC}"
    echo -e "${CYAN}  • Training time: ${PINN_TIME}s ($((PINN_TIME / 60))m)${NC}"
    echo ""
    echo "Phase 4 - Advanced PINN (2 total):"
    echo -e "${CYAN}  • StackedPINN, ResidualPINN${NC}"
    echo -e "${CYAN}  • Succeeded: $((2 - ADVANCED_FAILED))/2${NC}"
    echo -e "${CYAN}  • Training time: ${ADVANCED_TIME}s ($((ADVANCED_TIME / 60))m)${NC}"
    echo ""
    echo "Overall Statistics:"
    echo -e "${CYAN}  • Total models trained: $TRAINED_COUNT/14${NC}"
    echo -e "${CYAN}  • Epochs (baseline/PINN): $epochs${NC}"
    echo -e "${CYAN}  • Epochs (advanced): $advanced_epochs${NC}"
    echo -e "${CYAN}  • Total execution time: ${TOTAL_PIPELINE_TIME}s ($((TOTAL_PIPELINE_TIME / 60))m $((TOTAL_PIPELINE_TIME % 60))s)${NC}"
    echo -e "${CYAN}  • Average per model: $((TOTAL_PIPELINE_TIME / 14))s${NC}"
    echo ""
    echo "========================================"
    debug_msg "Full pipeline completed in $TOTAL_PIPELINE_TIME seconds"
    debug_msg "Training summary: $TRAINED_COUNT models trained"
    echo ""

    # Launch dashboard
    echo -e "${YELLOW}$(timestamp) Launching All Models Dashboard...${NC}"
    echo ""
    echo "The dashboard provides:"
    echo "  • Training status for all 14 models (✅ trained / ⚪ untrained)"
    echo "  • Comprehensive financial metrics (15+ metrics)"
    echo "  • Multi-category comparison (Risk, Capital, Trading, Signal)"
    echo "  • Interactive visualizations"
    echo ""
    echo -e "${GREEN}$(timestamp) Starting Streamlit dashboard at http://localhost:8501${NC}"
    echo -e "${CYAN}Navigate to 'All Models Dashboard' in the sidebar${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the dashboard${NC}"
    echo ""
    debug_msg "Command: streamlit run src/web/app.py"

    streamlit run src/web/app.py
}

# Function to train baseline models
train_baseline_models() {
    section_header "Training Baseline Models"
    debug_msg "Starting baseline models training..."

    echo -e "${YELLOW}$(timestamp) Training all 5 baseline models...${NC}"
    echo ""
    echo "This will train:"
    echo "  1. LSTM - Long Short-Term Memory"
    echo "  2. GRU - Gated Recurrent Unit"
    echo "  3. BiLSTM - Bidirectional LSTM"
    echo "  4. Attention LSTM - LSTM with attention mechanism"
    echo "  5. Transformer - Multi-head self-attention"
    echo ""

    read -p "Enter number of epochs (default: 100): " epochs
    epochs=${epochs:-100}
    debug_msg "User selected $epochs epochs for baseline models"
    echo ""

    BASELINE_START=$SECONDS
    BASELINE_MODELS=("lstm" "gru" "bilstm" "attention_lstm" "transformer")
    COMPLETED=0
    FAILED=0
    TOTAL=${#BASELINE_MODELS[@]}

    for model in "${BASELINE_MODELS[@]}"; do
        COMPLETED=$((COMPLETED + 1))
        MODEL_UPPER=$(echo "$model" | tr '[:lower:]' '[:upper:]')

        echo ""
        echo -e "${CYAN}========================================${NC}"
        echo -e "${CYAN}Training Model $COMPLETED/$TOTAL: $MODEL_UPPER${NC}"
        echo -e "${CYAN}========================================${NC}"

        debug_msg "Starting $model training..."
        debug_msg "Command: python3 -m src.training.train --model $model --epochs $epochs"
        MODEL_START=$SECONDS

        if python3 -m src.training.train --model $model --epochs $epochs 2>&1 | while IFS= read -r line; do
            echo "$line"
            if [[ "$line" =~ "Epoch" ]] || [[ "$line" =~ "Loss" ]] || [[ "$line" =~ "Saving" ]]; then
                debug_msg "$model: $line"
            fi
        done; then
            MODEL_TIME=$((SECONDS - MODEL_START))
            echo -e "${GREEN}$(timestamp) ✓ $model trained successfully (${MODEL_TIME}s)${NC}"
            debug_msg "$model training completed in $MODEL_TIME seconds"
            sync_models_for_webapp
        else
            FAILED=$((FAILED + 1))
            echo -e "${YELLOW}$(timestamp) Warning: $model training failed${NC}"
            debug_msg "$model training failed"
        fi
    done

    BASELINE_TIME=$((SECONDS - BASELINE_START))
    echo ""
    section_header "Baseline Training Complete"
    echo -e "${GREEN}$(timestamp) ✓ Baseline models training finished!${NC}"
    echo ""
    echo "Summary:"
    echo -e "${CYAN}  • Models trained: $((TOTAL - FAILED))/$TOTAL${NC}"
    echo -e "${CYAN}  • Training time: ${BASELINE_TIME}s ($((BASELINE_TIME / 60))m $((BASELINE_TIME % 60))s)${NC}"
    echo -e "${CYAN}  • Average per model: $((BASELINE_TIME / TOTAL))s${NC}"
    echo ""

    if [ $FAILED -gt 0 ]; then
        echo -e "${YELLOW}$(timestamp) $FAILED model(s) failed to train${NC}"
    fi

    debug_msg "Baseline training completed: $((TOTAL - FAILED)) succeeded, $FAILED failed"
    sync_models_for_webapp
}

# Function to compute financial metrics for all models
compute_financial_metrics() {
    section_header "Computing Comprehensive Financial Metrics"
    debug_msg "Starting financial metrics computation..."

    echo -e "${YELLOW}$(timestamp) Computing comprehensive financial metrics for all trained models...${NC}"
    echo ""
    echo "This will compute 20+ financial metrics including:"
    echo "  • Risk-Adjusted: Sharpe Ratio, Sortino Ratio, Volatility"
    echo "  • Capital Preservation: Max Drawdown, Calmar Ratio, Recovery Time"
    echo "  • Trading Viability: Profit Factor, Win Rate, Annualized Return"
    echo "  • Signal Quality: Directional Accuracy, Information Coefficient"
    echo "  • Robustness: Rolling window analysis, stability metrics"
    echo ""

    # Check if evaluation script exists
    if [ ! -f "compute_all_financial_metrics.py" ]; then
        echo -e "${RED}$(timestamp) [ERROR] compute_all_financial_metrics.py not found${NC}"
        debug_msg "Financial metrics script not found"
        return 1
    fi

    # First, evaluate Stacked/Residual PINN if models exist
    if [ -f "evaluate_stacked_pinn.py" ] && [ -d "models/stacked_pinn" ]; then
        echo -e "${BLUE}$(timestamp) Step 1: Evaluating Stacked/Residual PINN models...${NC}"
        debug_msg "Command: python3 evaluate_stacked_pinn.py"

        if python3 evaluate_stacked_pinn.py 2>&1 | while IFS= read -r line; do
            echo "$line"
            if [[ "$line" =~ "Evaluating" ]] || [[ "$line" =~ "✓" ]] || [[ "$line" =~ "Saved" ]]; then
                debug_msg "stacked_eval: $line"
            fi
        done; then
            echo -e "${GREEN}$(timestamp) ✓ Stacked/Residual PINN evaluated${NC}"
        else
            echo -e "${YELLOW}$(timestamp) Warning: Stacked/Residual evaluation had issues${NC}"
        fi
        echo ""
    fi

    echo -e "${BLUE}$(timestamp) Step 2: Computing metrics for all models...${NC}"
    debug_msg "Command: python3 compute_all_financial_metrics.py"
    METRICS_START=$SECONDS

    echo -e "${CYAN}$(timestamp) Evaluating all trained models...${NC}"
    echo ""

    if python3 compute_all_financial_metrics.py 2>&1 | while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "EVALUATING" ]] || [[ "$line" =~ "✓" ]] || [[ "$line" =~ "Sharpe" ]]; then
            debug_msg "metrics: $line"
        fi
    done; then
        METRICS_TIME=$((SECONDS - METRICS_START))
        echo ""
        echo -e "${GREEN}$(timestamp) ✓ Financial metrics computed successfully (${METRICS_TIME}s)${NC}"
        debug_msg "Financial metrics computation completed in $METRICS_TIME seconds"
    else
        echo -e "${RED}$(timestamp) [ERROR] Financial metrics computation failed${NC}"
        debug_msg "Metrics computation failed"
        return 1
    fi

    echo ""
    echo -e "${CYAN}Results saved to: results/*_results.json${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  • View metrics: python3 view_metrics.py --compare"
    echo "  • Launch dashboard: streamlit run src/web/app.py"
    echo "  • Individual model: python3 view_metrics.py --model pinn_gbm"
    echo ""
}

# Function to view metrics in terminal
view_metrics_terminal() {
    section_header "View Financial Metrics in Terminal"
    debug_msg "Starting metrics viewer..."

    echo -e "${YELLOW}$(timestamp) Financial Metrics Viewer${NC}"
    echo ""
    echo "Select view mode:"
    echo ""
    echo "1) Compare All Models (Table)"
    echo "2) Quick Summary"
    echo "3) View Specific Model"
    echo "4) View All Models (Detailed)"
    echo "5) Back to Main Menu"
    echo ""

    read -p "Enter choice [1-5]: " view_choice
    debug_msg "User selected view option: $view_choice"
    echo ""

    # Check if viewer script exists
    if [ ! -f "view_metrics.py" ]; then
        echo -e "${RED}$(timestamp) [ERROR] view_metrics.py not found${NC}"
        debug_msg "Metrics viewer script not found"
        return 1
    fi

    case $view_choice in
        1)
            debug_msg "Command: python3 view_metrics.py --compare"
            python3 view_metrics.py --compare
            ;;
        2)
            debug_msg "Command: python3 view_metrics.py --summary"
            python3 view_metrics.py --summary
            ;;
        3)
            echo "Available models:"
            echo "  Baseline: lstm, gru, bilstm, attention_lstm, transformer"
            echo "  PINN: pinn_baseline, pinn_gbm, pinn_ou, pinn_black_scholes, pinn_gbm_ou, pinn_global"
            echo ""
            read -p "Enter model key: " model_key
            debug_msg "User selected model: $model_key"
            debug_msg "Command: python3 view_metrics.py --model $model_key"
            python3 view_metrics.py --model "$model_key"
            ;;
        4)
            debug_msg "Command: python3 view_metrics.py"
            python3 view_metrics.py
            ;;
        5)
            echo -e "${CYAN}$(timestamp) Returning to main menu...${NC}"
            return 0
            ;;
        *)
            echo -e "${RED}$(timestamp) Invalid option${NC}"
            return 1
            ;;
    esac

    echo ""
    echo -e "${GREEN}$(timestamp) Metrics viewer completed${NC}"
    debug_msg "Metrics viewer session finished"
}

# Function to evaluate stacked/residual PINN models
evaluate_stacked_residual_pinn() {
    section_header "Evaluating Stacked/Residual PINN Models"
    debug_msg "Starting stacked/residual PINN evaluation..."

    echo -e "${YELLOW}$(timestamp) Evaluating Stacked/Residual PINN models...${NC}"
    echo ""
    echo "This will:"
    echo "  • Load trained StackedPINN and ResidualPINN models"
    echo "  • Generate predictions on test data"
    echo "  • Compute comprehensive ML and financial metrics"
    echo "  • Save results in dashboard-compatible format"
    echo ""

    # Check if evaluation script exists
    if [ ! -f "evaluate_stacked_pinn.py" ]; then
        echo -e "${RED}$(timestamp) [ERROR] evaluate_stacked_pinn.py not found${NC}"
        debug_msg "Stacked PINN evaluation script not found"
        return 1
    fi

    # Check if models exist
    if [ ! -d "models/stacked_pinn" ]; then
        echo -e "${RED}$(timestamp) [ERROR] models/stacked_pinn directory not found${NC}"
        echo -e "${YELLOW}Train models first with: ./run.sh → Option 11${NC}"
        debug_msg "Stacked PINN models directory not found"
        return 1
    fi

    debug_msg "Command: python3 evaluate_stacked_pinn.py"
    EVAL_START=$SECONDS

    echo -e "${CYAN}$(timestamp) Running evaluation...${NC}"
    echo ""

    if python3 evaluate_stacked_pinn.py 2>&1 | while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "Evaluating" ]] || [[ "$line" =~ "✓" ]] || [[ "$line" =~ "Saved" ]] || [[ "$line" =~ "RMSE" ]] || [[ "$line" =~ "Sharpe" ]]; then
            debug_msg "eval: $line"
        fi
    done; then
        EVAL_TIME=$((SECONDS - EVAL_START))
        echo ""
        echo -e "${GREEN}$(timestamp) ✓ Stacked/Residual PINN evaluation completed (${EVAL_TIME}s)${NC}"
        debug_msg "Evaluation completed in $EVAL_TIME seconds"
    else
        echo -e "${RED}$(timestamp) [ERROR] Evaluation failed${NC}"
        debug_msg "Stacked/Residual PINN evaluation failed"
        return 1
    fi

    echo ""
    echo -e "${CYAN}Results saved to:${NC}"
    echo "  • results/pinn_stacked_results.json"
    echo "  • results/pinn_residual_results.json"
    echo "  • results/pinn_stacked_predictions.npz"
    echo "  • results/pinn_residual_predictions.npz"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  • View in dashboard: streamlit run src/web/app.py"
    echo "  • Navigate to 'PINN Comparison' or 'All Models Dashboard'"
    echo ""
}

# Function to launch backtesting dashboard
launch_backtesting_dashboard() {
    section_header "Launching Backtesting Dashboard"
    debug_msg "Starting backtesting dashboard..."

    # Check if backtesting dashboard exists
    if [ ! -f "src/web/backtesting_dashboard.py" ]; then
        echo -e "${RED}$(timestamp) [ERROR] src/web/backtesting_dashboard.py not found${NC}"
        debug_msg "Backtesting dashboard file missing"
        return 1
    fi

    debug_msg "Backtesting dashboard file found: $(pwd)/src/web/backtesting_dashboard.py"

    echo -e "${YELLOW}$(timestamp) Launching backtesting dashboard...${NC}"
    echo ""
    echo "The Backtesting Dashboard provides:"
    echo "  • Multiple strategy comparison (Buy & Hold, SMA, Momentum, Mean Reversion)"
    echo "  • Transaction cost modeling (Commission, Slippage)"
    echo "  • Risk management (Stop-loss, Take-profit)"
    echo "  • Position sizing methods (Fixed, Kelly, Volatility-based)"
    echo "  • Equity curves and drawdown analysis"
    echo "  • Walk-forward validation"
    echo "  • Monte Carlo simulation"
    echo ""
    echo -e "${GREEN}$(timestamp) ✓ Starting Streamlit app at http://localhost:8502${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    debug_msg "Command: streamlit run src/web/backtesting_dashboard.py --server.port 8502"

    streamlit run src/web/backtesting_dashboard.py --server.port 8502
}

# Function to recompute all metrics for consistency
recompute_all_metrics() {
    section_header "Recomputing All Metrics for Consistency"
    debug_msg "Starting metrics recomputation..."

    echo -e "${YELLOW}$(timestamp) Recomputing financial metrics for all trained models...${NC}"
    echo ""
    echo "This will:"
    echo "  • Re-run evaluation using unified evaluation pipeline"
    echo "  • Fix any inconsistencies between result files"
    echo "  • Validate all metrics and replace invalid values (inf/nan)"
    echo "  • Ensure max_drawdown is capped at -100%"
    echo "  • Apply overflow protection to cumulative returns"
    echo ""

    # Check if recompute script exists
    if [ ! -f "recompute_metrics.py" ]; then
        echo -e "${RED}$(timestamp) [ERROR] recompute_metrics.py not found${NC}"
        debug_msg "Recompute metrics script not found"
        return 1
    fi

    echo "Select mode:"
    echo "1) Validate only (no changes)"
    echo "2) Recompute all models"
    echo "3) Recompute specific models"
    echo ""
    read -p "Enter choice [1-3]: " recompute_choice
    debug_msg "User selected recompute option: $recompute_choice"
    echo ""

    case $recompute_choice in
        1)
            debug_msg "Command: python3 recompute_metrics.py --validate-only"
            RECOMPUTE_START=$SECONDS
            if python3 recompute_metrics.py --validate-only 2>&1 | while IFS= read -r line; do
                echo "$line"
                debug_msg "validate: $line"
            done; then
                RECOMPUTE_TIME=$((SECONDS - RECOMPUTE_START))
                echo -e "${GREEN}$(timestamp) ✓ Validation completed (${RECOMPUTE_TIME}s)${NC}"
            else
                echo -e "${RED}$(timestamp) Validation found issues${NC}"
            fi
            ;;
        2)
            debug_msg "Command: python3 recompute_metrics.py --force"
            RECOMPUTE_START=$SECONDS
            if python3 recompute_metrics.py --force 2>&1 | while IFS= read -r line; do
                echo "$line"
                if [[ "$line" =~ "Recomputing" ]] || [[ "$line" =~ "✓" ]] || [[ "$line" =~ "Updated" ]]; then
                    debug_msg "recompute: $line"
                fi
            done; then
                RECOMPUTE_TIME=$((SECONDS - RECOMPUTE_START))
                echo -e "${GREEN}$(timestamp) ✓ Recomputation completed (${RECOMPUTE_TIME}s)${NC}"
            else
                echo -e "${RED}$(timestamp) Recomputation had errors${NC}"
            fi
            ;;
        3)
            echo "Available models:"
            echo "  Baseline: lstm, gru, bilstm, attention_lstm, transformer"
            echo "  PINN: pinn_baseline, pinn_gbm, pinn_ou, pinn_black_scholes, pinn_gbm_ou, pinn_global"
            echo "  Advanced: pinn_stacked, pinn_residual"
            echo ""
            read -p "Enter model keys (space-separated): " model_keys
            debug_msg "User selected models: $model_keys"
            debug_msg "Command: python3 recompute_metrics.py --models $model_keys --force"
            RECOMPUTE_START=$SECONDS
            if python3 recompute_metrics.py --models $model_keys --force 2>&1 | while IFS= read -r line; do
                echo "$line"
                debug_msg "recompute: $line"
            done; then
                RECOMPUTE_TIME=$((SECONDS - RECOMPUTE_START))
                echo -e "${GREEN}$(timestamp) ✓ Recomputation completed (${RECOMPUTE_TIME}s)${NC}"
            else
                echo -e "${RED}$(timestamp) Recomputation had errors${NC}"
            fi
            ;;
        *)
            echo -e "${RED}$(timestamp) Invalid option${NC}"
            return 1
            ;;
    esac

    echo ""
    echo -e "${CYAN}Results updated in: results/*_results.json${NC}"
    echo ""
}

# Function to generate dissertation figures
generate_dissertation_figures() {
    section_header "Generating Dissertation-Quality Figures"
    debug_msg "Starting dissertation figure generation..."

    echo -e "${YELLOW}$(timestamp) Generating publication-quality visualizations...${NC}"
    echo ""
    echo "This will generate figures for:"
    echo "  1. Core Forecast Accuracy (predicted vs realized, residuals)"
    echo "  2. Loss and Calibration Diagnostics (QLIKE, PIT, VaR)"
    echo "  3. Economic Performance (equity curves, Sharpe, drawdown)"
    echo "  4. Model Stability & Sensitivity Analysis"
    echo "  5. Physics Compliance (SDE residuals, learned parameters)"
    echo "  6. Model Comparison Framework (PINN vs GARCH vs LSTM)"
    echo ""

    # Check if script exists
    if [ ! -f "scripts/dissertation_visualizations.py" ]; then
        echo -e "${RED}$(timestamp) [ERROR] scripts/dissertation_visualizations.py not found${NC}"
        debug_msg "Dissertation visualization script not found"
        return 1
    fi

    echo "Select output format:"
    echo "1) PDF (recommended for LaTeX)"
    echo "2) PNG (for presentations)"
    echo "3) SVG (for web)"
    echo ""
    read -p "Enter choice [1-3]: " format_choice

    case $format_choice in
        1) FORMAT="pdf" ;;
        2) FORMAT="png" ;;
        3) FORMAT="svg" ;;
        *) FORMAT="pdf" ;;
    esac

    echo ""
    read -p "Use synthetic data for demo? [y/N]: " use_synthetic
    SYNTHETIC_FLAG=""
    if [[ "$use_synthetic" =~ ^[Yy]$ ]]; then
        SYNTHETIC_FLAG="--synthetic"
    fi

    debug_msg "Format: $FORMAT, Synthetic: $SYNTHETIC_FLAG"
    debug_msg "Command: python3 scripts/dissertation_visualizations.py --format $FORMAT $SYNTHETIC_FLAG"
    FIGURES_START=$SECONDS

    mkdir -p figures/dissertation

    echo -e "${CYAN}$(timestamp) Generating figures...${NC}"
    echo ""

    if python3 scripts/dissertation_visualizations.py --format $FORMAT --output figures/dissertation $SYNTHETIC_FLAG 2>&1 | while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "Generating" ]] || [[ "$line" =~ "Saved" ]] || [[ "$line" =~ "✓" ]]; then
            debug_msg "figures: $line"
        fi
    done; then
        FIGURES_TIME=$((SECONDS - FIGURES_START))
        echo ""
        echo -e "${GREEN}$(timestamp) ✓ Dissertation figures generated successfully (${FIGURES_TIME}s)${NC}"
        debug_msg "Figure generation completed in $FIGURES_TIME seconds"
    else
        echo -e "${RED}$(timestamp) [ERROR] Figure generation failed${NC}"
        debug_msg "Dissertation figure generation failed"
        return 1
    fi

    echo ""
    echo -e "${CYAN}Figures saved to: figures/dissertation/${NC}"
    echo ""
    echo "Generated files include:"
    echo "  • fig1_predicted_vs_realized.$FORMAT - Core forecast visualization"
    echo "  • fig2_residual_diagnostics.$FORMAT - Residual analysis"
    echo "  • fig4_loss_evolution.$FORMAT - QLIKE/MSE over time"
    echo "  • fig5_pit_histogram.$FORMAT - Probability calibration"
    echo "  • fig8_equity_curve.$FORMAT - Strategy performance"
    echo "  • fig12_physics_residuals.$FORMAT - SDE compliance"
    echo "  • fig15_model_comparison.$FORMAT - Multi-model comparison"
    echo "  • performance_table.csv - Summary statistics"
    echo "  • performance_table.tex - LaTeX table"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  • Include figures in your dissertation LaTeX document"
    echo "  • Use performance_table.tex for results tables"
    echo "  • Run with --show flag to preview figures interactively"
    echo ""
}

# Function to launch training visualizations dashboard
launch_training_visualizations() {
    section_header "Launching Training Visualizations Dashboard"
    debug_msg "Starting training visualizations dashboard..."

    # Check if training dashboard exists
    if [ ! -f "src/web/training_dashboard.py" ]; then
        echo -e "${RED}$(timestamp) [ERROR] src/web/training_dashboard.py not found${NC}"
        debug_msg "Training dashboard file missing"
        return 1
    fi

    debug_msg "Training dashboard file found: $(pwd)/src/web/training_dashboard.py"

    echo -e "${YELLOW}$(timestamp) Launching training visualizations dashboard...${NC}"
    echo ""
    echo "The Training Visualizations Dashboard provides:"
    echo "  • Training overview with statistics for all models"
    echo "  • Loss curves (train vs validation) for all models"
    echo "  • Single model deep dive analysis"
    echo "  • Learning rate schedule visualization"
    echo "  • Overfitting detection and analysis"
    echo "  • Convergence comparison across models"
    echo "  • Physics loss decomposition (for PINN models)"
    echo ""
    echo -e "${GREEN}$(timestamp) ✓ Starting Streamlit app at http://localhost:8503${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    debug_msg "Command: streamlit run src/web/training_dashboard.py --server.port 8503"

    streamlit run src/web/training_dashboard.py --server.port 8503
}

# Function to run research-grade training (standalone script)
run_research_training() {
    section_header "Research-Grade Training (Standalone Script)"
    debug_msg "Starting research-grade training menu..."

    echo -e "${YELLOW}$(timestamp) Research-Grade Neural Network Training${NC}"
    echo ""
    echo "This uses the standalone training script with research-locked parameters:"
    echo ""
    echo "  Research Config (Locked for Fair Comparison):"
    echo "    • Epochs: 100 (full training for convergence)"
    echo "    • Batch Size: 16 (better gradient estimates)"
    echo "    • Learning Rate: 0.0005 (stability with deep models)"
    echo "    • Hidden Dim: 512 (deep architecture)"
    echo "    • Num Layers: 4 (hierarchical features)"
    echo "    • Dropout: 0.15 (moderate regularization)"
    echo "    • Sequence Length: 180 (6 months lookback)"
    echo "    • Early Stopping: DISABLED (all models train full epochs)"
    echo ""
    echo "Select training mode:"
    echo ""
    echo "1) Train single model"
    echo "2) Train all baseline models (LSTM, GRU, BiLSTM, Attention, Transformer)"
    echo "3) Train all PINN models (6 variants)"
    echo "4) Train all advanced models (Stacked, Residual)"
    echo "5) Train ALL models (13 total - full research run)"
    echo "6) Back to Main Menu"
    echo ""

    read -p "Enter choice [1-6]: " training_choice
    debug_msg "User selected training option: $training_choice"
    echo ""

    # Activate backend venv for training
    if [ -d "backend/venv" ]; then
        debug_msg "Activating backend virtual environment..."
        source backend/venv/bin/activate
    fi

    case $training_choice in
        1)
            echo "Available models:"
            echo "  Baseline: lstm, gru, bilstm, attention_lstm, transformer"
            echo "  PINN: baseline_pinn, gbm, ou, black_scholes, gbm_ou, global"
            echo "  Advanced: stacked, residual"
            echo ""
            read -p "Enter model key: " model_key
            debug_msg "Training single model: $model_key"
            debug_msg "Command: python scripts/train_models.py --model $model_key"

            echo ""
            echo -e "${CYAN}$(timestamp) Training $model_key with research config...${NC}"
            TRAIN_START=$SECONDS

            if python scripts/train_models.py --model "$model_key" 2>&1 | while IFS= read -r line; do
                echo "$line"
                if [[ "$line" =~ "Epoch" ]] || [[ "$line" =~ "RMSE" ]] || [[ "$line" =~ "Sharpe" ]]; then
                    debug_msg "training: $line"
                fi
            done; then
                TRAIN_TIME=$((SECONDS - TRAIN_START))
                echo -e "${GREEN}$(timestamp) ✓ Training completed (${TRAIN_TIME}s)${NC}"
                sync_models_for_webapp
            else
                echo -e "${RED}$(timestamp) Training failed${NC}"
            fi
            ;;
        2)
            debug_msg "Training all baseline models"
            debug_msg "Command: python scripts/train_models.py --type baseline"

            echo -e "${CYAN}$(timestamp) Training all baseline models...${NC}"
            TRAIN_START=$SECONDS

            if python scripts/train_models.py --type baseline 2>&1 | while IFS= read -r line; do
                echo "$line"
            done; then
                TRAIN_TIME=$((SECONDS - TRAIN_START))
                echo -e "${GREEN}$(timestamp) ✓ Baseline training completed (${TRAIN_TIME}s)${NC}"
                sync_models_for_webapp
            else
                echo -e "${RED}$(timestamp) Baseline training failed${NC}"
            fi
            ;;
        3)
            debug_msg "Training all PINN models"
            debug_msg "Command: python scripts/train_models.py --type pinn"

            echo -e "${CYAN}$(timestamp) Training all PINN models...${NC}"
            TRAIN_START=$SECONDS

            if python scripts/train_models.py --type pinn 2>&1 | while IFS= read -r line; do
                echo "$line"
            done; then
                TRAIN_TIME=$((SECONDS - TRAIN_START))
                echo -e "${GREEN}$(timestamp) ✓ PINN training completed (${TRAIN_TIME}s)${NC}"
                sync_models_for_webapp
            else
                echo -e "${RED}$(timestamp) PINN training failed${NC}"
            fi
            ;;
        4)
            debug_msg "Training all advanced models"
            debug_msg "Command: python scripts/train_models.py --type advanced"

            echo -e "${CYAN}$(timestamp) Training all advanced models...${NC}"
            TRAIN_START=$SECONDS

            if python scripts/train_models.py --type advanced 2>&1 | while IFS= read -r line; do
                echo "$line"
            done; then
                TRAIN_TIME=$((SECONDS - TRAIN_START))
                echo -e "${GREEN}$(timestamp) ✓ Advanced training completed (${TRAIN_TIME}s)${NC}"
                sync_models_for_webapp
            else
                echo -e "${RED}$(timestamp) Advanced training failed${NC}"
            fi
            ;;
        5)
            debug_msg "Training ALL models"
            debug_msg "Command: python scripts/train_models.py --all"

            echo -e "${CYAN}$(timestamp) Training ALL 13 models (this will take a while)...${NC}"
            echo ""
            echo "Models to train:"
            echo "  Baseline (5): lstm, gru, bilstm, attention_lstm, transformer"
            echo "  PINN (6): baseline_pinn, gbm, ou, black_scholes, gbm_ou, global"
            echo "  Advanced (2): stacked, residual"
            echo ""

            TRAIN_START=$SECONDS

            if python scripts/train_models.py --all 2>&1 | while IFS= read -r line; do
                echo "$line"
            done; then
                TRAIN_TIME=$((SECONDS - TRAIN_START))
                echo ""
                echo -e "${GREEN}$(timestamp) ✓ Full research training completed!${NC}"
                echo -e "${CYAN}  Total time: $((TRAIN_TIME / 60))m $((TRAIN_TIME % 60))s${NC}"
                sync_models_for_webapp
            else
                echo -e "${RED}$(timestamp) Training had errors${NC}"
            fi
            ;;
        6)
            echo -e "${CYAN}$(timestamp) Returning to main menu...${NC}"
            return 0
            ;;
        *)
            echo -e "${RED}$(timestamp) Invalid option${NC}"
            return 1
            ;;
    esac

    echo ""
    echo -e "${CYAN}Results saved to: results/<model>_results.json${NC}"
    echo -e "${CYAN}Checkpoints saved to: models/<model>_best.pt${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  • Option 21: Launch Full Web App to view results"
    echo "  • Option 14: View metrics in terminal"
    echo ""
}

# Function to launch full web app (backend + frontend)
launch_full_webapp() {
    section_header "Launching Full Web App (Backend + Frontend)"
    debug_msg "Starting full web application..."

    echo -e "${YELLOW}$(timestamp) Launching Full Web Application...${NC}"
    echo ""
    echo "This will start:"
    echo "  • Backend API (FastAPI) at http://localhost:8000"
    echo "  • Frontend UI (React) at http://localhost:5173"
    echo ""

    # Check if backend exists
    if [ ! -d "backend" ]; then
        echo -e "${RED}$(timestamp) [ERROR] backend/ directory not found${NC}"
        return 1
    fi

    # Check if frontend exists
    if [ ! -d "frontend" ]; then
        echo -e "${RED}$(timestamp) [ERROR] frontend/ directory not found${NC}"
        return 1
    fi

    # Start backend in background
    echo -e "${BLUE}$(timestamp) Starting Backend API...${NC}"
    debug_msg "Command: cd backend && source venv/bin/activate && python run.py"

    (
        cd backend
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        python run.py 2>&1 | while IFS= read -r line; do
            echo -e "${CYAN}[BACKEND] $line${NC}"
        done
    ) &
    BACKEND_PID=$!
    debug_msg "Backend started with PID: $BACKEND_PID"

    # Wait for backend to be ready
    echo -e "${YELLOW}$(timestamp) Waiting for backend to start...${NC}"
    sleep 3

    # Start frontend
    echo -e "${BLUE}$(timestamp) Starting Frontend UI...${NC}"
    debug_msg "Command: cd frontend && npm run dev"

    (
        cd frontend
        npm run dev 2>&1 | while IFS= read -r line; do
            echo -e "${MAGENTA}[FRONTEND] $line${NC}"
        done
    ) &
    FRONTEND_PID=$!
    debug_msg "Frontend started with PID: $FRONTEND_PID"

    echo ""
    echo -e "${GREEN}$(timestamp) ✓ Web Application Started!${NC}"
    echo ""
    echo "Access the application:"
    echo -e "${CYAN}  • Frontend (React): http://localhost:5173${NC}"
    echo -e "${CYAN}  • Backend API: http://localhost:8000${NC}"
    echo -e "${CYAN}  • API Docs: http://localhost:8000/docs${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"
    echo ""

    # Wait for either process to exit
    trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
    wait $BACKEND_PID $FRONTEND_PID
}

# Main loop
debug_msg "Entering main menu loop"
MENU_ITERATIONS=0

while true; do
    MENU_ITERATIONS=$((MENU_ITERATIONS + 1))
    debug_msg "Menu iteration: $MENU_ITERATIONS"

    show_menu
    read -p "$(echo -e ${CYAN})Enter your choice [1-22]: $(echo -e ${NC})" choice
    debug_msg "User selected option: $choice"

    case $choice in
        1)
            debug_msg "Option 1: Quick Demo selected"
            quick_demo
            debug_msg "Quick demo finished, exiting menu"
            break
            ;;
        2)
            debug_msg "Option 2: Fetch Data selected"
            fetch_data
            debug_msg "Fetch data completed, returning to menu"
            ;;
        3)
            debug_msg "Option 3: Train PINN selected"
            train_pinn
            debug_msg "Train PINN completed, returning to menu"
            ;;
        4)
            debug_msg "Option 4: Train All Models selected"
            train_all
            debug_msg "Train all completed, returning to menu"
            ;;
        5)
            debug_msg "Option 5: Launch Web Interface selected"
            launch_web
            debug_msg "Web interface closed, exiting menu"
            break
            ;;
        6)
            debug_msg "Option 6: Run Tests selected"
            run_tests
            debug_msg "Tests completed, returning to menu"
            ;;
        7)
            debug_msg "Option 7: Start Database selected"
            start_database
            debug_msg "Database started, returning to menu"
            ;;
        8)
            debug_msg "Option 8: Full Docker Stack selected"
            start_docker_stack
            debug_msg "Docker stack closed, exiting menu"
            break
            ;;
        9)
            debug_msg "Option 9: Complete Pipeline selected"
            run_complete_pipeline
            debug_msg "Complete pipeline finished, returning to menu"
            ;;
        10)
            debug_msg "Option 10: Systematic PINN Comparison selected"
            run_pinn_comparison
            debug_msg "PINN comparison finished, returning to menu"
            ;;
        11)
            debug_msg "Option 11: Full Model Pipeline selected"
            run_full_model_pipeline
            debug_msg "Full model pipeline finished, exiting menu"
            break
            ;;
        12)
            debug_msg "Option 12: Train Baseline Models selected"
            train_baseline_models
            debug_msg "Baseline training completed, returning to menu"
            ;;
        13)
            debug_msg "Option 13: Compute Financial Metrics selected"
            compute_financial_metrics
            debug_msg "Financial metrics computation completed, returning to menu"
            ;;
        14)
            debug_msg "Option 14: View Metrics in Terminal selected"
            view_metrics_terminal
            debug_msg "Metrics viewer closed, returning to menu"
            ;;
        15)
            debug_msg "Option 15: Evaluate Stacked/Residual PINN selected"
            evaluate_stacked_residual_pinn
            debug_msg "Stacked/Residual evaluation completed, returning to menu"
            ;;
        16)
            debug_msg "Option 16: Launch Backtesting Dashboard selected"
            launch_backtesting_dashboard
            debug_msg "Backtesting dashboard closed, exiting menu"
            break
            ;;
        17)
            debug_msg "Option 17: Recompute All Metrics selected"
            recompute_all_metrics
            debug_msg "Metrics recomputation completed, returning to menu"
            ;;
        18)
            debug_msg "Option 18: Launch Training Visualizations Dashboard selected"
            launch_training_visualizations
            debug_msg "Training visualizations dashboard closed, exiting menu"
            break
            ;;
        19)
            debug_msg "Option 19: Generate Dissertation Figures selected"
            generate_dissertation_figures
            debug_msg "Dissertation figures generated, returning to menu"
            ;;
        20)
            debug_msg "Option 20: Research-Grade Training selected"
            run_research_training
            debug_msg "Research training completed, returning to menu"
            ;;
        21)
            debug_msg "Option 21: Launch Full Web App selected"
            launch_full_webapp
            debug_msg "Web app closed, exiting menu"
            break
            ;;
        22)
            debug_msg "Option 22: Exit selected"
            echo -e "${GREEN}$(timestamp) Exiting...${NC}"
            debug_msg "Total menu iterations: $MENU_ITERATIONS"
            debug_msg "Total execution time: $SECONDS seconds"
            debug_msg "Log saved to: $LOGFILE"
            exit 0
            ;;
        *)
            debug_msg "Invalid option entered: $choice"
            echo -e "${RED}$(timestamp) [ERROR] Invalid option. Please select 1-22${NC}"
            ;;
    esac
done

debug_msg "Exited main menu loop"
debug_msg "Total runtime: $SECONDS seconds"
debug_msg "Session log: $LOGFILE"
echo ""
echo -e "${CYAN}Session log saved to: $LOGFILE${NC}"
echo -e "${CYAN}Tip: Run with DEBUG=1 ./run.sh to enable verbose debugging${NC}"

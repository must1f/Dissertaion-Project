#!/bin/bash

# Physics-Informed Neural Network (PINN) Financial Forecasting - Setup Script
# ============================================================================

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
LOGFILE="setup_$(date +%Y%m%d_%H%M%S).log"
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

section_header "PINN Financial Forecasting Setup"
debug_msg "Script started"
debug_msg "Logging to: $LOGFILE"
debug_msg "Current directory: $(pwd)"
debug_msg "User: $(whoami)"
debug_msg "Shell: $SHELL"
echo ""

# Check Python version
section_header "STAGE 1/7: Checking Python Installation"
debug_msg "Searching for Python 3 executable..."
echo -e "${YELLOW}Checking Python version...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}$(timestamp) [ERROR] Python 3 is not installed${NC}"
    debug_msg "Python 3 not found in PATH"
    exit 1
fi

debug_msg "Python 3 found at: $(which python3)"
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
debug_msg "Detected Python version: $PYTHON_VERSION"
PYTHON_FULL_VERSION=$(python3 --version)
debug_msg "Full version info: $PYTHON_FULL_VERSION"

REQUIRED_VERSION="3.10"
debug_msg "Required minimum version: $REQUIRED_VERSION"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}$(timestamp) [ERROR] Python $REQUIRED_VERSION+ is required. Found: $PYTHON_VERSION${NC}"
    debug_msg "Version check failed"
    exit 1
fi

echo -e "${GREEN}$(timestamp) ✓ Python $PYTHON_VERSION detected${NC}"
debug_msg "Python version check passed"
echo ""

# Create virtual environment
section_header "STAGE 2/7: Setting Up Virtual Environment"
debug_msg "Checking for existing virtual environment..."
echo -e "${YELLOW}Creating virtual environment...${NC}"

if [ ! -d "venv" ]; then
    debug_msg "Virtual environment not found, creating new one..."
    debug_msg "Running: python3 -m venv venv"
    python3 -m venv venv
    echo -e "${GREEN}$(timestamp) ✓ Virtual environment created${NC}"
    debug_msg "Virtual environment created successfully"
else
    echo -e "${GREEN}$(timestamp) ✓ Virtual environment already exists${NC}"
    debug_msg "Using existing virtual environment at: $(pwd)/venv"
fi

# Check venv directory structure
debug_msg "Virtual environment structure:"
debug_msg "  - bin: $(ls -la venv/bin/ 2>/dev/null | wc -l) files"
debug_msg "  - lib: $(ls -la venv/lib/ 2>/dev/null | wc -l) files"
echo ""

# Activate virtual environment
echo -e "${YELLOW}$(timestamp) Activating virtual environment...${NC}"
debug_msg "Running: source venv/bin/activate"

if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}$(timestamp) [ERROR] venv/bin/activate not found${NC}"
    exit 1
fi

source venv/bin/activate
echo -e "${GREEN}$(timestamp) ✓ Virtual environment activated${NC}"
debug_msg "VIRTUAL_ENV=$VIRTUAL_ENV"
debug_msg "Python executable: $(which python3)"
debug_msg "Pip executable: $(which pip)"
echo ""

# Upgrade pip
section_header "STAGE 3/7: Upgrading pip"
debug_msg "Current pip version: $(pip --version)"
echo -e "${YELLOW}$(timestamp) Upgrading pip...${NC}"
debug_msg "Running: pip install --upgrade pip"

pip install --upgrade pip 2>&1 | while IFS= read -r line; do
    debug_msg "pip upgrade: $line"
done

echo -e "${GREEN}$(timestamp) ✓ pip upgraded${NC}"
debug_msg "New pip version: $(pip --version)"
echo ""

# Install requirements
section_header "STAGE 4/7: Installing Python Dependencies"
debug_msg "Checking requirements.txt..."

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}$(timestamp) [ERROR] requirements.txt not found${NC}"
    debug_msg "Cannot proceed without requirements.txt"
    exit 1
fi

TOTAL_PACKAGES=$(grep -v "^#" requirements.txt | grep -v "^$" | wc -l | tr -d ' ')
debug_msg "Found $TOTAL_PACKAGES packages to install"
debug_msg "Requirements file size: $(wc -c < requirements.txt) bytes"

echo -e "${YELLOW}$(timestamp) Installing Python dependencies...${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"
echo -e "${CYAN}Installing $TOTAL_PACKAGES packages${NC}"
debug_msg "Running: pip install -r requirements.txt"

# Install with progress tracking
SECONDS=0
pip install -r requirements.txt 2>&1 | while IFS= read -r line; do
    # Only show important messages, not every detail
    if [[ "$line" =~ "Collecting" ]] || [[ "$line" =~ "Installing" ]] || [[ "$line" =~ "Successfully" ]]; then
        echo -e "${CYAN}  $(timestamp) $line${NC}"
    fi
    debug_msg "pip: $line"
done

INSTALL_TIME=$SECONDS
echo -e "${GREEN}$(timestamp) ✓ Dependencies installed${NC}"
debug_msg "Installation completed in $INSTALL_TIME seconds"
debug_msg "Installed packages: $(pip list | wc -l | tr -d ' ')"
echo ""

# Check for .env file
section_header "STAGE 5/7: Configuring Environment Variables"
debug_msg "Checking for .env file..."
echo -e "${YELLOW}$(timestamp) Checking environment variables...${NC}"

if [ ! -f ".env" ]; then
    debug_msg ".env file not found"
    if [ -f ".env.example" ]; then
        debug_msg ".env.example found, creating .env..."
        echo -e "${YELLOW}$(timestamp) Creating .env from .env.example...${NC}"
        cp .env.example .env
        debug_msg "Copied .env.example to .env"
        echo -e "${GREEN}$(timestamp) ✓ .env file created${NC}"
        echo -e "${YELLOW}⚠ Please edit .env and add your API keys${NC}"
        debug_msg ".env file size: $(wc -c < .env 2>/dev/null || echo 0) bytes"
    else
        debug_msg ".env.example not found"
        echo -e "${YELLOW}$(timestamp) ⚠ No .env.example found. You may need to create .env manually${NC}"
    fi
else
    echo -e "${GREEN}$(timestamp) ✓ .env file exists${NC}"
    debug_msg ".env file found at: $(pwd)/.env"
    debug_msg ".env file size: $(wc -c < .env) bytes"
    debug_msg ".env variables count: $(grep -c "=" .env 2>/dev/null || echo 0)"
fi
echo ""

# Check for Docker
section_header "STAGE 6/7: Checking Docker Installation"
debug_msg "Searching for Docker executable..."
echo -e "${YELLOW}$(timestamp) Checking Docker installation...${NC}"

if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}$(timestamp) ✓ Docker is installed${NC}"
    debug_msg "Docker version: $DOCKER_VERSION"
    debug_msg "Docker path: $(which docker)"

    # Check Docker daemon
    debug_msg "Checking Docker daemon status..."
    if docker ps &> /dev/null; then
        debug_msg "Docker daemon is running"

        # Check for docker-compose
        if command -v docker-compose &> /dev/null; then
            COMPOSE_VERSION=$(docker-compose --version)
            debug_msg "Docker Compose version: $COMPOSE_VERSION"
        else
            debug_msg "Docker Compose not found (may use 'docker compose' instead)"
        fi

        # Ask user if they want to start the database
        echo ""
        read -p "$(echo -e ${YELLOW})Do you want to start the TimescaleDB database with Docker? (y/n) $(echo -e ${NC})" -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}$(timestamp) Starting TimescaleDB...${NC}"
            debug_msg "Running: docker-compose up -d timescaledb"

            if docker-compose up -d timescaledb 2>&1 | while IFS= read -r line; do
                debug_msg "docker-compose: $line"
                echo -e "${CYAN}  $line${NC}"
            done; then
                echo -e "${GREEN}$(timestamp) ✓ Database started${NC}"
                echo "Database will be available at localhost:5432"
                debug_msg "TimescaleDB container started successfully"

                # Wait for database to be ready
                echo -e "${CYAN}$(timestamp) Waiting for database to be ready...${NC}"
                sleep 3
                debug_msg "Database initialization delay completed"
            else
                echo -e "${RED}$(timestamp) [ERROR] Failed to start database${NC}"
                debug_msg "Docker compose failed"
            fi
        else
            debug_msg "User chose not to start database"
        fi
    else
        echo -e "${YELLOW}$(timestamp) ⚠ Docker daemon is not running${NC}"
        debug_msg "Docker daemon check failed"
    fi
else
    echo -e "${YELLOW}$(timestamp) ⚠ Docker not found. Database functionality will be limited.${NC}"
    echo "  Install Docker from: https://www.docker.com/get-started"
    debug_msg "Docker not installed"
fi
echo ""

# Create necessary directories
section_header "STAGE 7/7: Creating Project Directories"
debug_msg "Setting up project directory structure..."
echo -e "${YELLOW}$(timestamp) Creating project directories...${NC}"

DIRECTORIES=("data" "checkpoints" "results" "logs")
for dir in "${DIRECTORIES[@]}"; do
    debug_msg "Creating directory: $dir"
    if mkdir -p "$dir" 2>/dev/null; then
        echo -e "${CYAN}  ✓ Created: $dir/${NC}"
        debug_msg "  Size: $(du -sh "$dir" 2>/dev/null | cut -f1)"
        debug_msg "  Permissions: $(ls -ld "$dir" | cut -d' ' -f1)"
    else
        echo -e "${RED}  ✗ Failed to create: $dir${NC}"
        debug_msg "  Failed to create $dir"
    fi
done

echo -e "${GREEN}$(timestamp) ✓ Directories created${NC}"
debug_msg "Directory structure complete"
echo ""

# Setup complete
section_header "Setup Complete!"
debug_msg "All stages completed successfully"
debug_msg "Total execution time: $SECONDS seconds"
debug_msg "Log file saved to: $LOGFILE"

# Display summary
echo -e "${GREEN}$(timestamp) ✓ Setup completed successfully!${NC}"
echo ""
echo "========================================"
echo "SETUP SUMMARY"
echo "========================================"
echo -e "${CYAN}Python Version:${NC} $(python3 --version)"
echo -e "${CYAN}Virtual Environment:${NC} $VIRTUAL_ENV"
echo -e "${CYAN}Pip Version:${NC} $(pip --version | cut -d' ' -f2)"
echo -e "${CYAN}Installed Packages:${NC} $(pip list | wc -l | tr -d ' ')"
echo -e "${CYAN}Log File:${NC} $LOGFILE"
echo "========================================"
echo ""

debug_msg "Displaying next steps to user..."
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo -e "   ${CYAN}source venv/bin/activate${NC}"
echo ""
echo "2. Edit .env file and add your API keys (if needed):"
echo -e "   ${CYAN}nano .env${NC} or ${CYAN}vim .env${NC}"
echo ""
echo "3. Run the system using run.sh:"
echo -e "   ${CYAN}./run.sh${NC}"
echo ""
echo "Or run components manually:"
echo -e "   ${CYAN}python3 -m src.data.fetcher${NC}           # Fetch data"
echo -e "   ${CYAN}python3 -m src.training.train --help${NC}  # Train models"
echo -e "   ${CYAN}streamlit run src/web/app.py${NC}          # Launch web UI"
echo ""
echo -e "${YELLOW}⚠ Remember: This is for academic research only - NOT financial advice${NC}"
echo ""

# Enable debug mode for future runs
echo -e "${CYAN}Tip: Run with DEBUG=1 ./setup.sh to enable verbose debugging${NC}"
echo ""

debug_msg "Setup script finished"

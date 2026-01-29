#!/bin/bash
# Launch Monte Carlo Simulation Dashboard
#
# Usage:
#   ./launch_monte_carlo.sh              # Launch web dashboard
#   ./launch_monte_carlo.sh --terminal   # Run terminal visualization

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Monte Carlo Simulation Visualization                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if terminal mode
if [ "$1" == "--terminal" ]; then
    echo -e "${GREEN}Running Terminal Visualization...${NC}"
    echo ""
    python visualize_monte_carlo.py --synthetic --stress-test --horizon 30 --n-simulations 1000
else
    echo -e "${GREEN}Launching Web Dashboard...${NC}"
    echo ""
    echo "Opening at: http://localhost:8503"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    streamlit run src/web/monte_carlo_dashboard.py --server.port 8503
fi

#!/bin/bash

# PINN Dashboard Launcher
# Launches the comprehensive PINN comparison web application

echo "================================="
echo "PINN Dashboard Launcher"
echo "================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly pandas numpy
fi

# Navigate to project directory
cd "$(dirname "$0")"

echo "✓ Starting PINN Dashboard..."
echo ""
echo "📊 Dashboard Options:"
echo "  1. Main App (all features)"
echo "  2. Dedicated PINN Dashboard"
echo ""

read -p "Select option (1 or 2, default=1): " choice
choice=${choice:-1}

if [ "$choice" = "2" ]; then
    echo ""
    echo "Launching dedicated PINN dashboard..."
    streamlit run src/web/pinn_dashboard.py
else
    echo ""
    echo "Launching main application..."
    streamlit run src/web/app.py
fi

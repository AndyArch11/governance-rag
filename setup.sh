#!/bin/bash
# setup.sh - Initial setup script for Governance Intelligence Console
# This script sets up the development environment, creates necessary directories,
# and prepares the system for first run.

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  Governance Intelligence Console - Initial Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Colour

# Determine project root (where this script lives)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}[1/7]${NC} Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION detected (>= $REQUIRED_VERSION required)"
else
    echo -e "${RED}✗${NC} Python $PYTHON_VERSION is too old. Python >= $REQUIRED_VERSION required."
    exit 1
fi

# Check if virtual environment exists
echo ""
echo -e "${BLUE}[2/7]${NC} Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓${NC} Virtual environment created at .venv/"
else
    echo -e "${YELLOW}⚠${NC} Virtual environment already exists, skipping creation"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
echo ""
echo -e "${BLUE}[3/7]${NC} Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo -e "${GREEN}✓${NC} pip upgraded"

# Install dependencies
echo ""
echo -e "${BLUE}[4/7]${NC} Installing dependencies..."
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${RED}✗${NC} requirements.txt not found!"
    exit 1
fi

# Install development dependencies (optional)
read -p "Install development dependencies (pytest, jupyter, etc.)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install pytest pytest-cov jupyter ipykernel --quiet
    echo -e "${GREEN}✓${NC} Development dependencies installed"
fi

# Create project directories
echo ""
echo -e "${BLUE}[5/7]${NC} Creating project directories..."
mkdir -p data_raw/downloads
mkdir -p data_raw/url_imports
mkdir -p data_raw/public_docs
mkdir -p logs
mkdir -p models
mkdir -p notebooks
mkdir -p scripts/ingest
mkdir -p scripts/rag
mkdir -p scripts/consistency_graph
mkdir -p scripts/utils
mkdir -p tests
mkdir -p rag_data/chromadb  # ChromaDB persistent storage
mkdir -p rag_data/cache     # LLM and embedding caches
mkdir -p rag_data/consistency_graphs  # Consistency graph outputs
echo -e "${GREEN}✓${NC} Project directories created"

# Set up .env file
echo ""
echo -e "${BLUE}[6/7]${NC} Setting up environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo -e "${GREEN}✓${NC} .env file created from template"
        echo -e "${YELLOW}⚠${NC} Edit .env to customise paths and settings (optional)"
    else
        echo -e "${YELLOW}⚠${NC} .env.example not found, skipping .env creation"
    fi
else
    echo -e "${YELLOW}⚠${NC} .env already exists, skipping"
fi

# Check for Ollama
echo ""
echo -e "${BLUE}[7/7]${NC} Checking for Ollama..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | head -n1)
    echo -e "${GREEN}✓${NC} Ollama detected: $OLLAMA_VERSION"
    
    # Check if required models are available
    echo "Checking for required models..."
    MODELS_NEEDED=("mistral" "mxbai-embed-large")
    MISSING_MODELS=()
    
    for model in "${MODELS_NEEDED[@]}"; do
        if ollama list | grep -q "^$model"; then
            echo -e "  ${GREEN}✓${NC} $model"
        else
            echo -e "  ${RED}✗${NC} $model (not found)"
            MISSING_MODELS+=("$model")
        fi
    done
    
    if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}⚠${NC} Missing models detected. To install them, run:"
        for model in "${MISSING_MODELS[@]}"; do
            echo "    ollama pull $model"
        done
    fi
else
    echo -e "${RED}✗${NC} Ollama not found!"
    echo ""
    echo "This project requires Ollama for LLM and embedding operations."
    echo "Install Ollama from: https://ollama.ai"
    echo ""
    echo "After installing Ollama, pull the required models:"
    echo "    ollama pull mistral"
    echo "    ollama pull mxbai-embed-large"
fi

# Final instructions
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo -e "   ${BLUE}source .venv/bin/activate${NC}"
echo ""
echo "2. (Optional) Review and customise .env file:"
echo -e "   ${BLUE}nano .env${NC}"
echo ""
echo "3. Add your HTML and PDF documents to be ingested to:"
echo -e "   ${BLUE}data_raw/downloads/${NC}"
echo ""
echo "4. Run the ingestion pipeline:"
echo -e "   ${BLUE}make ingest${NC}"
echo "   or:"
echo -e "   ${BLUE}python scripts/ingest/ingest.py${NC}"
echo ""
echo "5. Build the consistency graph:"
echo -e "   ${BLUE}make graph${NC}"
echo "   or:"
echo -e "   ${BLUE}python scripts/consistency_graph/build_consistency_graph.py${NC}"
echo ""
echo "6. Launch the dashboard:"
echo -e "   ${BLUE}make dashboard${NC}"
echo "   or:"
echo -e "   ${BLUE}python scripts/ui/dashboard.py${NC}"
echo ""
echo "For help with available commands:"
echo -e "   ${BLUE}make help${NC}"
echo ""
echo "════════════════════════════════════════════════════════════════"

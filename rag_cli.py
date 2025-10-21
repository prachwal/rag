#!/usr/bin/env python3
"""
RAG Application CLI Entry Point

Main entry point for the RAG (Retrieval-Augmented Generation) application CLI.
This file only contains the application startup logic.
"""

import sys
from pathlib import Path

# Add Common to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rag_cli.cli import main
except ImportError as e:
    print(f"Error importing CLI module: {e}", file=sys.stderr)
    print("Make sure you're running from the project root directory.", file=sys.stderr)
    sys.exit(1)

if __name__ == '__main__':
    main()
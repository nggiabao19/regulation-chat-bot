#!/usr/bin/env python3
"""
Wrapper script to run the chatbot query interface.
Usage: python query.py
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.query.query_data import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Wrapper script to create/update the database.
Usage: python create_db.py [--reset] [--data path/to/pdfs]
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database.create_database import main

if __name__ == "__main__":
    main()

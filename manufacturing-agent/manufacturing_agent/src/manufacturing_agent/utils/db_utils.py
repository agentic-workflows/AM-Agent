"""
Database Utilities

Robust database path handling and initialization utilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional


def get_robust_db_path(preferred_name: str = "manufacturing_runs.db") -> str:
    """
    Get a robust database path that works across different deployment scenarios.
    
    Tries multiple locations in order of preference:
    1. Project output directory (if exists)
    2. User's home directory cache
    3. System temp directory
    
    Args:
        preferred_name: Name of the database file
        
    Returns:
        Absolute path to database file
    """
    
    # Option 1: Try project output directory
    try:
        # Get the project root by going up from this file
        project_root = Path(__file__).resolve().parents[4]  # utils -> manufacturing_agent -> src -> manufacturing_agent -> project_root
        output_dir = project_root / "manufacturing-agent" / "manufacturing_agent" / "output"
        
        if output_dir.exists() or _try_create_dir(output_dir):
            db_path = output_dir / preferred_name
            if _test_db_access(db_path):
                return str(db_path)
    except Exception:
        pass
    
    # Option 2: Try user's home directory cache
    try:
        home_cache = Path.home() / ".cache" / "am_agent"
        if home_cache.exists() or _try_create_dir(home_cache):
            db_path = home_cache / preferred_name
            if _test_db_access(db_path):
                print(f"ℹ️  Using user cache directory for database: {db_path}")
                return str(db_path)
    except Exception:
        pass
    
    # Option 3: Fallback to temp directory
    temp_dir = Path(tempfile.gettempdir()) / "am_agent_cache"
    if temp_dir.exists() or _try_create_dir(temp_dir):
        db_path = temp_dir / preferred_name
        print(f"⚠️  Using temporary directory for database: {db_path}")
        print("   Note: Database will be lost when system restarts.")
        return str(db_path)
    
    # Final fallback: temp file
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".db", prefix="am_agent_")
    os.close(fd)
    print(f"⚠️  Using temporary file for database: {path}")
    print("   Note: Database will be lost when system restarts.")
    return path


def _try_create_dir(path: Path) -> bool:
    """Try to create directory, return True if successful."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def _test_db_access(db_path: Path) -> bool:
    """Test if we can create/access a database at the given path."""
    try:
        import sqlite3
        # Try to create a connection and a simple table
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
            conn.execute("DROP TABLE IF EXISTS test")
        return True
    except Exception:
        return False


def ensure_db_directory(db_path: str) -> bool:
    """
    Ensure the directory for a database path exists.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

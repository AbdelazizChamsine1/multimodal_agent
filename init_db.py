"""
Database initialization script for pgvector RAG system.

This script:
1. Creates the PostgreSQL database if it doesn't exist
2. Enables the pgvector extension
3. Verifies the setup

Usage:
    python init_db.py
"""

import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from colorama import Fore, Style
from config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD
)


def create_database():
    """Create the database if it doesn't exist."""
    # Connect to default 'postgres' database to create our target database
    admin_connection_string = (
        f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"
    )

    try:
        engine = create_engine(admin_connection_string, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": POSTGRES_DB}
            )

            if result.fetchone() is None:
                print(Fore.YELLOW + f"[INFO] Creating database '{POSTGRES_DB}'...")
                conn.execute(text(f'CREATE DATABASE "{POSTGRES_DB}"'))
                print(Fore.GREEN + f"[INFO] Database '{POSTGRES_DB}' created successfully!")
            else:
                print(Fore.GREEN + f"[INFO] Database '{POSTGRES_DB}' already exists.")

        engine.dispose()
        return True

    except OperationalError as e:
        print(Fore.RED + f"[ERROR] Could not connect to PostgreSQL: {e}")
        print(Fore.YELLOW + "\nPlease ensure:")
        print("  1. PostgreSQL is running")
        print(f"  2. User '{POSTGRES_USER}' exists and has proper permissions")
        print(f"  3. Connection details in .env are correct")
        return False
    except Exception as e:
        print(Fore.RED + f"[ERROR] Failed to create database: {e}")
        return False


def enable_pgvector():
    """Enable pgvector extension in the target database."""
    connection_string = (
        f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            print(Fore.YELLOW + "[INFO] Enabling pgvector extension...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print(Fore.GREEN + "[INFO] pgvector extension enabled successfully!")

            # Verify installation
            result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
            version = result.fetchone()
            if version:
                print(Fore.GREEN + f"[INFO] pgvector version: {version[0]}")

        engine.dispose()
        return True

    except ProgrammingError as e:
        print(Fore.RED + f"[ERROR] pgvector extension not available: {e}")
        print(Fore.YELLOW + "\nPlease install pgvector:")
        print("  - Ubuntu/Debian: sudo apt install postgresql-pgvector")
        print("  - macOS: brew install pgvector")
        print("  - Windows: Download from https://github.com/pgvector/pgvector/releases")
        print("  - Docker: Use postgres image with pgvector (e.g., pgvector/pgvector:pg16)")
        return False
    except Exception as e:
        print(Fore.RED + f"[ERROR] Failed to enable pgvector: {e}")
        return False


def verify_setup():
    """Verify the database setup is complete."""
    connection_string = (
        f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Test vector operations
            conn.execute(text("SELECT '[1,2,3]'::vector"))
            print(Fore.GREEN + "[INFO] Vector operations working correctly!")

        engine.dispose()
        return True

    except Exception as e:
        print(Fore.RED + f"[ERROR] Setup verification failed: {e}")
        return False


def main():
    """Main initialization flow."""
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "PostgreSQL + pgvector Database Initialization")
    print(Fore.CYAN + "=" * 60 + "\n")

    print(Fore.YELLOW + f"Database: {POSTGRES_DB}")
    print(Fore.YELLOW + f"Host: {POSTGRES_HOST}:{POSTGRES_PORT}")
    print(Fore.YELLOW + f"User: {POSTGRES_USER}\n")

    # Step 1: Create database
    if not create_database():
        sys.exit(1)

    # Step 2: Enable pgvector
    if not enable_pgvector():
        sys.exit(1)

    # Step 3: Verify setup
    if not verify_setup():
        sys.exit(1)

    print(Fore.GREEN + "\n" + "=" * 60)
    print(Fore.GREEN + "Database initialization complete!")
    print(Fore.GREEN + "You can now run: python main.py")
    print(Fore.GREEN + "=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nInitialization cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(Fore.RED + f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)

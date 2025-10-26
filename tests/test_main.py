import pytest
from pathlib import Path
from src.main import SQLQueryGenerator
import os

@pytest.fixture
def query_generator():
    return SQLQueryGenerator()

def test_simple_query(query_generator):
    """Test a simple count query"""
    result = query_generator.run("How many customers are there?")
    assert result is not None
    assert isinstance(result, dict)

def test_database_connection():
    """Test that the database file exists and is accessible"""
    db_path = os.getenv("DATABASE_PATH")
    assert db_path is not None
    assert Path(db_path).exists(), "Database file not found"

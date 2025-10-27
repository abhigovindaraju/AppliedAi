import os
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tool_get_tables import get_tables_from_db

@pytest.fixture
def test_db_path(tmp_path):
    # Create a temporary SQLite database
    db_path = tmp_path / "test.db"
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create test tables
    cursor.execute('''
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            name TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    
    return str(db_path)

def test_get_tables_from_db(test_db_path):
    schema = get_tables_from_db(test_db_path)
    
    assert isinstance(schema, dict)
    assert 'products' in schema
    assert 'categories' in schema
    
    assert set(schema['products']) == {'product_id', 'name', 'price'}
    assert set(schema['categories']) == {'category_id', 'name'}

def test_get_tables_from_nonexistent_db():
    schema = get_tables_from_db('nonexistent.db')
    assert schema == {}

import os
import pytest
import sqlite3
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sql_connector import SQLConnector

# Fixture to create a test database
@pytest.fixture
def test_db():
    # Create a test database in memory
    db_path = ":memory:"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create test tables
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            total REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    
    return db_path

def test_sql_connector_initialization(test_db):
    connector = SQLConnector(test_db)
    assert connector.conn is not None

def test_get_tables_from_db(test_db):
    connector = SQLConnector(test_db)
    schema = connector.get_tables_from_db()
    
    assert 'users' in schema
    assert 'orders' in schema
    assert schema['users'] == ['id', 'name', 'email']
    assert schema['orders'] == ['order_id', 'user_id', 'total']

def test_execute_query(test_db):
    connector = SQLConnector(test_db)
    
    # Insert test data
    connector.execute_query('''
        INSERT INTO users (id, name, email) 
        VALUES (1, 'Test User', 'test@example.com')
    ''')
    
    # Query the data
    result = connector.execute_query('SELECT * FROM users WHERE id = 1')
    assert result is not None
    assert len(result) == 1
    assert result[0] == (1, 'Test User', 'test@example.com')

def test_execute_invalid_query(test_db):
    connector = SQLConnector(test_db)
    result = connector.execute_query('SELECT * FROM nonexistent_table')
    assert result is None

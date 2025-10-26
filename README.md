# SQL Query Generator

This application converts natural language questions into SQL queries and executes them against a SQLite database using LangChain, LangGraph, and Google's Gemini LLM.

## Setup

1. Copy `.env.template` to `.env` and fill in your API keys:
```bash
cp .env.template .env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure your database file is accessible at the path specified in `.env`

## Usage

Run the application with a question:
```bash
python src/main.py query "How many customers are there in the database?"
```

## Features

- Natural language to SQL query conversion
- Direct database querying
- Rich formatted output
- Error handling and validation
- Uses Gemini Pro for accurate query generation

## Database Schema

This application works with the Chinook database, which includes tables for:
- Customers
- Employees
- Invoices
- Tracks
- Albums
- Artists
- and more...

## Development

To run tests:
```bash
pytest tests/
```

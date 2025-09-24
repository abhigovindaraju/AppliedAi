# Document Analysis Tool

A sophisticated document analysis tool that uses various embedding models and chunking strategies to process PDF documents and enable semantic search capabilities. The tool is particularly useful for analyzing large documents like financial reports (10-Q), technical documentation, or research papers.

## Features

### 1. Multiple Embedding Models
- **BERT**: Transformer-based model for contextual embeddings
  - Best for understanding context and relationships
  - Handles complex queries effectively
  - Pre-trained on large text corpora

- **Word2Vec**: Traditional word embedding model
  - Fast and efficient for simple queries
  - Good for keyword-based search
  - Lightweight implementation

- **Ada**: OpenAI's embedding model
  - State-of-the-art semantic understanding
  - Excellent for complex relationships
  - Requires API key for access

### 2. Advanced Chunking Strategies
- **Fixed Size**: 
  - Consistent chunk sizes
  - Predictable performance
  - Best for uniform documents

- **Semantic**: 
  - Context-aware splitting
  - Preserves meaning
  - Ideal for natural text

- **Recursive**: 
  - Hierarchical document analysis
  - Maintains document structure
  - Great for nested content

- **Adaptive**: 
  - Dynamic chunk sizing
  - Content-aware splitting
  - Optimizes for document structure

### 3. Smart Configuration
- **Relevancy Control**: 
  - Adjustable minimum relevancy score
  - Fine-tune result precision
  - Balance recall vs. precision

- **Result Management**:
  - Configurable chunk count
  - Customizable chunk sizes
  - Optimized for performance

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/abhigovindaraju/AppliedAi.git
   cd AppliedAI/Document_Analyzer
   ```

2. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install NLP Dependencies**:
   ```python
   # Download required NLTK data
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('maxent_ne_chunker')
   nltk.download('words')
   
   # Install spaCy model
   python -m spacy download en_core_web_sm
   ```

4. **Configure API Keys** (Optional):
   ```bash
   # Create .env file for Ada embeddings
   echo "GOOGLE_API_KEY=your-api-key-here" > .env
   ```

## Usage Guide

### 1. Interactive Mode

Perfect for exploring documents and running ad-hoc queries:

```bash
python main.py --folder /path/to/pdfs --embedding bert --chunking semantic
```

Example session:
```
Document Analysis Tool
Enter your queries (type 'exit' to quit)

Query: What were the key financial metrics for Q3?

Results:
==================================================

1. Relevancy Score: 0.92
------------------------------
Key financial metrics for Q3 2024:
- Revenue: $81.8 billion (+2.1% YoY)
- Operating Income: $23.0 billion
- Net Income: $19.9 billion
- EPS: $1.26 per share

2. Relevancy Score: 0.85
------------------------------
Year-over-year performance indicators:
- Gross Margin: 44.1% (↑ 0.6%)
- Operating Margin: 28.2% (↑ 0.4%)
...
```

### 2. Evaluation Mode

For benchmarking and optimizing configurations:

```bash
python main.py --folder /path/to/pdfs --evaluate
```

This generates:

1. **Interactive HTML Report**:
   - Performance charts
   - Model comparisons
   - Strategy analysis
   - Best configurations

2. **Detailed Metrics**:
   ```
   Evaluation Results:
   ==================================================
   Best Configuration:
   - Model: BERT
   - Strategy: Semantic
   - Avg Query Time: 0.24s
   - Avg Relevancy: 0.87
   
   Model Performance:
   - BERT:    0.87 relevancy, 0.24s/query
   - Word2Vec: 0.82 relevancy, 0.18s/query
   - Ada:     0.91 relevancy, 0.45s/query
   ```

### 3. Custom Configuration

Fine-tune the analysis with detailed parameters:

```bash
python main.py \
  --folder /path/to/pdfs \
  --embedding bert \
  --chunking semantic \
  --min-score 0.7 \
  --max-chunks 5 \
  --chunk-size 1000
```

#### Configuration Parameters

| Parameter    | Description                  | Default | Options                    |
|-------------|------------------------------|---------|----------------------------|
| embedding   | Embedding model to use       | bert    | word2vec, bert, ada       |
| chunking    | Text chunking strategy      | semantic | fixed, semantic, recursive, adaptive |
| min-score   | Minimum relevancy (0-1)     | 0.7     | Any float between 0 and 1 |
| max-chunks  | Maximum results to return    | 5       | Any positive integer      |
| chunk-size  | Words per chunk             | 1000    | Any positive integer      |

## Output & Reports

### 1. Interactive Results

Query results are displayed with:
- Relevancy scores (0-1)
- Sorted by relevance
- Highlighted matches
- Context preservation

### 2. Evaluation Reports

Located in `Logs/` directory:

1. **HTML Reports** (`evaluation_report_*.html`):
   - Interactive charts
   - Performance metrics
   - Configuration comparisons
   - Query analysis

2. **JSON Summaries** (`evaluation_summary_*.json`):
   - Detailed metrics
   - Best configurations
   - Model comparisons
   - Raw data

3. **CSV Results** (`evaluation_results_*.csv`):
   - Complete test data
   - All configurations
   - Raw metrics
   - Query-by-query results

### 3. Example Queries
- Comparative Analysis
- Risk Analysis
- Segmentation
- Trend Analysis
- Investment Analysis
- Balance Sheet Analysis
- Financial Structure
- Management Analysis
- Future Outlook

### Metrics Tracked

1. **Performance Metrics:**
   - Average query time
   - Average chunks per query
   - Average relevancy score

2. **Best Configurations:**
   - Fastest configuration
   - Most relevant configuration
   - Balanced configuration (speed vs. relevance)

### Output Files

1. `evaluation_results_{timestamp}.csv`:
   - Detailed results for each configuration
   - Individual query performance
   - Aggregate metrics

2. `evaluation_summary_{timestamp}.json`:
   - Best configurations
   - Model comparisons
   - Strategy comparisons

## Embedding Models

1. **Word2Vec**
   - Static word embeddings
   - Fast and lightweight
   - Good for general-purpose analysis

2. **BERT**
   - Contextual embeddings
   - Better understanding of context
   - More resource-intensive

3. **Ada**
   - OpenAI's powerful embedding model
   - Best semantic understanding
   - Requires API key and internet connection

## Chunking Strategies

1. **Fixed-Size Chunking**
   - Simple, consistent chunks
   - Based purely on word count
   - May break semantic units

2. **Semantic Chunking**
   - Respects sentence and paragraph boundaries
   - Maintains context
   - Variable chunk sizes

3. **Recursive Chunking**
   - Hierarchical approach
   - Includes summarization
   - Good for very long documents

4. **Adaptive Chunking**
   - Adjusts based on content complexity
   - Considers named entities and structure
   - Most flexible approach

## Example Queries

```bash
> Enter your query: What was the revenue in Q1 2023?
> Enter your query: How do the operating expenses compare to last year?
> Enter your query: What are the main risk factors mentioned?
```

## Performance Considerations

- BERT embeddings require more memory but provide better context understanding
- Ada embeddings provide best results but require API calls
- Semantic and Adaptive chunking are more compute-intensive
- Adjust chunk size based on available memory and document length

## Error Handling

The application handles:
- Missing PDF files
- Invalid PDF formats
- API failures
- Memory constraints
- Invalid queries

## Limitations

- PDF text extraction quality depends on PDF formatting
- Some embedding models require significant memory
- Ada embeddings require internet connection
- Processing very large documents may be slow

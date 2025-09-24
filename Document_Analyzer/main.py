import os
import sys
import PyPDF2
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
import argparse
from pathlib import Path
from datetime import datetime

from embedding_models import Word2VecEmbedding, BertEmbedding, AdaEmbedding
from chunking_strategies import (
    FixedSizeChunking,
    SemanticChunking,
    RecursiveChunking,
    AdaptiveChunking
)
from performance_evaluator import PerformanceEvaluator

load_dotenv()

class DocumentAnalyzer:
    def __init__(
        self,
        embedding_model: str = "bert",
        chunking_strategy: str = "semantic",
        min_relevancy_score: float = 0.7,
        max_chunk_count: int = 5,
        chunk_size: int = 1000
    ):
        # Initialize embedding model
        self.embedding_model = self._get_embedding_model(embedding_model)
        
        # Initialize chunking strategy
        self.chunking_strategy = self._get_chunking_strategy(chunking_strategy)
        
        # Set parameters
        self.min_relevancy_score = min_relevancy_score
        self.max_chunk_count = max_chunk_count
        self.chunk_size = chunk_size
        
        # Storage for document chunks and embeddings
        self.chunks = []
        self.chunk_embeddings = []
        
    def _get_embedding_model(self, model_name: str):
        models = {
            "word2vec": Word2VecEmbedding(),
            "bert": BertEmbedding(),
            "ada": AdaEmbedding()
        }
        return models.get(model_name.lower())
    
    def _get_chunking_strategy(self, strategy_name: str):
        strategies = {
            "fixed": FixedSizeChunking(),
            "semantic": SemanticChunking(),
            "recursive": RecursiveChunking(),
            "adaptive": AdaptiveChunking()
        }
        return strategies.get(strategy_name.lower())
    
    def load_documents(self, folder_path: str):
        """Load and process all PDF documents from the specified folder"""
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    # Chunk the document
                    new_chunks = self.chunking_strategy.chunk_text(text, self.chunk_size)
                    self.chunks.extend(new_chunks)
                    
                    # Generate embeddings for chunks
                    for chunk in new_chunks:
                        embedding = self.embedding_model.embed(chunk)
                        self.chunk_embeddings.append(embedding)
                        
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant chunks based on the query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed(query)
            
            # Calculate cosine similarities
            similarities = []
            results = []
            
            for chunk, chunk_embedding in zip(self.chunks, self.chunk_embeddings):
                try:
                    if query_embedding.shape != chunk_embedding.shape:
                        print(f"Warning: Shape mismatch - Query: {query_embedding.shape}, Chunk: {chunk_embedding.shape}")
                        continue
                        
                    similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    if similarity >= self.min_relevancy_score:
                        results.append({
                            "text": chunk,
                            "relevancy_score": float(similarity)
                        })
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
            
            # Sort by relevancy score and limit results
            results.sort(key=lambda x: x["relevancy_score"], reverse=True)
            return results[:self.max_chunk_count]
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Analysis Tool")
    parser.add_argument("--folder", type=str, help="Path to folder containing PDF documents")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on the analyzer")
    parser.add_argument("--embedding", choices=["word2vec", "bert", "ada"],
                      default="bert", help="Embedding model to use")
    parser.add_argument("--chunking", choices=["fixed", "semantic", "recursive", "adaptive"],
                      default="semantic", help="Chunking strategy to use")
    parser.add_argument("--min-score", type=float, default=0.7,
                      help="Minimum relevancy score (0-1)")
    parser.add_argument("--max-chunks", type=int, default=5,
                      help="Maximum number of chunks to return")
    parser.add_argument("--chunk-size", type=int, default=1000,
                      help="Size of text chunks")
    
    args = parser.parse_args()

    if not args.folder:
        print("Please provide a folder path containing PDF documents")
        sys.exit(1)

    # Initialize analyzer with arguments
    analyzer = DocumentAnalyzer(
        embedding_model=args.embedding,
        chunking_strategy=args.chunking,
        min_relevancy_score=args.min_score,
        max_chunk_count=args.max_chunks,
        chunk_size=args.chunk_size
    )

    # Load documents
    analyzer.load_documents(args.folder)

    if args.evaluate:
        # Run evaluation
        evaluator = PerformanceEvaluator(analyzer)
        results = evaluator.run_evaluation()
        
        # Get the latest summary file
        latest_summary = os.path.join('Logs', 'evaluation_summary_latest.json')
        if os.path.exists(latest_summary):
            # Generate HTML report name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"evaluation_report_{timestamp}.html"
            report_path = os.path.join('Logs', report_name)
            
            # Copy template and update with results
            with open('evaluation_report_template.html', 'r') as template_file:
                template = template_file.read()
                
            with open(report_path, 'w') as report_file:
                report_file.write(template)
            
            print("\nEvaluation Results:")
            print("=" * 50)
            print(results)
            print(f"\nDetailed HTML report generated: {report_path}")
            
            # Open the report in the default browser
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(report_path)}')
    else:
        # Interactive mode
        print("\nDocument Analysis Tool")
        print("Enter your queries (type 'exit' to quit)")
        while True:
            query = input("\nQuery: ")
            if query.lower() == 'exit':
                break

            results = analyzer.search(query)
            print("\nResults:")
            print("=" * 50)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Relevancy Score: {result['relevancy_score']:.2f}")
                print("-" * 30)
                print(result['text'])
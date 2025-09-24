import os
import sys
import json
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
            # Load the summary data
            with open(latest_summary, 'r') as f:
                summary_data = json.load(f)
            
            # Generate HTML report name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"evaluation_report_{timestamp}.html"
            report_path = os.path.join('Logs', report_name)
            
            # Read template and update with results
            with open('evaluation_report_template.html', 'r') as template_file:
                template = template_file.read()
                
            # Convert summary data to JavaScript
            summary_js = f"""
                <script>
                    // Evaluation data
                    const evaluationData = {json.dumps(summary_data, indent=2)};
                    
                    document.addEventListener('DOMContentLoaded', function() {{
                        // Update timestamp
                        document.getElementById('timestamp').textContent = evaluationData.readable_timestamp;

                        // Update summary metrics
                        document.getElementById('total-configs').textContent = evaluationData.results.length;
                        document.getElementById('total-queries').textContent = evaluationData.query_stats.total_queries;
                        
                        // Update complexity breakdown
                        const complexityHtml = Object.entries(evaluationData.query_stats.by_complexity)
                            .map(([complexity, count]) => `
                                <div class="flex justify-between">
                                    <span class="text-gray-600">${{complexity}}:</span>
                                    <span class="font-medium">${{count}}</span>
                                </div>
                            `).join('');
                        document.getElementById('complexity-breakdown').innerHTML = complexityHtml;

                        // Update best configurations
                        const bestConfigsHtml = Object.entries(evaluationData.best_configurations)
                            .map(([category, config]) => `
                                <div class="border rounded p-4">
                                    <h4 class="font-semibold mb-2">${{category.charAt(0).toUpperCase() + category.slice(1)}}</h4>
                                    <div class="text-sm">
                                        <p>Model: ${{config.embedding_model}}</p>
                                        <p>Strategy: ${{config.chunking_strategy}}</p>
                                        <p>Time: ${{config.performance.avg_query_time.toFixed(3)}}s</p>
                                        <p>Score: ${{config.performance.avg_relevancy_score.toFixed(3)}}</p>
                                    </div>
                                </div>
                            `).join('');
                        document.getElementById('best-configs').innerHTML = bestConfigsHtml;

                        // Create model comparison chart
                        const modelData = evaluationData.model_comparisons;
                        new Chart(document.getElementById('modelChart'), {{
                            type: 'bar',
                            data: {{
                                labels: Object.keys(modelData),
                                datasets: [
                                    {{
                                        label: 'Avg Query Time (s)',
                                        data: Object.values(modelData).map(d => d.avg_query_time),
                                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                        yAxisID: 'y-time'
                                    }},
                                    {{
                                        label: 'Avg Relevancy Score',
                                        data: Object.values(modelData).map(d => d.avg_relevancy_score),
                                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                        yAxisID: 'y-score'
                                    }}
                                ]
                            }},
                            options: {{
                                responsive: true,
                                scales: {{
                                    'y-time': {{
                                        type: 'linear',
                                        position: 'left',
                                        title: {{
                                            display: true,
                                            text: 'Time (seconds)'
                                        }}
                                    }},
                                    'y-score': {{
                                        type: 'linear',
                                        position: 'right',
                                        title: {{
                                            display: true,
                                            text: 'Relevancy Score'
                                        }}
                                    }}
                                }}
                            }}
                        }});

                        // Create strategy comparison chart
                        const strategyData = evaluationData.strategy_comparisons;
                        new Chart(document.getElementById('strategyChart'), {{
                            type: 'bar',
                            data: {{
                                labels: Object.keys(strategyData),
                                datasets: [
                                    {{
                                        label: 'Avg Query Time (s)',
                                        data: Object.values(strategyData).map(d => d.avg_query_time),
                                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                        yAxisID: 'y-time'
                                    }},
                                    {{
                                        label: 'Avg Relevancy Score',
                                        data: Object.values(strategyData).map(d => d.avg_relevancy_score),
                                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                        yAxisID: 'y-score'
                                    }}
                                ]
                            }},
                            options: {{
                                responsive: true,
                                scales: {{
                                    'y-time': {{
                                        type: 'linear',
                                        position: 'left',
                                        title: {{
                                            display: true,
                                            text: 'Time (seconds)'
                                        }}
                                    }},
                                    'y-score': {{
                                        type: 'linear',
                                        position: 'right',
                                        title: {{
                                            display: true,
                                            text: 'Relevancy Score'
                                        }}
                                    }}
                                }}
                            }}
                        }});

                        // Populate results table
                        const resultsHtml = evaluationData.results.map(result => `
                            <tr>
                                <td class="px-6 py-4 whitespace-nowrap">${{result.embedding_model}}</td>
                                <td class="px-6 py-4 whitespace-nowrap">${{result.chunking_strategy}}</td>
                                <td class="px-6 py-4 whitespace-nowrap">${{result.min_relevancy_score}}</td>
                                <td class="px-6 py-4 whitespace-nowrap">${{result.chunk_size}}</td>
                                <td class="px-6 py-4 whitespace-nowrap">${{result.avg_query_time.toFixed(3)}}</td>
                                <td class="px-6 py-4 whitespace-nowrap">${{result.avg_relevancy_score.toFixed(3)}}</td>
                            </tr>
                        `).join('');
                        document.getElementById('results-body').innerHTML = resultsHtml;
                    }});
                </script>
            """
            
            # Find the closing </body> tag and insert our data before it
            template_parts = template.split('</body>')
            populated_template = template_parts[0] + summary_js + '\n</body>' + template_parts[1]
                
            # Save the populated template
            with open(report_path, 'w') as report_file:
                report_file.write(populated_template)
            
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
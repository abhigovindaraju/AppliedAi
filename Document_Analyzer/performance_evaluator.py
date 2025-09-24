import time
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class PerformanceEvaluator:
    """Evaluate different model and chunking combinations"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        # Check if Gemini API key is available
        self.api_key_available = os.getenv("GOOGLE_API_KEY") is not None
        self.embedding_models = ["word2vec", "bert"]
        if self.api_key_available:
            self.embedding_models.append("gemini")
            
        # Configuration state tracking
        self.current_config = {
            "model": None,
            "strategy": None,
            "min_score": None,
            "chunk_size": None
        }
            
        self.example_queries = [
            {
                "query": "What was the total revenue in the most recent quarter?",
                "category": "Financial Metrics",
                "complexity": "Simple"
            },
            {
                "query": "Compare the operating expenses between the current and previous quarter.",
                "category": "Comparative Analysis",
                "complexity": "Moderate"
            },
            {
                "query": "What are the main risk factors mentioned in the report?",
                "category": "Risk Analysis",
                "complexity": "Complex"
            },
            {
                "query": "Show the breakdown of revenue by geographic segments.",
                "category": "Segmentation",
                "complexity": "Moderate"
            },
            {
                "query": "What is the trend in gross margin over the last four quarters?",
                "category": "Trend Analysis",
                "complexity": "Complex"
            },
            {
                "query": "List all mentions of R&D investments and their impact.",
                "category": "Investment Analysis",
                "complexity": "Complex"
            },
            {
                "query": "What is the current cash and cash equivalents position?",
                "category": "Balance Sheet",
                "complexity": "Simple"
            },
            {
                "query": "Explain the changes in the company's debt structure.",
                "category": "Financial Structure",
                "complexity": "Complex"
            },
            {
                "query": "What are the key performance indicators mentioned in the management discussion?",
                "category": "Management Analysis",
                "complexity": "Moderate"
            },
            {
                "query": "Summarize all forward-looking statements about future growth.",
                "category": "Future Outlook",
                "complexity": "Complex"
            }
        ]
    
    def _convert_to_serializable(self, obj):
        """Convert numpy and pandas objects to native Python types for JSON serialization"""
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        elif isinstance(obj, (float, np.floating)):
            return float(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        return obj
    
    def run_evaluation(
        self,
        embedding_models: Optional[List[str]] = None,
        chunking_strategies: List[str] = ["fixed", "semantic", "recursive", "adaptive"],
        min_scores: List[float] = [0.5, 0.7, 0.9],
        chunk_sizes: List[int] = [500, 1000, 2000]
    ) -> pd.DataFrame:
        """Run comprehensive evaluation with different configurations"""
        results = []
        successful_configs = 0
        
        # Use instance models if none provided
        if embedding_models is None:
            embedding_models = self.embedding_models
        
        # Calculate total configurations    
        total_configs = len(embedding_models) * len(chunking_strategies) * len(min_scores) * len(chunk_sizes)
        print(f"\nStarting evaluation with {total_configs} configurations")
        
        for model in embedding_models:
            print(f"\nTesting model: {model}")
            try:
                self.analyzer.embedding_model = self.analyzer._get_embedding_model(model)
            except Exception as e:
                print(f"✗ Error initializing model {model}: {e}")
                continue

            for strategy in chunking_strategies:
                for min_score in min_scores:
                    for chunk_size in chunk_sizes:
                        print(f"  Configuration: {strategy}, min_score={min_score}, chunk_size={chunk_size}")
                        try:
                            # Configure analyzer
                            self.analyzer.chunking_strategy = self.analyzer._get_chunking_strategy(strategy)
                            self.analyzer.min_relevancy_score = min_score
                            self.analyzer.chunk_size = chunk_size
                            
                            # Validation test
                            test_result = self.analyzer.search("Test query")
                            if not isinstance(test_result, list):
                                print(f"  ✗ Warning: Invalid result type")
                                continue
                            
                            # Run evaluation
                            config_results = self._evaluate_configuration()
                            config_results.update({
                                "embedding_model": model,
                                "chunking_strategy": strategy,
                                "min_relevancy_score": min_score,
                                "chunk_size": chunk_size
                            })
                            
                            results.append(config_results)
                            successful_configs += 1
                            print("  ✓ Configuration evaluated successfully")
                            
                        except Exception as e:
                            print(f"  ✗ Error evaluating configuration: {e}")
                            continue
        
        print(f"\nCompleted evaluation. Successful configs: {successful_configs}/{total_configs}")
        
        # Create empty DataFrame if no results
        if not results:
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        self._save_results(df)
        
        return df
    
    def _evaluate_configuration(self) -> Dict[str, Any]:
        """Evaluate current configuration with all example queries"""
        total_time = 0
        total_chunks = 0
        relevancy_scores = []
        
        query_results = []
        
        for query_info in self.example_queries:
            start_time = time.time()
            results = self.analyzer.search(query_info["query"])
            query_time = time.time() - start_time
            
            # Collect metrics
            total_time += query_time
            total_chunks += len(results)
            relevancy_scores.extend([r["relevancy_score"] for r in results])
            
            query_results.append({
                "query": query_info["query"],
                "category": query_info["category"],
                "complexity": query_info["complexity"],
                "time": query_time,
                "num_results": len(results),
                "avg_relevancy": sum([r["relevancy_score"] for r in results]) / max(1, len(results))
            })
        
        # Calculate aggregate metrics
        return {
            "avg_query_time": total_time / len(self.example_queries),
            "avg_chunks_per_query": total_chunks / len(self.example_queries),
            "avg_relevancy_score": sum(relevancy_scores) / max(1, len(relevancy_scores)),
            "query_results": query_results,
            "total_queries": len(self.example_queries)
        }
    
    def _save_results(self, df: pd.DataFrame):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        readable_timestamp = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")
        
        # Convert DataFrame to dict with native Python types
        df_dict = df.to_dict(orient='records')
        df_dict = [
            {k: self._convert_to_serializable(v) for k, v in record.items()}
            for record in df_dict
        ]
        
        # Organize queries by complexity
        queries_by_complexity = {
            "Simple": [],
            "Moderate": [],
            "Complex": []
        }
        
        for query in self.example_queries:
            queries_by_complexity[query["complexity"]].append(query)
        
        # Ensure Logs directory exists
        logs_dir = "Logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save detailed CSV in Logs directory
        csv_filename = f"evaluation_results_{timestamp}.csv"
        csv_path = os.path.join(logs_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved detailed results to: {csv_path}")
        
        # Save summary report
        summary = {
            "timestamp": timestamp,
            "readable_timestamp": readable_timestamp,
            "results": df_dict,
            "example_queries": self.example_queries,
            "queries_by_complexity": queries_by_complexity,
            "query_stats": {
                "total_queries": len(self.example_queries),
                "by_complexity": {
                    complexity: len(queries)
                    for complexity, queries in queries_by_complexity.items()
                }
            },
            "best_configurations": {
                "fastest": self._get_best_config(df, "avg_query_time", True),
                "most_relevant": self._get_best_config(df, "avg_relevancy_score", False),
                "balanced": self._get_best_config(df, "balanced_score", False)
            },
            "model_comparisons": self._get_model_comparisons(df),
            "strategy_comparisons": self._get_strategy_comparisons(df)
        }
        
        # Convert all values to native Python types
        summary = {k: self._convert_to_serializable(v) for k, v in summary.items()}
        
        # Ensure Logs directory exists
        logs_dir = "Logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save JSON summary in Logs directory
        summary_filename = f"evaluation_summary_{timestamp}.json"
        summary_path = os.path.join(logs_dir, summary_filename)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        # Create symlink to latest summary
        latest_link = os.path.join(logs_dir, "evaluation_summary_latest.json")
        try:
            if os.path.exists(latest_link) or os.path.islink(latest_link):
                os.unlink(latest_link)
            os.symlink(summary_filename, latest_link)
        except Exception as e:
            print(f"Warning: Could not create symlink: {e}")
        
        print(f"Saved summary report to: {summary_path}")
        print(f"Latest results link: {latest_link}")
    
    def _get_best_config(self, df: pd.DataFrame, metric: str, minimize: bool = True) -> Dict[str, Any]:
        """Get the best configuration for a given metric"""
        if metric == "balanced_score":
            df = df.copy()
            df["balanced_score"] = (df["avg_relevancy_score"] / df["avg_query_time"])
        
        # Convert to list to handle empty DataFrames
        values = df[metric].tolist()
        if not values:
            return {}
            
        idx = values.index(min(values) if minimize else max(values))
        best_row = df.iloc[int(idx)]
        
        return {
            "embedding_model": best_row["embedding_model"],
            "chunking_strategy": best_row["chunking_strategy"],
            "min_relevancy_score": best_row["min_relevancy_score"],
            "chunk_size": best_row["chunk_size"],
            "performance": {
                "avg_query_time": best_row["avg_query_time"],
                "avg_relevancy_score": best_row["avg_relevancy_score"]
            }
        }
    
    def _get_model_comparisons(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compare performance across embedding models"""
        return {
            model: {
                "avg_query_time": df[df["embedding_model"] == model]["avg_query_time"].mean(),
                "avg_relevancy_score": df[df["embedding_model"] == model]["avg_relevancy_score"].mean()
            }
            for model in df["embedding_model"].unique()
        }
    
    def _get_strategy_comparisons(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compare performance across chunking strategies"""
        return {
            strategy: {
                "avg_query_time": df[df["chunking_strategy"] == strategy]["avg_query_time"].mean(),
                "avg_relevancy_score": df[df["chunking_strategy"] == strategy]["avg_relevancy_score"].mean()
            }
            for strategy in df["chunking_strategy"].unique()
        }

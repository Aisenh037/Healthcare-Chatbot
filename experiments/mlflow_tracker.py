"""
MLflow Experiment Tracker

Professional ML experiment tracking and model management
using MLflow for the medical RAG chatbot.
"""

import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import json


class MLflowExperimentTracker:
    """
    Manage ML experiments with MLflow.
    
    Features:
    - Log experiment parameters and metrics
    - Track model artifacts
    - Compare experiment runs
    - Model registry for production models
    """
    
    def __init__(self, experiment_name: str = "medical-rag-chatbot",
                 tracking_uri: Optional[str] = None):
        """
        Initialize MLflow experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (None = local)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local SQLite database
            db_path = Path(__file__).parent.parent.parent / "mlruns.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        print(f"✅ MLflow experiment initialized: '{experiment_name}'")
        print(f"📊 Tracking URI: {mlflow.get_tracking_uri()}")
    
    def log_experiment(self, 
                      run_name: str,
                      params: Dict[str, Any],
                      metrics: Dict[str, float],
                      tags: Optional[Dict[str, str]] = None,
                      artifacts: Optional[Dict[str, str]] = None) -> str:
        """
        Log a complete experiment run.
        
        Args:
            run_name: Name for this run
            params: Hyperparameters (e.g., model_name, chunk_size)
            metrics: Performance metrics (e.g., precision, recall)
            tags: Optional tags for categorization
            artifacts: Optional file paths to log as artifacts
            
        Returns:
            Run ID
        """
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Log artifacts
            if artifacts:
                for name, file_path in artifacts.items():
                    if Path(file_path).exists():
                        mlflow.log_artifact(file_path, artifact_path=name)
            
            run_id = run.info.run_id
            
            print(f"✅ Logged experiment run: '{run_name}' (ID: {run_id})")
            print(f"   Parameters: {len(params)}")
            print(f"   Metrics: {len(metrics)}")
            
            return run_id
    
    def log_embedding_experiment(self,
                                embedding_model: str,
                                retrieval_metrics: Dict[str, float],
                                generation_metrics: Dict[str, float],
                                config: Dict[str, Any]) -> str:
        """
        Log an embedding model experiment.
        
        Args:
            embedding_model: Name of embedding model
            retrieval_metrics: Retrieval performance metrics
            generation_metrics: Generation quality metrics
            config: Model configuration
            
        Returns:
            Run ID
        """
        # Prepare parameters
        params = {
            "embedding_model": embedding_model,
            "chunk_size": config.get("chunk_size", 1000),
            "chunk_overlap": config.get("chunk_overlap", 100),
            "top_k": config.get("top_k", 10),
            "temperature": config.get("temperature", 0.3),
            "llm_model": config.get("model_name", "llama-3.1-8b-instant")
        }
        
        # Combine all metrics
        all_metrics = {**retrieval_metrics, **generation_metrics}
        
        # Calculate average score
        avg_score = sum(all_metrics.values()) / len(all_metrics)
        all_metrics["avg_score"] = avg_score
        
        # Add tags
        tags = {
            "experiment_type": "embedding_comparison",
            "model_family": self._get_model_family(embedding_model),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate run name
        run_name = f"{embedding_model.split('/')[-1]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return self.log_experiment(
            run_name=run_name,
            params=params,
            metrics=all_metrics,
            tags=tags
        )
    
    def log_retrieval_strategy_experiment(self,
                                         strategy_name: str,
                                         metrics: Dict[str, float],
                                         config: Dict[str, Any]) -> str:
        """
        Log a retrieval strategy experiment.
        
        Args:
            strategy_name: Name of strategy (dense, hybrid, rerank)
            metrics: Performance metrics
            config: Strategy configuration
            
        Returns:
            Run ID
        """
        params = {
            "strategy": strategy_name,
            **config
        }
        
        tags = {
            "experiment_type": "retrieval_strategy",
            "strategy_name": strategy_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        run_name = f"{strategy_name}_retrieval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return self.log_experiment(
            run_name=run_name,
            params=params,
            metrics=metrics,
            tags=tags
        )
    
    def compare_runs(self, run_ids: Optional[List[str]] = None, 
                    metric_key: str = "avg_score") -> List[Dict[str, Any]]:
        """
        Compare multiple experiment runs.
        
        Args:
            run_ids: List of run IDs to compare (None = all runs)
            metric_key: Metric to rank by
            
        Returns:
            List of run summaries sorted by metric
        """
        if run_ids:
            runs = [mlflow.get_run(run_id) for run_id in run_ids]
        else:
            # Get all runs from current experiment
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=[f"metrics.{metric_key} DESC"]
            )
            runs = [mlflow.get_run(row.run_id) for _, row in runs.iterrows()]
        
        comparison = []
        for run in runs:
            summary = {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
                "params": run.data.params,
                "metrics": run.data.metrics,
                "timestamp": datetime.fromtimestamp(run.info.start_time / 1000).isoformat()
            }
            comparison.append(summary)
        
        # Sort by metric
        if metric_key in comparison[0]["metrics"]:
            comparison.sort(key=lambda x: x["metrics"].get(metric_key, 0), reverse=True)
        
        return comparison
    
    def get_best_run(self, metric_key: str = "avg_score") -> Dict[str, Any]:
        """
        Get the best performing run based on a metric.
        
        Args:
            metric_key: Metric to optimize
            
        Returns:
            Best run summary
        """
        runs = self.compare_runs(metric_key=metric_key)
        
        if runs:
            best_run = runs[0]
            print(f"\n🏆 Best Run: {best_run['run_name']}")
            print(f"   {metric_key}: {best_run['metrics'].get(metric_key, 'N/A')}")
            return best_run
        
        return {}
    
    def generate_comparison_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comparison report of all experiments.
        
        Args:
            output_path: Path to save report (None = print only)
            
        Returns:
            Report as markdown string
        """
        runs = self.compare_runs()
        
        report = f"# Experiment Comparison Report\n\n"
        report += f"**Experiment**: {self.experiment_name}\n"
        report += f"**Generated**: {datetime.utcnow().isoformat()}\n"
        report += f"**Total Runs**: {len(runs)}\n\n"
        
        report += "## Top 5 Runs by Average Score\n\n"
        report += "| Rank | Run Name | Avg Score | Model | Top-K | Precision@5 | Recall@5 |\n"
        report += "|------|----------|-----------|-------|-------|-------------|----------|\n"
        
        for i, run in enumerate(runs[:5], 1):
            metrics = run["metrics"]
            params = run["params"]
            report += f"| {i} | {run['run_name'][:30]} | {metrics.get('avg_score', 0):.3f} | "
            report += f"{params.get('embedding_model', 'N/A')[:20]} | "
            report += f"{params.get('top_k', 'N/A')} | "
            report += f"{metrics.get('precision@5', 0):.3f} | "
            report += f"{metrics.get('recall@5', 0):.3f} |\n"
        
        if output_path:
            Path(output_path).write_text(report)
            print(f"✅ Report saved to: {output_path}")
        
        return report
    
    @staticmethod
    def _get_model_family(model_name: str) -> str:
        """Determine model family from name"""
        model_name_lower = model_name.lower()
        
        if "biobert" in model_name_lower:
            return "medical"
        elif "scibert" in model_name_lower:
            return "scientific"
        elif "pubmed" in model_name_lower:
            return "medical"
        elif "minilm" in model_name_lower:
            return "general"
        elif "mpnet" in model_name_lower:
            return "general"
        elif "bge" in model_name_lower:
            return "general"
        else:
            return "other"


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = MLflowExperimentTracker(experiment_name="embedding-comparison-demo")
    
    # Example: Log an embedding experiment
    run_id = tracker.log_embedding_experiment(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        retrieval_metrics={
            "precision@3": 0.850,
            "precision@5": 0.800,
            "recall@3": 0.750,
            "recall@5": 0.850,
            "ndcg@5": 0.820,
            "mrr": 0.875
        },
        generation_metrics={
            "bleu-4": 0.650,
            "rouge-l": 0.700,
            "answer_relevance": 0.780
        },
        config={
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "top_k": 10,
            "temperature": 0.3,
            "model_name": "llama-3.1-8b-instant"
        }
    )
    
    print(f"\n📊 Run ID: {run_id}")
    
    # Generate report
    report = tracker.generate_comparison_report()
    print("\n" + report)
    
    print("\n💡 To view experiments in MLflow UI:")
    print("   mlflow ui --port 5000")
    print("   Open http://localhost:5000")

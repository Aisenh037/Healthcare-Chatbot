"""
Experiment Configuration Manager

This module provides utilities for tracking ML experiments
including model configurations, hyperparameters, and results.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ExperimentConfig:
    """Configuration for a single experiment run"""
    
    def __init__(
        self,
        experiment_name: str,
        model_name: str,
        embedding_model: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        top_k: int = 10,
        temperature: float = 0.3,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    def save(self, output_dir: str = "experiments"):
        """Save configuration to YAML file"""
        output_path = Path(output_dir) / f"{self.experiment_name}_config.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        return output_path
    
    @classmethod
    def load(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**{k: v for k, v in config_dict.items() if k != 'created_at'})


class ExperimentTracker:
    """Track experiment results and metrics"""
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.experiment_dir / "results.json"
        
        # Load existing results if available
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = []
    
    def log_experiment(
        self,
        config: ExperimentConfig,
        metrics: Dict[str, float],
        notes: str = ""
    ):
        """Log experiment configuration and results"""
        result = {
            "experiment_name": config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": config.to_dict(),
            "metrics": metrics,
            "notes": notes
        }
        
        self.results.append(result)
        self._save_results()
        
        # Also save individual experiment result
        result_path = self.experiment_dir / f"{config.experiment_name}_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result_path
    
    def _save_results(self):
        """Save all results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_experiments(self) -> list:
        """Get all logged experiments"""
        return self.results
    
    def get_best_experiment(self, metric: str = "avg_score", higher_is_better: bool = True) -> Dict:
        """Get best performing experiment based on a metric"""
        if not self.results:
            return None
        
        sorted_results = sorted(
            self.results,
            key=lambda x: x["metrics"].get(metric, 0),
            reverse=higher_is_better
        )
        
        return sorted_results[0]
    
    def compare_experiments(self, experiment_names: list) -> Dict[str, Dict]:
        """Compare multiple experiments"""
        comparison = {}
        
        for exp in self.results:
            if exp["experiment_name"] in experiment_names:
                comparison[exp["experiment_name"]] = {
                    "config": exp["config"],
                    "metrics": exp["metrics"]
                }
        
        return comparison
    
    def print_summary(self):
        """Print summary of all experiments"""
        print(f"\n{'='*60}")
        print(f"Experiment Summary ({len(self.results)} runs)")
        print(f"{'='*60}\n")
        
        for exp in self.results:
            print(f"📊 {exp['experiment_name']}")
            print(f"   Model: {exp['config']['embedding_model']}")
            print(f"   Metrics:")
            for metric, value in exp["metrics"].items():
                print(f"      - {metric}: {value:.4f}")
            print()

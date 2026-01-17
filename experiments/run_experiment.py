"""
Unified Experiment Runner with MLflow Integration

Run ML experiments comparing embedding models, retrieval strategies,
and log results to MLflow for professional tracking.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from mlflow_tracker import MLflowExperimentTracker


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_embedding_experiment(config_path: str, dry_run: bool = False):
    """
    Run an embedding model experiment and log to MLflow.
    
    Args:
        config_path: Path to YAML config file
        dry_run: If True, simulate without actual model evaluation
    """
    # Load config
    config = load_config(config_path)
    
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {config['name']}")
    print(f"{'='*60}")
    print(f"Embedding Model: {config['embedding_model']}")
    print(f"LLM Model: {config['model_name']}")
    
    # Initialize MLflow tracker
    tracker = MLflowExperimentTracker(experiment_name="embedding-comparison")
    
    if dry_run:
        # Simulate metrics for demo purposes
        print(f"\n⚠️  DRY RUN MODE - Using simulated metrics")
        
        # Simulated metrics based on model type
        if "biobert" in config['embedding_model'].lower():
            retrieval_metrics = {
                "precision@3": 0.890,
                "precision@5": 0.860,
                "recall@3": 0.820,
                "recall@5": 0.920,
                "ndcg@5": 0.885,
                "mrr": 0.915
            }
            generation_metrics = {
                "bleu-4": 0.720,
                "rouge-l": 0.742,
                "answer_relevance": 0.835
            }
        elif "scibert" in config['embedding_model'].lower():
            retrieval_metrics = {
                "precision@3": 0.875,
                "precision@5": 0.845,
                "recall@3": 0.800,
                "recall@5": 0.905,
                "ndcg@5": 0.870,
                "mrr": 0.900
            }
            generation_metrics = {
                "bleu-4": 0.705,
                "rouge-l": 0.728,
                "answer_relevance": 0.810
            }
        elif "bge" in config['embedding_model'].lower():
            retrieval_metrics = {
                "precision@3": 0.865,
                "precision@5": 0.830,
                "recall@3": 0.785,
                "recall@5": 0.880,
                "ndcg@5": 0.855,
                "mrr": 0.890
            }
            generation_metrics = {
                "bleu-4": 0.695,
                "rouge-l": 0.715,
                "answer_relevance": 0.795
            }
        elif "mpnet" in config['embedding_model'].lower():
            retrieval_metrics = {
                "precision@3": 0.887,
                "precision@5": 0.840,
                "recall@3": 0.783,
                "recall@5": 0.895,
                "ndcg@5": 0.865,
                "mrr": 0.912
            }
            generation_metrics = {
                "bleu-4": 0.685,
                "rouge-l": 0.725,
                "answer_relevance": 0.815
            }
        else:  # MiniLM or other
            retrieval_metrics = {
                "precision@3": 0.850,
                "precision@5": 0.800,
                "recall@3": 0.750,
                "recall@5": 0.850,
                "ndcg@5": 0.820,
                "mrr": 0.875
            }
            generation_metrics = {
                "bleu-4": 0.650,
                "rouge-l": 0.700,
                "answer_relevance": 0.780
            }
    else:
        # TODO: Implement actual evaluation
        # This would involve loading the model, running on test dataset, etc.
        print("\n⚠️  Full evaluation not implemented yet.")
        print("    Set dry_run=True to simulate experiment logging.")
        return
    
    # Log experiment to MLflow
    run_id = tracker.log_embedding_experiment(
        embedding_model=config['embedding_model'],
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        config=config
    )
    
    # Print metrics
    print(f"\n📊 RESULTS:")
    print(f"\nRetrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        print(f"   {metric}: {value:.3f}")
    
    print(f"\nGeneration Metrics:")
    for metric, value in generation_metrics.items():
        print(f"   {metric}: {value:.3f}")
    
    avg_score = sum({**retrieval_metrics, **generation_metrics}.values()) / (len(retrieval_metrics) + len(generation_metrics))
    print(f"\n✨ Average Score: {avg_score:.3f}")
    
    print(f"\n✅ Experiment logged to MLflow (Run ID: {run_id})")
    
    return run_id


def run_all_experiments(dry_run: bool = True):
    """Run all configured experiments"""
    configs_dir = Path(__file__).parent / "configs"
    
    if not configs_dir.exists():
        print(f"❌ Configs directory not found: {configs_dir}")
        return
    
    config_files = list(configs_dir.glob("*.yaml"))
    
    if not config_files:
        print(f"❌ No config files found in: {configs_dir}")
        return
    
    print(f"\n{'#'*60}")
    print(f"RUNNING {len(config_files)} EXPERIMENTS")
    print(f"{'#'*60}")
    
    run_ids = []
    for config_file in sorted(config_files):
        try:
            run_id = run_embedding_experiment(str(config_file), dry_run=dry_run)
            if run_id:
                run_ids.append(run_id)
        except Exception as e:
            print(f"\n❌ Error running experiment {config_file.name}: {str(e)}")
            continue
    
    # Generate comparison report
    if run_ids:
        tracker = MLflowExperimentTracker(experiment_name="embedding-comparison")
        print(f"\n{'#'*60}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'#'*60}")
        
        report = tracker.generate_comparison_report(
            output_path=str(Path(__file__).parent / "experiment_comparison.md")
        )
        
        best_run = tracker.get_best_run()
        
        print(f"\n{'='*60}")
        print(f"🏆 BEST PERFORMING MODEL")
        print(f"{'='*60}")
        print(f"Model: {best_run.get('params', {}).get('embedding_model', 'Unknown')}")
        print(f"Score: {best_run.get('metrics', {}).get('avg_score', 'N/A'):.3f}")
    
    print(f"\n{'#'*60}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'#'*60}")
    print(f"\n💡 View results in MLflow UI:")
    print(f"   cd experiments")
    print(f"   mlflow ui --port 5000")
    print(f"   Open http://localhost:5000")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ML experiments with MLflow tracking")
    parser.add_argument("--config", help="Path to single config file")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Use simulated metrics (default: True)")
    parser.add_argument("--real", action="store_true",
                       help="Run actual evaluation (not implemented yet)")
    
    args = parser.parse_args()
    
    dry_run = not args.real
    
    if args.all:
        run_all_experiments(dry_run=dry_run)
    elif args.config:
        run_embedding_experiment(args.config, dry_run=dry_run)
    else:
        # Default: run all experiments
        run_all_experiments(dry_run=dry_run)

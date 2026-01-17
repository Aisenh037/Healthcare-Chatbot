"""
Demo script for Data Engineering Pipeline

This script demonstrates the ETL pipeline capabilities without
requiring full Pinecone configuration.
"""

import asyncio
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

from data_engineering.pipeline import ETLPipeline


async def demo_pipeline():
    """Run demo of the data engineering pipeline"""
    
    print("\n" + "="*70)
    print(" "*20 + "DATA ENGINEERING PIPELINE DEMO")
    print("="*70)
    
    # Configure pipeline
    config = {
        "min_quality_score": 70.0,
        "source_file": "medical_qa_dataset.csv"
    }
    
    pipeline = ETLPipeline(config=config)
    
    # Path to sample dataset
    data_path = Path(__file__).parent.parent / "data" / "medical_qa_dataset.csv"
    
    if not data_path.exists():
        print(f"\n❌ Sample dataset not found at: {data_path}")
        print("ℹ️   Please ensure the data/ directory contains medical_qa_dataset.csv")
        return
    
    print(f"\n📁 Data source: {data_path}")
    
    # Run pipeline (without loading to vector DB for demo)
    try:
        stats = await pipeline.run_pipeline(
            source=str(data_path),
            text_columns=["question", "answer"],
            required_columns=["question", "answer"],
            load_to_vector_db=False  # Disable for demo
        )
        
        print("\n" + "="*70)
        print("PIPELINE RESULTS SUMMARY")
        print("="*70)
        print(f"✅ Success: {stats['success']}")
        print(f"📊 Extracted: {stats['extracted_rows']} rows")
        print(f"🔄 Transformed: {stats['transformed_rows']} rows")
        print(f"⏱️  Duration: {stats['duration_seconds']:.2f} seconds")
        
        if stats['errors']:
            print(f"⚠️   Errors: {len(stats['errors'])}")
        else:
            print("✨ No errors!")
        
        print("\n" + "="*70)
        print("💡 Next Steps:")
        print("="*70)
        print("1. To load data to Pinecone, set load_to_vector_db=True")
        print("2. Ensure PINECONE_API_KEY is set in .env file")
        print("3. Run: python -m server.data_engineering.pipeline --source <path> --load")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(demo_pipeline())

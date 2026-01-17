"""
ETL Pipeline Module

End-to-end pipeline for Extract, Transform, Load operations
on medical datasets to the vector database.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from tqdm import tqdm

from .ingestion import DataIngestion
from .validation import DataValidator


class ETLPipeline:
    """
    Complete ETL pipeline for medical data processing.
    
    Workflow:
    1. Extract: Load data from source files
    2. Transform: Clean, validate, enrich data
    3. Load: Generate embeddings and store in Pinecone
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ETL pipeline.
        
        Args:
            config: Pipeline configuration (chunk_size, overlap, etc.)
        """
        self.config = config or {}
        self.ingestion = DataIngestion()
        self.validator = DataValidator()
        
        # Pipeline statistics
        self.stats = {
            "extracted_rows": 0,
            "transformed_rows": 0,
            "loaded_rows": 0,
            "errors": []
        }
    
    def extract(self, source: str) -> pd.DataFrame:
        """
        Extract data from source file.
        
        Args:
            source: Path to data file (CSV or JSON)
            
        Returns:
            Raw DataFrame
        """
        print(f"\n{'='*60}")
        print("STEP 1: EXTRACT")
        print(f"{'='*60}")
        
        df = self.ingestion.load_dataset(source)
        self.stats["extracted_rows"] = len(df)
        
        print(f"\n✅ Extracted {len(df)} rows from {source}")
        return df
    
    def transform(self, df: pd.DataFrame, 
                 text_columns: Optional[List[str]] = None,
                 required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform and clean data.
        
        Args:
            df: Raw DataFrame
            text_columns: Columns to clean
            required_columns: Columns that must be present
            
        Returns:
            Cleaned DataFrame
        """
        print(f"\n{'='*60}")
        print("STEP 2: TRANSFORM")
        print(f"{'='*60}")
        
        df_clean = df.copy()
        
        # 1. Schema validation
        if required_columns:
            if not self.ingestion.validate_schema(df_clean, required_columns):
                raise ValueError(f"Schema validation failed. Required columns: {required_columns}")
        
        # 2. Remove missing values in critical columns
        if required_columns:
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=required_columns)
            dropped_count = initial_count - len(df_clean)
            
            if dropped_count > 0:
                print(f"🗑️  Dropped {dropped_count} rows with missing values in required columns")
        
        # 3. Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=required_columns if required_columns else None)
        dup_count = initial_count - len(df_clean)
        
        if dup_count > 0:
            print(f"🗑️  Removed {dup_count} duplicate rows")
        
        # 4. Clean text columns
        if text_columns:
            df_clean = self.ingestion.clean_dataframe(df_clean, text_columns)
        
        # 5. Add metadata
        df_clean['ingested_at'] = datetime.utcnow().isoformat()
        df_clean['source_file'] = self.config.get('source_file', 'unknown')
        
        # 6. Generate quality report
        quality_report = self.validator.generate_quality_report(
            df_clean,
            text_columns=text_columns
        )
        
        # 7. Check quality threshold
        min_quality_score = self.config.get('min_quality_score', 70)
        if quality_report['quality_score'] < min_quality_score:
            print(f"\n⚠️  WARNING: Quality score ({quality_report['quality_score']}) below threshold ({min_quality_score})")
            
            # Get recommendations
            recommendations = self.validator.get_recommendations(quality_report)
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
        
        self.stats["transformed_rows"] = len(df_clean)
        
        print(f"\n✅ Transformed {len(df_clean)} rows (quality score: {quality_report['quality_score']}/100)")
        return df_clean
    
    async def load_to_vectorstore(self, df: pd.DataFrame, 
                                  text_column: str = "question",
                                  metadata_columns: Optional[List[str]] = None) -> int:
        """
        Load data to vector database (Pinecone).
        
        Args:
            df: Cleaned DataFrame
            text_column: Column containing text to embed
            metadata_columns: Additional columns to store as metadata
            
        Returns:
            Number of vectors loaded
        """
        print(f"\n{'='*60}")
        print("STEP 3: LOAD TO VECTOR DATABASE")
        print(f"{'='*60}")
        
        try:
            # Import vector store utilities
            from ..docs.vectorstore import get_pinecone_index, embed_text
            
            index = await asyncio.to_thread(get_pinecone_index)
            metadata_columns = metadata_columns or []
            
            vectors_to_upsert = []
            
            print(f"\n🔄 Generating embeddings and preparing vectors...")
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
                try:
                    # Get text to embed
                    text = str(row[text_column])
                    
                    if not text or len(text.strip()) < 5:
                        continue
                    
                    # Generate embedding
                    embedding = await asyncio.to_thread(embed_text, text)
                    
                    # Prepare metadata
                    metadata = {
                        "text": text,
                        "source": "etl_pipeline",
                        "ingested_at": row.get('ingested_at', datetime.utcnow().isoformat())
                    }
                    
                    # Add custom metadata
                    for col in metadata_columns:
                        if col in row:
                            metadata[col] = str(row[col])
                    
                    # Prepare vector
                    vector_id = f"pipeline_{idx}_{datetime.utcnow().timestamp()}"
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
                
                except Exception as e:
                    self.stats["errors"].append({
                        "row": idx,
                        "error": str(e)
                    })
                    continue
            
            # Batch upsert to Pinecone
            if vectors_to_upsert:
                batch_size = 100
                total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
                
                print(f"\n📤 Uploading {len(vectors_to_upsert)} vectors in {total_batches} batches...")
                
                for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Uploading batches"):
                    batch = vectors_to_upsert[i:i + batch_size]
                    await asyncio.to_thread(index.upsert, vectors=batch)
                
                self.stats["loaded_rows"] = len(vectors_to_upsert)
                print(f"\n✅ Successfully loaded {len(vectors_to_upsert)} vectors to Pinecone")
            else:
                print("\n⚠️  No valid vectors to upload")
        
        except ImportError:
            print("\n⚠️  Vector store integration not available. Skipping load step.")
            print("ℹ️   To enable loading, ensure Pinecone is configured and vectorstore.py is accessible.")
            self.stats["loaded_rows"] = 0
        
        except Exception as e:
            print(f"\n❌ Error during load: {str(e)}")
            raise
        
        return self.stats["loaded_rows"]
    
    async def run_pipeline(self, source: str, 
                          text_columns: Optional[List[str]] = None,
                          required_columns: Optional[List[str]] = None,
                          load_to_vector_db: bool = False) -> Dict[str, Any]:
        """
        Run complete ETL pipeline.
        
        Args:
            source: Path to source data file
            text_columns: Columns containing text to clean
            required_columns: Columns that must be present
            load_to_vector_db: Whether to load to Pinecone (requires configuration)
            
        Returns:
            Pipeline statistics
        """
        start_time = datetime.utcnow()
        
        print(f"\n{'#'*60}")
        print("ETL PIPELINE STARTED")
        print(f"{'#'*60}")
        print(f"Source: {source}")
        print(f"Time: {start_time.isoformat()}")
        
        # Store source in config
        self.config['source_file'] = Path(source).name
        
        try:
            # Step 1: Extract
            df_raw = self.extract(source)
            
            # Step 2: Transform
            df_clean = self.transform(
                df_raw,
                text_columns=text_columns,
                required_columns=required_columns
            )
            
            # Step 3: Load (optional)
            if load_to_vector_db:
                loaded_count = await self.load_to_vectorstore(
                    df_clean,
                    text_column=text_columns[0] if text_columns else "question",
                    metadata_columns=required_columns
                )
            else:
                print(f"\n{'='*60}")
                print("STEP 3: LOAD (SKIPPED)")
                print(f"{'='*60}")
                print("ℹ️   Set load_to_vector_db=True to enable vector database loading")
                self.stats["loaded_rows"] = 0
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Final statistics
            self.stats["duration_seconds"] = duration
            self.stats["success"] = True
            
            print(f"\n{'#'*60}")
            print("ETL PIPELINE COMPLETED")
            print(f"{'#'*60}")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            print(f"📊 Statistics:")
            print(f"   - Extracted: {self.stats['extracted_rows']} rows")
            print(f"   - Transformed: {self.stats['transformed_rows']} rows")
            print(f"   - Loaded: {self.stats['loaded_rows']} vectors")
            print(f"   - Errors: {len(self.stats['errors'])}")
            
            if self.stats['errors']:
                print(f"\n⚠️  {len(self.stats['errors'])} errors occurred during processing")
        
        except Exception as e:
            self.stats["success"] = False
            self.stats["error"] = str(e)
            print(f"\n{'#'*60}")
            print("ETL PIPELINE FAILED")
            print(f"{'#'*60}")
            print(f"❌ Error: {str(e)}")
            raise
        
        return self.stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ETL pipeline on medical data")
    parser.add_argument("--source", required=True, help="Path to source data file")
    parser.add_argument("--text-columns", nargs="+", default=["question", "answer"],
                       help="Text columns to clean")
    parser.add_argument("--required-columns", nargs="+", default=["question", "answer"],
                       help="Required columns")
    parser.add_argument("--load", action="store_true", 
                       help="Load to vector database (requires Pinecone config)")
    parser.add_argument("--min-quality", type=float, default=70.0,
                       help="Minimum quality score threshold")
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = {
        "min_quality_score": args.min_quality,
        "source_file": Path(args.source).name
    }
    
    # Run pipeline
    pipeline = ETLPipeline(config=config)
    
    asyncio.run(pipeline.run_pipeline(
        source=args.source,
        text_columns=args.text_columns,
        required_columns=args.required_columns,
        load_to_vector_db=args.load
    ))

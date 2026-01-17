"""
Data Ingestion Module

Loads medical datasets from various formats (CSV, JSON, TXT)
with built-in validation and cleaning.
"""

import re
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from tqdm import tqdm


class DataIngestion:
    """Handle data loading and basic preprocessing"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.txt']
    
    def load_csv_dataset(self, path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load medical Q&A dataset from CSV file.
        
        Args:
            path: Path to CSV file
            encoding: File encoding (default: utf-8)
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if file_path.suffix != '.csv':
            raise ValueError(f"Expected CSV file, got {file_path.suffix}")
        
        print(f"📂 Loading CSV dataset from {path}...")
        df = pd.read_csv(path, encoding=encoding)
        print(f"✅ Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        return df
    
    def load_json_dataset(self, path: str) -> pd.DataFrame:
        """
        Load medical Q&A dataset from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            DataFrame with loaded data
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        print(f"📂 Loading JSON dataset from {path}...")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # If JSON has a 'data' or 'records' key
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            elif 'records' in data:
                df = pd.DataFrame(data['records'])
            elif 'test_cases' in data:
                df = pd.DataFrame(data['test_cases'])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
        
        print(f"✅ Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df
    
    def load_dataset(self, path: str) -> pd.DataFrame:
        """
        Auto-detect format and load dataset.
        
        Args:
            path: Path to dataset file
            
        Returns:
            DataFrame with loaded data
        """
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return self.load_csv_dataset(path)
        elif suffix == '.json':
            return self.load_json_dataset(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: {self.supported_formats}")
    
    def validate_schema(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, False otherwise
        """
        missing_cols = set(required_columns) - set(df.columns)
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print(f"✅ Schema validation passed. All required columns present.")
        return True
    
    def clean_medical_text(self, text: str) -> str:
        """
        Clean medical text by removing unwanted characters.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters (keep medical punctuation)
        text = re.sub(r'[^\w\s.,?!;:()\-/]', '', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def clean_dataframe(self, df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
        """
        Clean all text columns in DataFrame.
        
        Args:
            df: DataFrame to clean
            text_columns: List of column names containing text
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        print(f"🧹 Cleaning text in columns: {text_columns}...")
        
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.clean_medical_text)
        
        print("✅ Text cleaning completed")
        return df_clean
    
    def sample_dataset(self, df: pd.DataFrame, n: int = 5, random_state: int = 42) -> pd.DataFrame:
        """
        Get random sample from dataset for inspection.
        
        Args:
            df: DataFrame to sample
            n: Number of samples
            random_state: Random seed
            
        Returns:
            Sample DataFrame
        """
        if len(df) < n:
            return df
        
        return df.sample(n=n, random_state=random_state)


# Example usage
if __name__ == "__main__":
    # Example: Load and clean a medical dataset
    ingestion = DataIngestion()
    
    # Load CSV
    try:
        df = ingestion.load_csv_dataset("data/medical_qa_dataset.csv")
        
        # Validate schema
        required_cols = ["question", "answer"]
        if ingestion.validate_schema(df, required_cols):
            # Clean text
            df_clean = ingestion.clean_dataframe(df, text_columns=["question", "answer"])
            
            # Show sample
            print("\n📊 Sample data:")
            print(ingestion.sample_dataset(df_clean, n=3))
    
    except FileNotFoundError:
        print("ℹ️  Sample dataset not found. This is expected if running for the first time.")

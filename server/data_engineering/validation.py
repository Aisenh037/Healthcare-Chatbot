"""
Data Validation Module

Performs quality checks on medical datasets including
missing value detection, duplicate detection, and statistics.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter


class DataValidator:
    """Validate data quality and generate reports"""
    
    def __init__(self):
        self.validation_results = {}
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Detect missing values in each column.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with column names and missing value counts
        """
        missing = df.isnull().sum().to_dict()
        
        # Filter only columns with missing values
        missing = {k: v for k, v in missing.items() if v > 0}
        
        if missing:
            print(f"⚠️  Missing values detected:")
            for col, count in missing.items():
                pct = (count / len(df)) * 100
                print(f"   - {col}: {count} ({pct:.1f}%)")
        else:
            print("✅ No missing values found")
        
        return missing
    
    def detect_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
        """
        Detect duplicate rows in DataFrame.
        
        Args:
            df: DataFrame to check
            subset: Columns to check for duplicates (None = all columns)
            
        Returns:
            Number of duplicate rows
        """
        duplicates = df.duplicated(subset=subset).sum()
        
        if duplicates > 0:
            pct = (duplicates / len(df)) * 100
            print(f"⚠️  {duplicates} duplicate rows found ({pct:.1f}%)")
        else:
            print("✅ No duplicates found")
        
        return duplicates
    
    def validate_text_length(self, df: pd.DataFrame, column: str, 
                            min_length: int = 10, max_length: int = 5000) -> Dict[str, Any]:
        """
        Validate text length in a column.
        
        Args:
            df: DataFrame to check
            column: Column name to validate
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            
        Returns:
            Statistics about text length
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        lengths = df[column].fillna("").astype(str).str.len()
        
        stats = {
            "column": column,
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "too_short": int((lengths < min_length).sum()),
            "too_long": int((lengths > max_length).sum())
        }
        
        print(f"\n📏 Text length stats for '{column}':")
        print(f"   - Min: {stats['min']} chars")
        print(f"   - Max: {stats['max']} chars")
        print(f"   - Mean: {stats['mean']:.1f} chars")
        print(f"   - Median: {stats['median']:.1f} chars")
        
        if stats['too_short'] > 0:
            print(f"   ⚠️  {stats['too_short']} rows below min length ({min_length})")
        
        if stats['too_long'] > 0:
            print(f"   ⚠️  {stats['too_long']} rows exceed max length ({max_length})")
        
        return stats
    
    def check_category_distribution(self, df: pd.DataFrame, column: str) -> Dict[str, int]:
        """
        Check distribution of categorical values.
        
        Args:
            df: DataFrame to check
            column: Categorical column name
            
        Returns:
            Dictionary with value counts
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        distribution = df[column].value_counts().to_dict()
        
        print(f"\n📊 Category distribution for '{column}':")
        for category, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(df)) * 100
            print(f"   - {category}: {count} ({pct:.1f}%)")
        
        return distribution
    
    def validate_medical_terminology(self, text: str, 
                                    medical_keywords: Optional[List[str]] = None) -> bool:
        """
        Check if text contains medical terminology.
        
        Args:
            text: Text to validate
            medical_keywords: List of medical terms to check for
            
        Returns:
            True if medical terms found, False otherwise
        """
        if medical_keywords is None:
            # Common medical keywords
            medical_keywords = [
                'symptom', 'diagnosis', 'treatment', 'disease', 'condition',
                'medication', 'patient', 'doctor', 'clinical', 'medical',
                'therapy', 'syndrome', 'disorder', 'infection', 'health'
            ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in medical_keywords)
    
    def generate_quality_report(self, df: pd.DataFrame, 
                               text_columns: Optional[List[str]] = None,
                               category_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            text_columns: Columns containing text
            category_columns: Columns containing categories
            
        Returns:
            Complete quality report
        """
        print("\n" + "="*60)
        print("📋 DATA QUALITY REPORT")
        print("="*60)
        
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_values": self.check_missing_values(df),
            "duplicates": self.detect_duplicates(df),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        # Text column analysis
        if text_columns:
            report["text_stats"] = {}
            for col in text_columns:
                if col in df.columns:
                    report["text_stats"][col] = self.validate_text_length(df, col)
        
        # Category distribution
        if category_columns:
            report["category_distribution"] = {}
            for col in category_columns:
                if col in df.columns:
                    report["category_distribution"][col] = self.check_category_distribution(df, col)
        
        # Overall quality score (0-100)
        quality_score = 100
        
        # Penalize for missing values
        if report["missing_values"]:
            total_missing = sum(report["missing_values"].values())
            missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
            quality_score -= min(missing_pct * 2, 30)  # Max 30 points penalty
        
        # Penalize for duplicates
        if report["duplicates"] > 0:
            dup_pct = (report["duplicates"] / len(df)) * 100
            quality_score -= min(dup_pct * 3, 20)  # Max 20 points penalty
        
        report["quality_score"] = max(0, round(quality_score, 1))
        
        print(f"\n{'='*60}")
        print(f"📊 OVERALL QUALITY SCORE: {report['quality_score']}/100")
        print(f"{'='*60}\n")
        
        return report
    
    def get_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on quality report.
        
        Args:
            quality_report: Report from generate_quality_report()
            
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        if quality_report["missing_values"]:
            recommendations.append("🔧 Remove or impute missing values")
        
        if quality_report["duplicates"] > 0:
            recommendations.append("🔧 Remove duplicate rows")
        
        if quality_report.get("text_stats"):
            for col, stats in quality_report["text_stats"].items():
                if stats["too_short"] > 0:
                    recommendations.append(f"🔧 Review short texts in '{col}' column")
        
        if quality_report["quality_score"] < 70:
            recommendations.append("⚠️  Data quality is below acceptable threshold (< 70)")
        
        if not recommendations:
            recommendations.append("✅ Data quality is good! No critical issues found.")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Create sample dataset
    sample_data = {
        "question": [
            "What are symptoms of diabetes?",
            "What are symptoms of diabetes?",  # Duplicate
            "How to treat hypertension?",
            None,  # Missing
            "TB"  # Too short
        ],
        "answer": [
            "Common symptoms include increased thirst and frequent urination.",
            "Common symptoms include increased thirst and frequent urination.",
            "Treatment includes medication and lifestyle changes.",
            "Anti-hypertensive drugs are prescribed.",
            "Tuberculosis is a bacterial infection."
        ],
        "category": [
            "endocrinology",
            "endocrinology",
            "cardiology",
            "cardiology",
            "pulmonology"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Validate
    validator = DataValidator()
    report = validator.generate_quality_report(
        df, 
        text_columns=["question", "answer"],
        category_columns=["category"]
    )
    
    # Get recommendations
    print("\n💡 RECOMMENDATIONS:")
    for rec in validator.get_recommendations(report):
        print(f"   {rec}")

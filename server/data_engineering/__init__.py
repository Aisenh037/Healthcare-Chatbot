"""
Data Engineering Module

Handles data ingestion, validation, transformation, and loading
for the medical RAG chatbot.
"""

from .ingestion import DataIngestion
from .validation import DataValidator
from .pipeline import ETLPipeline

__all__ = ["DataIngestion", "DataValidator", "ETLPipeline"]

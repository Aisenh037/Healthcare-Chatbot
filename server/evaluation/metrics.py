"""
RAG Evaluation Metrics Module

This module implements comprehensive evaluation metrics for 
Retrieval-Augmented Generation (RAG) systems.
"""
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import re


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality"""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Precision@K score (0-1)
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        num_relevant_retrieved = len(retrieved_at_k.intersection(relevant_set))
        return num_relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Recall@K score (0-1)
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        num_relevant_retrieved = len(retrieved_at_k.intersection(relevant_set))
        return num_relevant_retrieved / len(relevant_docs)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs_list: List[List[str]], relevant_docs_list: List[List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_docs_list: List of lists of retrieved document IDs
            relevant_docs_list: List of lists of relevant document IDs
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            relevant_set = set(relevant)
            
            for rank, doc in enumerate(retrieved, 1):
                if doc in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs with relevance scores
            k: Number of top documents to consider
            
        Returns:
            NDCG@K score
        """
        def dcg(relevances: List[int], k: int) -> float:
            relevances = relevances[:k]
            return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
        
        # Binary relevance: 1 if relevant, 0 otherwise
        relevant_set = set(relevant_docs)
        retrieved_relevances = [1 if doc in relevant_set else 0 for doc in retrieved_docs[:k]]
        
        # Ideal ranking (all relevant docs first)
        ideal_relevances = sorted(retrieved_relevances, reverse=True)
        
        dcg_score = dcg(retrieved_relevances, k)
        idcg_score = dcg(ideal_relevances, k)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0


class GenerationMetrics:
    """Metrics for evaluating generated text quality"""
    
    @staticmethod
    def bleu_score(reference: str, hypothesis: str, n: int = 4) -> float:
        """
        Calculate BLEU score (simplified implementation)
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            n: Maximum n-gram size
            
        Returns:
            BLEU score (0-1)
        """
        def get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
            words = text.lower().split()
            return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Brevity penalty
        bp = 1.0 if len(hyp_words) > len(ref_words) else np.exp(1 - len(ref_words) / max(len(hyp_words), 1))
        
        # Calculate precision for each n-gram
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = get_ngrams(reference, i)
            hyp_ngrams = get_ngrams(hypothesis, i)
            
            if not hyp_ngrams:
                precisions.append(0.0)
                continue
            
            matches = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
            precisions.append(matches / len(hyp_ngrams))
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            score = bp * np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            score = 0.0
        
        return score
    
    @staticmethod
    def rouge_l(reference: str, hypothesis: str) -> float:
        """
        Calculate ROUGE-L score (Longest Common Subsequence)
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            
        Returns:
            ROUGE-L F1 score (0-1)
        """
        def lcs_length(x: List[str], y: List[str]) -> int:
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            
            return dp[m][n]
        
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        lcs_len = lcs_length(ref_words, hyp_words)
        
        if len(ref_words) == 0 or len(hyp_words) == 0:
            return 0.0
        
        precision = lcs_len / len(hyp_words)
        recall = lcs_len / len(ref_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def answer_relevance(answer: str, query: str) -> float:
        """
        Simple keyword-based answer relevance score
        
        Args:
            answer: Generated answer
            query: Original query
            
        Returns:
            Relevance score (0-1)
        """
        # Remove stopwords and punctuation
        stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for'}
        
        def clean_text(text: str) -> set:
            words = re.findall(r'\b\w+\b', text.lower())
            return set(w for w in words if w not in stopwords)
        
        query_words = clean_text(query)
        answer_words = clean_text(answer)
        
        if not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words.intersection(answer_words))
        union = len(query_words.union(answer_words))
        
        return intersection / union if union > 0 else 0.0


class RAGEvaluator:
    """Complete RAG system evaluator"""
    
    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k_values: List[int] = [3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        for k in k_values:
            results[f"precision@{k}"] = self.retrieval_metrics.precision_at_k(retrieved_docs, relevant_docs, k)
            results[f"recall@{k}"] = self.retrieval_metrics.recall_at_k(retrieved_docs, relevant_docs, k)
            results[f"ndcg@{k}"] = self.retrieval_metrics.ndcg_at_k(retrieved_docs, relevant_docs, k)
        
        return results
    
    def evaluate_generation(
        self,
        reference: str,
        hypothesis: str,
        query: str = None
    ) -> Dict[str, float]:
        """
        Evaluate generation quality
        
        Args:
            reference: Reference answer
            hypothesis: Generated answer
            query: Original query (optional)
            
        Returns:
            Dictionary of metric scores
        """
        results = {
            "bleu": self.generation_metrics.bleu_score(reference, hypothesis),
            "rouge_l": self.generation_metrics.rouge_l(reference, hypothesis)
        }
        
        if query:
            results["answer_relevance"] = self.generation_metrics.answer_relevance(hypothesis, query)
        
        return results
    
    def evaluate_end_to_end(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate complete RAG system on test cases
        
        Args:
            test_cases: List of test case dictionaries with keys:
                - query: User query
                - retrieved_docs: Retrieved document IDs
                - relevant_docs: Ground truth relevant document IDs
                - generated_answer: System-generated answer
                - reference_answer: Ground truth answer
                
        Returns:
            Aggregated evaluation metrics
        """
        retrieval_results = defaultdict(list)
        generation_results = defaultdict(list)
        
        for case in test_cases:
            # Evaluate retrieval
            ret_metrics = self.evaluate_retrieval(
                case["retrieved_docs"],
                case["relevant_docs"]
            )
            for metric, score in ret_metrics.items():
                retrieval_results[metric].append(score)
            
            # Evaluate generation
            gen_metrics = self.evaluate_generation(
                case["reference_answer"],
                case["generated_answer"],
                case["query"]
            )
            for metric, score in gen_metrics.items():
                generation_results[metric].append(score)
        
        # Calculate averages
        avg_retrieval = {k: np.mean(v) for k, v in retrieval_results.items()}
        avg_generation = {k: np.mean(v) for k, v in generation_results.items()}
        
        return {
            "retrieval_metrics": avg_retrieval,
            "generation_metrics": avg_generation,
            "num_test_cases": len(test_cases)
        }

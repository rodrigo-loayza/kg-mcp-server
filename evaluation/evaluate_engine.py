# evaluate_engine.py
from typing import List, Dict
import json
from pathlib import Path
import numpy as np
from hierarchical_engine import HierarchicalEngine, SearchResult


class IRMetrics:
    """Métricas de Information Retrieval"""

    @staticmethod
    def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
        """Calcula nDCG@k"""

        def dcg(scores):
            return sum((2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores[:k]))

        dcg_score = dcg(relevance_scores)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg_score = dcg(ideal_scores)

        return dcg_score / idcg_score if idcg_score > 0 else 0.0

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
        """Calcula Recall@k"""
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)

        if not relevant_set:
            return 0.0

        return len(retrieved_k & relevant_set) / len(relevant_set)

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
        """Calcula Precision@k"""
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)

        if not retrieved_k:
            return 0.0

        return len(retrieved_k & relevant_set) / len(retrieved_k)


class EngineEvaluator:
    """Evaluador del motor de indexación"""

    def __init__(self, engine: HierarchicalEngine):
        self.engine = engine
        self.metrics = IRMetrics()

    def evaluate(self, test_queries: List[Dict]) -> Dict:
        """
        Evalúa el motor con un conjunto de consultas de prueba

        test_queries formato:
        [
            {
                "query": "¿Cómo funciona QuickSort?",
                "relevant_chunks": ["chunk_id_1", "chunk_id_2"],
                "relevance_scores": [2, 2, 1, 0, ...]  # Relevancia gradual
            }
        ]
        """
        all_ndcg = []
        all_recall = []
        all_precision = []

        results_detail = []

        for test_case in test_queries:
            query = test_case["query"]
            relevant = test_case["relevant_chunks"]
            relevance_scores = test_case.get("relevance_scores", [])

            # Realizar búsqueda
            search_results = self.engine.search(query, k=10)
            retrieved_ids = [r.chunk_id for r in search_results]

            # Calcular métricas
            recall = self.metrics.recall_at_k(retrieved_ids, relevant, k=10)
            precision = self.metrics.precision_at_k(retrieved_ids, relevant, k=10)

            # nDCG requiere relevance scores
            if relevance_scores:
                # Mapear retrieved a sus scores de relevancia
                retrieved_scores = []
                relevant_dict = {
                    chunk_id: score for chunk_id, score in zip(relevant, relevance_scores)
                }

                for chunk_id in retrieved_ids[:10]:
                    retrieved_scores.append(relevant_dict.get(chunk_id, 0))

                ndcg = self.metrics.ndcg_at_k(retrieved_scores, k=10)
            else:
                # Si no hay scores, usar binario (2=relevante, 0=no relevante)
                retrieved_scores = [2 if rid in relevant else 0 for rid in retrieved_ids]
                ndcg = self.metrics.ndcg_at_k(retrieved_scores, k=10)

            all_ndcg.append(ndcg)
            all_recall.append(recall)
            all_precision.append(precision)

            results_detail.append(
                {
                    "query": query,
                    "ndcg@10": ndcg,
                    "recall@10": recall,
                    "precision@10": precision,
                    "retrieved_ids": retrieved_ids,
                }
            )

        # Calcular promedios
        evaluation_results = {
            "mean_ndcg@10": np.mean(all_ndcg),
            "mean_recall@10": np.mean(all_recall),
            "mean_precision@10": np.mean(all_precision),
            "num_queries": len(test_queries),
            "details": results_detail,
        }

        return evaluation_results


# Ejemplo de uso y baseline
class BaselineRAG:
    """Sistema RAG baseline para comparación"""

    def __init__(self, dimension: int = 384):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []
        self.embeddings = None

    def add_documents(self, processed_dir: Path):
        """Carga documentos procesados"""
        chunk_files = list(processed_dir.glob("*_chunks.json"))

        for chunk_file in chunk_files:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                self.chunks.extend(chunks)

            embedding_file = chunk_file.parent / chunk_file.name.replace(
                "_chunks.json", "_embeddings.npy"
            )
            embeddings = np.load(embedding_file)

            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Búsqueda simple por similitud coseno"""
        query_embedding = self.model.encode([query])[0]

        # Similitud coseno
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_k_idx = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k_idx:
            chunk = self.chunks[idx]
            result = SearchResult(
                doc_id=chunk["doc_id"],
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                score=float(similarities[idx]),
                metadata=chunk["metadata"],
            )
            results.append(result)

        return results


# Script de evaluación
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso: python evaluate_engine.py <processed_dir> <test_queries.json>")
        sys.exit(1)

    processed_dir = Path(sys.argv[1])
    test_queries_file = Path(sys.argv[2])

    # Cargar consultas de prueba
    with open(test_queries_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    print("=== Evaluando Motor Jerárquico ===\n")

    # Motor propuesto
    engine = HierarchicalEngine()
    engine.add_documents(processed_dir)

    evaluator = EngineEvaluator(engine)
    results_proposed = evaluator.evaluate(test_queries)

    print(f"Motor Propuesto:")
    print(f"  nDCG@10: {results_proposed['mean_ndcg@10']:.4f}")
    print(f"  Recall@10: {results_proposed['mean_recall@10']:.4f}")
    print(f"  Precision@10: {results_proposed['mean_precision@10']:.4f}\n")

    # Baseline RAG
    print("=== Evaluando Baseline RAG ===\n")
    baseline = BaselineRAG()
    baseline.add_documents(processed_dir)

    evaluator_baseline = EngineEvaluator(baseline)
    results_baseline = evaluator_baseline.evaluate(test_queries)

    print(f"Baseline RAG:")
    print(f"  nDCG@10: {results_baseline['mean_ndcg@10']:.4f}")
    print(f"  Recall@10: {results_baseline['mean_recall@10']:.4f}")
    print(f"  Precision@10: {results_baseline['mean_precision@10']:.4f}\n")

    # Calcular mejora
    recall_improvement = (
        (results_proposed["mean_recall@10"] - results_baseline["mean_recall@10"])
        / results_baseline["mean_recall@10"]
        * 100
    )

    print(f"=== Comparación ===")
    print(f"Mejora en Recall@10: {recall_improvement:+.2f}%")
    print(
        f"nDCG@10 cumple IOV (≥0.75): {'✅' if results_proposed['mean_ndcg@10'] >= 0.75 else '❌'}"
    )
    print(f"Mejora Recall@10 cumple IOV (≥25%): {'✅' if recall_improvement >= 25 else '❌'}")

    # Guardar resultados
    with open("evaluation_results.json", "w") as f:
        json.dump(
            {
                "proposed": results_proposed,
                "baseline": results_baseline,
                "improvement": {"recall_improvement_percent": recall_improvement},
            },
            f,
            indent=2,
        )

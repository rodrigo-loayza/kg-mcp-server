#!/usr/bin/env python3
"""
IOV2: Evaluación nDCG@10 y Recall@10 - Comparación RAG Baseline vs Hybrid MCP

Objetivo:
    Comparar el rendimiento del sistema Hybrid (RAG + Knowledge Graph) contra
    un baseline RAG puro usando métricas de recuperación de información.

Métricas evaluadas:
    - nDCG@10 ≥0.75 (Normalized Discounted Cumulative Gain)
    - Recall@10 mejora ≥25% vs baseline
    - Precision@10 (métrica adicional)

Queries:
    Usa queries_test_annotated.json con relevancia manual anotada

Salida:
    JSON comparativo con estadísticas detalladas para análisis y redacción con IA

Uso:
    python _iov2_evaluate_rag_vs_hybrid.py --queries evaluation/queries_test_annotated.json --output results/iov2_metrics.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass, asdict

# Agregar tesis_package al path
sys.path.insert(0, str(Path(__file__).parent / "tesis_package"))

from extract_context import ContextExtractor


@dataclass
class RetrievalResult:
    """Resultado de recuperación"""

    rank: int
    chunk_id: str
    document: str
    similarity: float
    text: str
    # Campos adicionales para hybrid
    hybrid_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0


class EvaluationMetrics:
    """Clase para calcular métricas de evaluación IR"""

    @staticmethod
    def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
        """
        Calcula nDCG@k (Normalized Discounted Cumulative Gain)

        Args:
            relevance_scores: Lista de scores de relevancia (mayor = más relevante)
                             Escala típica: 0=no relevante, 1=algo relevante,
                                          2=relevante, 3=muy relevante
            k: Top-k documentos a considerar

        Returns:
            nDCG@k score (0-1, donde 1 es perfecto)
        """
        if not relevance_scores:
            return 0.0

        # Truncar a k elementos
        relevance_scores = relevance_scores[:k]

        # DCG (Discounted Cumulative Gain)
        # DCG = rel_1 + sum(rel_i / log2(i+1)) for i=2..k
        dcg = relevance_scores[0]
        for i, score in enumerate(relevance_scores[1:], start=2):
            dcg += score / np.log2(i + 1)

        # IDCG (Ideal DCG) - ordenar por relevancia descendente
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = ideal_scores[0]
        for i, score in enumerate(ideal_scores[1:], start=2):
            idcg += score / np.log2(i + 1)

        # nDCG
        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Calcula Recall@k

        Recall = |retrieved ∩ relevant| / |relevant|

        Args:
            retrieved: IDs de chunks recuperados (en orden de ranking)
            relevant: IDs de chunks relevantes (ground truth)
            k: Top-k a considerar

        Returns:
            Recall@k (0-1)
        """
        if not relevant:
            return 0.0

        retrieved_set = set(str(r) for r in retrieved[:k])
        relevant_set = set(str(r) for r in relevant)

        true_positives = len(retrieved_set & relevant_set)

        return true_positives / len(relevant_set)

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Calcula Precision@k

        Precision = |retrieved ∩ relevant| / k

        Args:
            retrieved: IDs de chunks recuperados
            relevant: IDs de chunks relevantes
            k: Top-k a considerar

        Returns:
            Precision@k (0-1)
        """
        if not retrieved:
            return 0.0

        retrieved_set = set(str(r) for r in retrieved[:k])
        relevant_set = set(str(r) for r in relevant)

        true_positives = len(retrieved_set & relevant_set)

        return true_positives / min(k, len(retrieved))


class RAGBaselineSystem:
    """Sistema RAG baseline usando solo similitud vectorial"""

    def __init__(self):
        # Crear extractor (sin modo)
        self.extractor = ContextExtractor()

    def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Búsqueda usando solo RAG vectorial

        Args:
            query: Consulta del usuario
            k: Top-k resultados

        Returns:
            Lista de resultados ordenados por similitud coseno
        """
        # Extraer contexto usando RAG baseline
        results = self.extractor.extract_rag_baseline(query, k=k)

        # Convertir a RetrievalResult
        retrieval_results = []
        for i, result in enumerate(results, 1):
            retrieval_results.append(
                RetrievalResult(
                    rank=i,
                    chunk_id=str(result.get("chunk_id", i)),
                    document=result.get("document", "unknown"),
                    similarity=result.get("similarity", 0.0),
                    text=result.get("text", "")[:200],  # Solo preview
                    vector_score=result.get("similarity", 0.0),
                )
            )

        return retrieval_results


class HybridMCPSystem:
    """Sistema Hybrid usando RAG + Knowledge Graph"""

    def __init__(self):
        # Crear extractor (sin modo)
        self.extractor = ContextExtractor()

    def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Búsqueda usando RAG + KG híbrido

        Args:
            query: Consulta del usuario
            k: Top-k resultados

        Returns:
            Lista de resultados ordenados por score híbrido
        """
        # Extraer contexto usando método híbrido
        results = self.extractor.extract_hybrid(query, k=k)

        # Convertir a RetrievalResult
        retrieval_results = []
        for i, result in enumerate(results, 1):
            retrieval_results.append(
                RetrievalResult(
                    rank=i,
                    chunk_id=str(result.get("chunk_id", i)),
                    document=result.get("document", "unknown"),
                    similarity=result.get("hybrid_score", result.get("similarity", 0.0)),
                    text=result.get("text", "")[:200],
                    hybrid_score=result.get("hybrid_score", 0.0),
                    vector_score=result.get("vector_score", 0.0),
                    graph_score=result.get("graph_score", 0.0),
                )
            )

        return retrieval_results


class SystemEvaluator:
    """Evaluador de sistemas de recuperación"""

    def __init__(self):
        self.metrics = EvaluationMetrics()

    def evaluate_single_query(
        self, query_data: Dict, results: List[RetrievalResult], k: int = 10
    ) -> Dict:
        """
        Evalúa una query individual

        Args:
            query_data: Dict con query y chunks relevantes
            results: Resultados del sistema
            k: Top-k para métricas

        Returns:
            Dict con métricas de la query
        """
        query = query_data["query"]
        relevant_chunks = [str(c) for c in query_data.get("relevant_chunks", [])]
        relevance_dict = query_data.get("relevance_scores", {})

        # Convertir relevance_scores a dict con string keys
        relevance_dict = {str(k): v for k, v in relevance_dict.items()}

        # Extraer chunk IDs recuperados
        retrieved_ids = [r.chunk_id for r in results[:k]]

        # Crear lista de relevancia para nDCG
        # Escala: 0=no relevante, 1-3 según anotación
        retrieved_relevance = [relevance_dict.get(str(chunk_id), 0) for chunk_id in retrieved_ids]

        # Calcular métricas
        ndcg = self.metrics.ndcg_at_k(retrieved_relevance, k)
        recall = self.metrics.recall_at_k(retrieved_ids, relevant_chunks, k)
        precision = self.metrics.precision_at_k(retrieved_ids, relevant_chunks, k)

        return {
            "query": query[:100],  # Preview
            "num_relevant": len(relevant_chunks),
            "num_retrieved": len(results),
            "ndcg@10": round(ndcg, 4),
            "recall@10": round(recall, 4),
            "precision@10": round(precision, 4),
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_chunks,
            "top_scores": [round(r.similarity, 4) for r in results[:3]],
        }

    def evaluate_system(self, system, queries: List[Dict], k: int = 10) -> Dict:
        """
        Evalúa sistema completo con conjunto de queries

        Args:
            system: Sistema a evaluar (RAGBaseline o HybridMCP)
            queries: Lista de queries anotadas
            k: Top-k para métricas

        Returns:
            Dict con métricas agregadas y detalles
        """
        all_ndcg = []
        all_recall = []
        all_precision = []
        query_results = []

        system_name = system.__class__.__name__
        print(f"\n🧪 Evaluando {system_name}...")

        for i, query_data in enumerate(queries, 1):
            query = query_data["query"]
            print(f"   [{i:2d}/{len(queries)}] {query[:60]}...")

            try:
                # Ejecutar búsqueda
                results = system.search(query, k=k)

                # Evaluar query
                query_eval = self.evaluate_single_query(query_data, results, k)

                all_ndcg.append(query_eval["ndcg@10"])
                all_recall.append(query_eval["recall@10"])
                all_precision.append(query_eval["precision@10"])

                query_results.append({"query_id": i, **query_eval})

            except Exception as e:
                print(f"   ⚠️ Error: {e}")
                continue

        # Calcular estadísticas agregadas
        return {
            "system": system_name,
            "num_queries": len(queries),
            "num_evaluated": len(query_results),
            "metrics": {
                "mean_ndcg@10": round(np.mean(all_ndcg), 4),
                "mean_recall@10": round(np.mean(all_recall), 4),
                "mean_precision@10": round(np.mean(all_precision), 4),
                "median_ndcg@10": round(np.median(all_ndcg), 4),
                "median_recall@10": round(np.median(all_recall), 4),
                "median_precision@10": round(np.median(all_precision), 4),
                "std_ndcg@10": round(np.std(all_ndcg), 4),
                "std_recall@10": round(np.std(all_recall), 4),
                "std_precision@10": round(np.std(all_precision), 4),
                "min_ndcg@10": round(np.min(all_ndcg), 4),
                "max_ndcg@10": round(np.max(all_ndcg), 4),
            },
            "query_details": query_results,
        }


def compare_systems(baseline_results: Dict, hybrid_results: Dict) -> Dict:
    """
    Compara resultados de baseline vs hybrid

    Args:
        baseline_results: Resultados del RAG baseline
        hybrid_results: Resultados del Hybrid MCP

    Returns:
        Dict con comparación y validación de IOVs
    """
    baseline_metrics = baseline_results["metrics"]
    hybrid_metrics = hybrid_results["metrics"]

    # Calcular mejoras
    recall_improvement = (
        (hybrid_metrics["mean_recall@10"] - baseline_metrics["mean_recall@10"])
        / baseline_metrics["mean_recall@10"]
        * 100
        if baseline_metrics["mean_recall@10"] > 0
        else 0
    )

    ndcg_improvement = (
        (hybrid_metrics["mean_ndcg@10"] - baseline_metrics["mean_ndcg@10"])
        / baseline_metrics["mean_ndcg@10"]
        * 100
        if baseline_metrics["mean_ndcg@10"] > 0
        else 0
    )

    precision_improvement = (
        (hybrid_metrics["mean_precision@10"] - baseline_metrics["mean_precision@10"])
        / baseline_metrics["mean_precision@10"]
        * 100
        if baseline_metrics["mean_precision@10"] > 0
        else 0
    )

    # Validar IOVs (convertir a bool nativo de Python)
    # Umbrales originales de la tesis:
    # - nDCG@10 ≥ 0.75 (excelencia en recuperación)
    # - Recall improvement ≥ 25% (mejora sustancial)
    iov_ndcg_passed = bool(hybrid_metrics["mean_ndcg@10"] >= 0.75)
    iov_recall_improvement_passed = bool(recall_improvement >= 25.0)

    comparison = {
        "validation_timestamp": datetime.now().isoformat(),
        "iov_criteria": {
            "iov_ndcg": {
                "requirement": "nDCG@10 ≥ 0.75 en evaluación automática",
                "achieved": hybrid_metrics["mean_ndcg@10"],
                "passed": iov_ndcg_passed,
            },
            "iov_recall_improvement": {
                "requirement": "Mejora ≥25% en Recall@10 vs baseline RAG con similitud coseno",
                "achieved_improvement_percent": round(recall_improvement, 2),
                "passed": iov_recall_improvement_passed,
            },
        },
        "baseline_vs_hybrid": {
            "ndcg@10": {
                "baseline": baseline_metrics["mean_ndcg@10"],
                "hybrid": hybrid_metrics["mean_ndcg@10"],
                "improvement_percent": round(ndcg_improvement, 2),
                "absolute_improvement": round(
                    hybrid_metrics["mean_ndcg@10"] - baseline_metrics["mean_ndcg@10"], 4
                ),
            },
            "recall@10": {
                "baseline": baseline_metrics["mean_recall@10"],
                "hybrid": hybrid_metrics["mean_recall@10"],
                "improvement_percent": round(recall_improvement, 2),
                "absolute_improvement": round(
                    hybrid_metrics["mean_recall@10"] - baseline_metrics["mean_recall@10"], 4
                ),
            },
            "precision@10": {
                "baseline": baseline_metrics["mean_precision@10"],
                "hybrid": hybrid_metrics["mean_precision@10"],
                "improvement_percent": round(precision_improvement, 2),
                "absolute_improvement": round(
                    hybrid_metrics["mean_precision@10"] - baseline_metrics["mean_precision@10"], 4
                ),
            },
        },
        "summary_for_ai": {
            "hybrid_ndcg": hybrid_metrics["mean_ndcg@10"],
            "hybrid_recall": hybrid_metrics["mean_recall@10"],
            "recall_improvement": f"{recall_improvement:+.1f}%",
            "ndcg_improvement": f"{ndcg_improvement:+.1f}%",
            "iov_ndcg_status": "CUMPLIDO" if iov_ndcg_passed else "NO CUMPLIDO",
            "iov_recall_status": "CUMPLIDO" if iov_recall_improvement_passed else "NO CUMPLIDO",
            "all_iovs_passed": iov_ndcg_passed and iov_recall_improvement_passed,
        },
    }

    return comparison


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="IOV2: Evaluar nDCG@10 y Recall@10 - RAG vs Hybrid"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="evaluation/queries_test_annotated.json",
        help="Archivo JSON con queries anotadas",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/iov2_rag_vs_hybrid.json",
        help="Archivo de salida JSON con resultados",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k documentos para métricas")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📊 IOV2: EVALUACIÓN nDCG@10 y Recall@10 - RAG vs HYBRID MCP")
    print("=" * 70)
    print()

    # Cargar queries
    print("📚 Cargando queries anotadas...")
    with open(args.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"   ✓ {len(queries)} queries cargadas")

    # Inicializar sistemas
    print("\n🔧 Inicializando sistemas...")
    baseline_system = RAGBaselineSystem()
    hybrid_system = HybridMCPSystem()
    print("   ✓ RAG Baseline inicializado")
    print("   ✓ Hybrid MCP inicializado")

    # Evaluar sistemas
    evaluator = SystemEvaluator()

    baseline_results = evaluator.evaluate_system(baseline_system, queries, k=args.k)
    hybrid_results = evaluator.evaluate_system(hybrid_system, queries, k=args.k)

    # Comparar resultados
    print("\n📊 Comparando sistemas...")
    comparison = compare_systems(baseline_results, hybrid_results)

    # Resultados completos
    results = {
        "baseline_rag": baseline_results,
        "hybrid_mcp": hybrid_results,
        "comparison": comparison,
    }

    # Guardar resultados
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Mostrar resumen
    print("\n" + "=" * 70)
    print("📋 RESUMEN IOV2")
    print("=" * 70)
    comp = comparison["baseline_vs_hybrid"]
    iov = comparison["iov_criteria"]

    print(f"\nRAG Baseline:")
    print(f"  • nDCG@10:     {baseline_results['metrics']['mean_ndcg@10']:.4f}")
    print(f"  • Recall@10:   {baseline_results['metrics']['mean_recall@10']:.4f}")
    print(f"  • Precision@10: {baseline_results['metrics']['mean_precision@10']:.4f}")

    print(f"\nHybrid MCP:")
    print(f"  • nDCG@10:     {hybrid_results['metrics']['mean_ndcg@10']:.4f}")
    print(f"  • Recall@10:   {hybrid_results['metrics']['mean_recall@10']:.4f}")
    print(f"  • Precision@10: {hybrid_results['metrics']['mean_precision@10']:.4f}")

    print(f"\nMejoras:")
    print(f"  • nDCG@10:     {comp['ndcg@10']['improvement_percent']:+.1f}%")
    print(f"  • Recall@10:   {comp['recall@10']['improvement_percent']:+.1f}%")
    print(f"  • Precision@10: {comp['precision@10']['improvement_percent']:+.1f}%")

    print(f"\nIOV2 - Validación:")
    print(f"  • nDCG@10 ≥ 0.75: {'✅ CUMPLIDO' if iov['iov_ndcg']['passed'] else '❌ NO CUMPLIDO'}")
    print(f"    (Alcanzado: {iov['iov_ndcg']['achieved']:.4f})")
    print(
        f"  • Recall mejora ≥25%: {'✅ CUMPLIDO' if iov['iov_recall_improvement']['passed'] else '❌ NO CUMPLIDO'}"
    )
    print(f"    (Alcanzado: {iov['iov_recall_improvement']['achieved_improvement_percent']:+.1f}%)")

    print("=" * 70)
    print(f"\n💾 Resultados guardados en: {output_file}")
    print()
    print("📝 Para análisis con IA, usa:")
    print("   - comparison.summary_for_ai: resumen ejecutivo")
    print("   - comparison.baseline_vs_hybrid: mejoras detalladas")
    print("   - baseline_rag.query_details: resultados por query del baseline")
    print("   - hybrid_mcp.query_details: resultados por query del hybrid")
    print()


if __name__ == "__main__":
    main()

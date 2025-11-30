#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para evaluar métricas nDCG@10 y Recall@10 comparando:
- Sistema MCP (híbrido: grafo + vectores)
- RAG baseline (similitud coseno pura)

USO:
    python evaluation/evaluate_retrieval_metrics.py --test-queries queries_test.json
    python evaluation/evaluate_retrieval_metrics.py --generate-queries  # Generar queries de ejemplo
"""

import sys
import os

# Configurar encoding para Windows
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def calculate_dcg(relevances, k=10):
    """Calcular DCG@k"""
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i+1)) for i in 1..k
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(relevances / discounts)


def calculate_ndcg(relevances, k=10):
    """Calcular nDCG@k"""
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_recall(retrieved_ids, relevant_ids, k=10):
    """Calcular Recall@k"""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    if len(relevant_set) == 0:
        return 0.0

    return len(retrieved_set & relevant_set) / len(relevant_set)


class RAGBaseline:
    """RAG baseline con similitud coseno pura"""

    def __init__(self, hnsw_index_path, mappings_path, embedding_model, dimension):
        """
        Args:
            embedding_model: Modelo SentenceTransformer ya inicializado (reutilizado del MCP engine)
        """
        import hnswlib

        self.model = embedding_model  # Reutilizar modelo ya cargado
        self.dimension = dimension

        # Cargar índice HNSW
        self.index = hnswlib.Index(space="cosine", dim=dimension)
        self.index.load_index(str(hnsw_index_path))

        # Cargar mappings
        with open(mappings_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        self.id_to_chunk = mappings["id_to_chunk"]

    def search(self, query, top_k=10):
        """Búsqueda por similitud coseno"""
        # Generar embedding
        query_embedding = self.model.encode([query])[0]

        # Buscar en HNSW
        labels, distances = self.index.knn_query(query_embedding, k=top_k)

        results = []
        for idx, dist in zip(labels[0], distances[0]):
            chunk_data = self.id_to_chunk[str(idx)]
            results.append(
                {
                    "id": idx,
                    "content": chunk_data["content"][:200] + "...",
                    "file_name": chunk_data["file_name"],
                    "distance": float(dist),
                }
            )

        return results


class RetrievalEvaluator:
    """Evaluador de métricas de recuperación"""

    def __init__(self):
        # Pre-cargar modelo ANTES de cualquier otra cosa para evitar conflictos
        print("🔧 Pre-cargando modelo de embeddings...")
        from sentence_transformers import SentenceTransformer

        self.config = load_config()
        emb_cfg = self.config["embeddings"]

        # Cargar modelo UNA SOLA VEZ
        shared_model = SentenceTransformer(emb_cfg["model"])
        print("✅ Modelo cargado")

        # Lazy imports
        from runtime_mcp.engines.hybrid_engine import HybridEngine

        self.base_path = Path(__file__).parent.parent

        # Inicializar sistema MCP (híbrido) con modelo pre-cargado
        print("\n🔧 Inicializando sistema MCP híbrido...")
        neo4j_cfg = self.config["neo4j"]

        self.mcp_engine = HybridEngine(
            neo4j_uri=neo4j_cfg["uri"],
            neo4j_user=neo4j_cfg["user"],
            neo4j_password=neo4j_cfg["password"],
            embedding_model=emb_cfg["model"],
            dimension=emb_cfg["dimension"],
            preloaded_model=shared_model,
        )

        # Cargar índice HNSW
        hnsw_path = self.base_path / self.config["paths"]["hnsw_index"]
        self.mcp_engine.load_index(hnsw_path)
        print(f"✓ Sistema MCP inicializado ({self.mcp_engine.index.get_current_count()} chunks)")

        # Inicializar RAG baseline (reutilizando el mismo modelo)
        print("\n🔧 Inicializando RAG baseline...")
        mappings_path = self.base_path / self.config["paths"]["hnsw_mappings"]
        self.rag_baseline = RAGBaseline(
            hnsw_path,
            mappings_path,
            shared_model,  # Reutilizar el mismo modelo pre-cargado
            emb_cfg["dimension"],
        )
        print("✓ RAG baseline inicializado")

        self.results = {"mcp": {"ndcg": [], "recall": []}, "rag": {"ndcg": [], "recall": []}}

    def evaluate_query(self, query_data, top_k=10, verbose=False):
        """Evaluar una query con ambos sistemas"""
        query = query_data["query"]
        relevant_chunks = [
            str(cid) for cid in query_data["relevant_chunks"]
        ]  # Normalizar a strings
        relevance_scores = query_data.get("relevance_scores", {})  # Scores de relevancia (0-3)

        if verbose:
            print(f"\n📝 Query: {query}")
            print(f"   Chunks relevantes: {len(relevant_chunks)}")

        # ===== MCP Hybrid System =====
        mcp_results = self.mcp_engine.hybrid_search(query, k=top_k)
        mcp_ids = [str(r.chunk_id) for r in mcp_results]  # chunk_id es string

        # Calcular relevancia para nDCG
        mcp_relevances = [relevance_scores.get(cid, 0) for cid in mcp_ids]
        mcp_ndcg = calculate_ndcg(mcp_relevances, k=top_k)
        mcp_recall = calculate_recall(mcp_ids, relevant_chunks, k=top_k)

        # ===== RAG Baseline =====
        rag_results = self.rag_baseline.search(query, top_k=top_k)
        rag_ids = [str(r["id"]) for r in rag_results]  # Normalizar a string también

        rag_relevances = [relevance_scores.get(cid, 0) for cid in rag_ids]
        rag_ndcg = calculate_ndcg(rag_relevances, k=top_k)
        rag_recall = calculate_recall(rag_ids, relevant_chunks, k=top_k)

        if verbose:
            print(f"   MCP:  nDCG@{top_k}={mcp_ndcg:.4f}, Recall@{top_k}={mcp_recall:.4f}")
            print(f"   RAG:  nDCG@{top_k}={rag_ndcg:.4f}, Recall@{top_k}={rag_recall:.4f}")

        self.results["mcp"]["ndcg"].append(mcp_ndcg)
        self.results["mcp"]["recall"].append(mcp_recall)
        self.results["rag"]["ndcg"].append(rag_ndcg)
        self.results["rag"]["recall"].append(rag_recall)

        return {
            "query": query,
            "mcp": {"ndcg": mcp_ndcg, "recall": mcp_recall, "results": mcp_ids[:5]},
            "rag": {"ndcg": rag_ndcg, "recall": rag_recall, "results": rag_ids[:5]},
        }

    def evaluate_all(self, test_queries, top_k=10, verbose=True):
        """Evaluar todas las queries"""
        print(f"\n🎯 EVALUANDO {len(test_queries)} QUERIES\n")
        print("=" * 80)

        detailed_results = []
        for i, query_data in enumerate(test_queries, 1):
            if verbose:
                print(f"\n[{i}/{len(test_queries)}]", end="")
            result = self.evaluate_query(query_data, top_k=top_k, verbose=verbose)
            detailed_results.append(result)

        return detailed_results

    def compute_metrics(self):
        """Calcular métricas agregadas"""
        metrics = {}

        for system in ["mcp", "rag"]:
            ndcg_scores = self.results[system]["ndcg"]
            recall_scores = self.results[system]["recall"]

            metrics[system] = {
                "ndcg_mean": np.mean(ndcg_scores),
                "ndcg_std": np.std(ndcg_scores),
                "recall_mean": np.mean(recall_scores),
                "recall_std": np.std(recall_scores),
                "ndcg_all": ndcg_scores,
                "recall_all": recall_scores,
            }

        return metrics

    def print_report(self, metrics):
        """Imprimir reporte de resultados"""
        print("\n" + "=" * 80)
        print("📊 RESULTADOS FINALES")
        print("=" * 80)

        print("\n🔹 RAG Baseline (Similitud Coseno):")
        print(f"   nDCG@10:  {metrics['rag']['ndcg_mean']:.4f} ± {metrics['rag']['ndcg_std']:.4f}")
        print(
            f"   Recall@10: {metrics['rag']['recall_mean']:.4f} ± {metrics['rag']['recall_std']:.4f}"
        )

        print("\n🔹 MCP Híbrido (Grafo + Vectores):")
        print(f"   nDCG@10:  {metrics['mcp']['ndcg_mean']:.4f} ± {metrics['mcp']['ndcg_std']:.4f}")
        print(
            f"   Recall@10: {metrics['mcp']['recall_mean']:.4f} ± {metrics['mcp']['recall_std']:.4f}"
        )

        # Calcular mejoras
        ndcg_improvement = (
            (metrics["mcp"]["ndcg_mean"] - metrics["rag"]["ndcg_mean"])
            / metrics["rag"]["ndcg_mean"]
            * 100
        )
        recall_improvement = (
            (metrics["mcp"]["recall_mean"] - metrics["rag"]["recall_mean"])
            / metrics["rag"]["recall_mean"]
            * 100
        )

        print("\n📈 MEJORAS:")
        print(f"   nDCG@10:  {ndcg_improvement:+.2f}%")
        print(f"   Recall@10: {recall_improvement:+.2f}%")

        # Verificar IOVs
        print("\n📋 VERIFICACIÓN DE IOVs:")

        iov2_passed = metrics["mcp"]["ndcg_mean"] >= 0.75
        print(f"   IOV #2 (nDCG@10 ≥ 0.75): {'✅ CUMPLIDO' if iov2_passed else '❌ NO CUMPLIDO'}")
        print(f"           Valor obtenido: {metrics['mcp']['ndcg_mean']:.4f}")

        iov3_passed = recall_improvement >= 25.0
        print(
            f"   IOV #3 (Mejora ≥ 25% en Recall@10): {'✅ CUMPLIDO' if iov3_passed else '❌ NO CUMPLIDO'}"
        )
        print(f"           Mejora obtenida: {recall_improvement:.2f}%")

        return {
            "iov2_passed": iov2_passed,
            "iov3_passed": iov3_passed,
            "ndcg_improvement": ndcg_improvement,
            "recall_improvement": recall_improvement,
        }

    def plot_results(self, metrics, output_dir):
        """Generar gráficas de comparación"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 5)

        # Gráfica 1: Comparación de métricas promedio
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        systems = ["RAG Baseline", "MCP Híbrido"]
        ndcg_values = [metrics["rag"]["ndcg_mean"], metrics["mcp"]["ndcg_mean"]]
        recall_values = [metrics["rag"]["recall_mean"], metrics["mcp"]["recall_mean"]]

        # nDCG@10
        bars1 = ax1.bar(systems, ndcg_values, color=["#ff7f0e", "#2ca02c"], alpha=0.8)
        ax1.axhline(y=0.75, color="r", linestyle="--", label="Umbral IOV (0.75)")
        ax1.set_ylabel("nDCG@10", fontsize=12)
        ax1.set_title("Normalized Discounted Cumulative Gain @ 10", fontsize=13, fontweight="bold")
        ax1.set_ylim(0, 1.0)
        ax1.legend()

        for bar, val in zip(bars1, ndcg_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        # Recall@10
        bars2 = ax2.bar(systems, recall_values, color=["#ff7f0e", "#2ca02c"], alpha=0.8)
        ax2.set_ylabel("Recall@10", fontsize=12)
        ax2.set_title("Recall @ 10", fontsize=13, fontweight="bold")
        ax2.set_ylim(0, 1.0)

        for bar, val in zip(bars2, recall_values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        plt.tight_layout()
        plt.savefig(output_dir / "metricas_comparacion.png", dpi=300, bbox_inches="tight")
        print(f"\n📊 Gráfica guardada: {output_dir / 'metricas_comparacion.png'}")
        plt.close()

        # Gráfica 2: Distribución de scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(
            [metrics["rag"]["ndcg_all"], metrics["mcp"]["ndcg_all"]],
            bins=15,
            label=systems,
            alpha=0.7,
            color=["#ff7f0e", "#2ca02c"],
        )
        ax1.set_xlabel("nDCG@10", fontsize=12)
        ax1.set_ylabel("Frecuencia", fontsize=12)
        ax1.set_title("Distribución de nDCG@10", fontsize=13, fontweight="bold")
        ax1.legend()
        ax1.axvline(x=0.75, color="r", linestyle="--", alpha=0.5)

        ax2.hist(
            [metrics["rag"]["recall_all"], metrics["mcp"]["recall_all"]],
            bins=15,
            label=systems,
            alpha=0.7,
            color=["#ff7f0e", "#2ca02c"],
        )
        ax2.set_xlabel("Recall@10", fontsize=12)
        ax2.set_ylabel("Frecuencia", fontsize=12)
        ax2.set_title("Distribución de Recall@10", fontsize=13, fontweight="bold")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "distribucion_scores.png", dpi=300, bbox_inches="tight")
        print(f"📊 Gráfica guardada: {output_dir / 'distribucion_scores.png'}")
        plt.close()

    def save_results(self, metrics, iov_results, detailed_results, output_path):
        """Guardar resultados en JSON"""

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)

        output = {
            "summary": {
                "rag_baseline": {
                    "ndcg_mean": float(metrics["rag"]["ndcg_mean"]),
                    "ndcg_std": float(metrics["rag"]["ndcg_std"]),
                    "recall_mean": float(metrics["rag"]["recall_mean"]),
                    "recall_std": float(metrics["rag"]["recall_std"]),
                },
                "mcp_hybrid": {
                    "ndcg_mean": float(metrics["mcp"]["ndcg_mean"]),
                    "ndcg_std": float(metrics["mcp"]["ndcg_std"]),
                    "recall_mean": float(metrics["mcp"]["recall_mean"]),
                    "recall_std": float(metrics["mcp"]["recall_std"]),
                },
                "improvements": {
                    "ndcg_improvement_pct": float(iov_results["ndcg_improvement"]),
                    "recall_improvement_pct": float(iov_results["recall_improvement"]),
                },
                "iov_verification": {
                    "iov2_ndcg_ge_075": bool(iov_results["iov2_passed"]),
                    "iov3_recall_improvement_ge_25pct": bool(iov_results["iov3_passed"]),
                },
            },
            "detailed_results": detailed_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        print(f"\n💾 Resultados guardados en: {output_path}")


def generate_example_queries():
    """Generar queries de ejemplo para evaluación"""
    # Estas queries deberían ser anotadas manualmente con chunks relevantes
    queries = [
        {
            "query": "¿Qué es un agente reactivo y cómo funciona?",
            "relevant_chunks": [0, 1, 2],  # IDs de chunks relevantes (anotar manualmente)
            "relevance_scores": {
                "0": 3,
                "1": 2,
                "2": 2,
            },  # 0-3: no relevante, algo, relevante, muy relevante
        },
        {
            "query": "Explica el algoritmo de búsqueda A* y su función heurística",
            "relevant_chunks": [],  # Completar con IDs reales
            "relevance_scores": {},
        },
        # Agregar más queries...
    ]

    output_path = Path(__file__).parent / "queries_test_template.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    print(f"✅ Template de queries generado en: {output_path}")
    print("\n📝 IMPORTANTE: Debes anotar manualmente:")
    print("   1. Los IDs de chunks relevantes para cada query")
    print("   2. Los scores de relevancia (0-3) para cada chunk")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluar nDCG@10 y Recall@10")
    parser.add_argument("--test-queries", help="Archivo JSON con queries de prueba anotadas")
    parser.add_argument(
        "--generate-queries", action="store_true", help="Generar template de queries de ejemplo"
    )
    parser.add_argument(
        "--output-dir", default="evaluation/results", help="Directorio para guardar resultados"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="K para nDCG@K y Recall@K (default: 10)"
    )

    args = parser.parse_args()

    if args.generate_queries:
        generate_example_queries()
        return 0

    if not args.test_queries:
        print("❌ Error: Debes proporcionar --test-queries o usar --generate-queries")
        return 1

    # Cargar queries de prueba
    with open(args.test_queries, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    print(f"📄 Cargadas {len(test_queries)} queries de prueba")

    # Inicializar evaluador
    evaluator = RetrievalEvaluator()

    # Evaluar
    detailed_results = evaluator.evaluate_all(test_queries, top_k=args.top_k)

    # Calcular métricas
    metrics = evaluator.compute_metrics()

    # Imprimir reporte
    iov_results = evaluator.print_report(metrics)

    # Generar gráficas
    evaluator.plot_results(metrics, args.output_dir)

    # Guardar resultados
    output_path = Path(args.output_dir) / "evaluation_results.json"
    evaluator.save_results(metrics, iov_results, detailed_results, output_path)

    # Cerrar conexiones
    evaluator.mcp_engine.driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

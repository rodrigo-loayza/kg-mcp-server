"""
SCRIPT COMPLETO PARA GENERAR GRÁFICOS DE EVALUACIÓN
Genera las visualizaciones de métricas RAG vs MCP Híbrido
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11


def load_results(results_path="evaluation/results/evaluation_results.json"):
    """Cargar resultados de evaluación"""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metrics_comparison(results, output_dir="evaluation/results"):
    """
    Gráfico de barras comparando nDCG@10 y Recall@10
    """
    summary = results["summary"]

    metrics = ["nDCG@10", "Recall@10"]
    rag_scores = [summary["rag_baseline"]["ndcg_mean"], summary["rag_baseline"]["recall_mean"]]
    mcp_scores = [summary["mcp_hybrid"]["ndcg_mean"], summary["mcp_hybrid"]["recall_mean"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2, rag_scores, width, label="RAG Baseline", color="#3498db", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2,
        mcp_scores,
        width,
        label="MCP Híbrido (Grafo + Vector)",
        color="#2ecc71",
        alpha=0.8,
    )

    # Etiquetas
    ax.set_xlabel("Métricas", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Comparación de Métricas: RAG Baseline vs MCP Híbrido",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim([0, 1])

    # Agregar valores sobre las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Línea de referencia para IOV
    ax.axhline(
        y=0.75, color="red", linestyle="--", linewidth=1.5, label="IOV Target (0.75)", alpha=0.6
    )
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    output_path = Path(output_dir) / "metricas_comparacion.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def plot_score_distribution(results, output_dir="evaluation/results"):
    """
    Distribución de scores por sistema
    """
    detailed = results["detailed_results"]

    rag_ndcg = [q["rag"]["ndcg"] for q in detailed]
    mcp_ndcg = [q["mcp"]["ndcg"] for q in detailed]
    rag_recall = [q["rag"]["recall"] for q in detailed]
    mcp_recall = [q["mcp"]["recall"] for q in detailed]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # nDCG Distribution
    axes[0].hist(rag_ndcg, bins=10, alpha=0.6, label="RAG", color="#3498db", edgecolor="black")
    axes[0].hist(mcp_ndcg, bins=10, alpha=0.6, label="MCP", color="#2ecc71", edgecolor="black")
    axes[0].set_xlabel("nDCG@10 Score", fontsize=12)
    axes[0].set_ylabel("Frecuencia", fontsize=12)
    axes[0].set_title("Distribución de nDCG@10", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Recall Distribution
    axes[1].hist(rag_recall, bins=10, alpha=0.6, label="RAG", color="#3498db", edgecolor="black")
    axes[1].hist(mcp_recall, bins=10, alpha=0.6, label="MCP", color="#2ecc71", edgecolor="black")
    axes[1].set_xlabel("Recall@10 Score", fontsize=12)
    axes[1].set_ylabel("Frecuencia", fontsize=12)
    axes[1].set_title("Distribución de Recall@10", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "distribucion_scores.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def plot_per_query_comparison(results, output_dir="evaluation/results"):
    """
    Comparación por query individual
    """
    detailed = results["detailed_results"]

    queries = [f"Q{i+1}" for i in range(len(detailed))]
    rag_ndcg = [q["rag"]["ndcg"] for q in detailed]
    mcp_ndcg = [q["mcp"]["ndcg"] for q in detailed]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(queries))
    width = 0.35

    bars1 = ax.bar(x - width / 2, rag_ndcg, width, label="RAG", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, mcp_ndcg, width, label="MCP Híbrido", color="#2ecc71", alpha=0.8)

    ax.set_xlabel("Query", fontsize=12, fontweight="bold")
    ax.set_ylabel("nDCG@10", fontsize=12, fontweight="bold")
    ax.set_title("Comparación nDCG@10 por Query", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=0)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")

    # Valores sobre barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    output_path = Path(output_dir) / "comparacion_por_query.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def plot_improvement_heatmap(results, output_dir="evaluation/results"):
    """
    Heatmap de mejoras por query
    """
    detailed = results["detailed_results"]

    queries = [f"Q{i+1}" for i in range(len(detailed))]
    ndcg_improvements = [
        ((q["mcp"]["ndcg"] - q["rag"]["ndcg"]) / (q["rag"]["ndcg"] + 1e-10)) * 100 for q in detailed
    ]
    recall_improvements = [
        ((q["mcp"]["recall"] - q["rag"]["recall"]) / (q["rag"]["recall"] + 1e-10)) * 100
        for q in detailed
    ]

    data = np.array([ndcg_improvements, recall_improvements])

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-50, vmax=50)

    ax.set_xticks(np.arange(len(queries)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(queries)
    ax.set_yticklabels(["nDCG@10", "Recall@10"])

    ax.set_title("Mejora Porcentual: MCP vs RAG (%)", fontsize=14, fontweight="bold", pad=20)

    # Agregar valores
    for i in range(2):
        for j in range(len(queries)):
            text = ax.text(
                j, i, f"{data[i, j]:.1f}%", ha="center", va="center", color="black", fontsize=10
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mejora (%)", rotation=270, labelpad=20)

    plt.tight_layout()
    output_path = Path(output_dir) / "mejoras_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def generate_all_plots(
    results_path="evaluation/results/evaluation_results.json", output_dir="evaluation/results"
):
    """
    Generar todos los gráficos
    """
    print("\n🎨 GENERANDO GRÁFICOS DE EVALUACIÓN\n")

    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Cargar resultados
    results = load_results(results_path)

    # Generar gráficos
    plot_metrics_comparison(results, output_dir)
    plot_score_distribution(results, output_dir)
    plot_per_query_comparison(results, output_dir)
    plot_improvement_heatmap(results, output_dir)

    print("\n✅ Todos los gráficos generados exitosamente")
    print(f"📁 Ubicación: {output_dir}/\n")


if __name__ == "__main__":
    # CONFIGURACIÓN
    RESULTS_PATH = "evaluation/results/evaluation_results.json"
    OUTPUT_DIR = "evaluation/results"

    # Generar todos los gráficos
    generate_all_plots(RESULTS_PATH, OUTPUT_DIR)

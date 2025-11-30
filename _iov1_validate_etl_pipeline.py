#!/usr/bin/env python3
"""
IOV1: Validación ETL Pipeline - Pipeline ETL operativo para repositorio de documentos completo
con procesamiento exitoso de ≥3 formatos documentales

Objetivo:
    Validar que el ETL pipeline offline puede procesar exitosamente documentos en múltiples
    formatos (PDF, DOCX, TXT, IPYNB, etc.) generando chunks semánticos con embeddings.

Métricas evaluadas:
    - Número de formatos diferentes procesados (≥3)
    - Total de documentos procesados
    - Total de chunks generados
    - Distribución de chunks por documento
    - Cobertura de embeddings (% chunks con vectores)

Salida:
    JSON con estadísticas completas para análisis y redacción con IA

Uso:
    python _iov1_validate_etl_pipeline.py --input-dir data/raw/INF265 --processed-dir data/processed --output results/iov1_etl_validation.json

Nota:
    El script debe ejecutarse DESPUÉS de correr el ETL pipeline:
    python offline_etl/main_etl.py --input data/raw/INF265 --output data/processed --max-documents 50
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
import numpy as np

# Agregar offline_etl al path
sys.path.insert(0, str(Path(__file__).parent))


class ETLPipelineValidator:
    """Valida el pipeline ETL offline y genera estadísticas"""

    def __init__(self, input_dir: Path, processed_dir: Path):
        self.input_dir = Path(input_dir)
        self.processed_dir = Path(processed_dir)

    def collect_input_files(self) -> Dict[str, List[Path]]:
        """
        Recolecta archivos de entrada por formato

        Returns:
            Dict con formato -> lista de archivos
        """
        format_files = defaultdict(list)

        # Buscar recursivamente en directorio de entrada
        for ext in [".pdf", ".docx", ".txt", ".md", ".html", ".pptx", ".ipynb"]:
            files = list(self.input_dir.rglob(f"*{ext}"))
            if files:
                format_files[ext.upper().replace(".", "")] = files

        return dict(format_files)

    def load_processing_summary(self) -> Dict:
        """
        Carga el summary de documentos procesados desde chunks

        El ETL no genera processing_summary.json, sino que guarda
        directamente los chunks con metadata
        """
        documents = []
        total_images = 0

        # Buscar todos los archivos *_chunks.json
        chunk_files = list(self.processed_dir.glob("*_chunks.json"))

        for chunk_file in chunk_files:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if not chunks:
                continue

            # Obtener metadata del primer chunk
            first_chunk = chunks[0]
            doc_name = first_chunk.get("source", chunk_file.stem.replace("_chunks", ""))
            file_type = first_chunk.get("file_type", "UNKNOWN")
            relative_path = first_chunk.get("relative_path", doc_name)

            # Detectar formato desde file_type o nombre
            fmt = file_type.upper().replace(".", "")
            if fmt not in ["PDF", "DOCX", "TXT", "MD", "HTML", "PPTX", "IPYNB"]:
                # Intentar detectar desde el nombre
                if ".pdf" in doc_name.lower():
                    fmt = "PDF"
                elif ".docx" in doc_name.lower():
                    fmt = "DOCX"
                elif ".txt" in doc_name.lower():
                    fmt = "TXT"
                elif ".ipynb" in doc_name.lower():
                    fmt = "IPYNB"
                else:
                    fmt = "UNKNOWN"

            documents.append(
                {
                    "doc_id": chunk_file.stem.replace("_chunks", ""),
                    "filepath": relative_path,
                    "metadata": {
                        "format": fmt,
                        "pages": 0,  # No disponible en chunks
                        "num_chunks": len(chunks),
                    },
                }
            )

        return {
            "total_documents": len(documents),
            "total_images": total_images,
            "documents": documents,
        }

    def analyze_chunks(self) -> Dict:
        """
        Analiza chunks procesados por documento

        Returns:
            Dict con análisis de chunks
        """
        chunks_by_doc = {}
        total_chunks = 0

        # Buscar archivos de chunks
        for chunk_file in self.processed_dir.glob("*_chunks.json"):
            doc_name = chunk_file.stem.replace("_chunks", "")

            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            chunks_by_doc[doc_name] = {
                "count": len(chunks),
                "has_embeddings": False,
                "avg_chunk_length": (
                    np.mean([len(c.get("text", "")) for c in chunks]) if chunks else 0
                ),
            }

            total_chunks += len(chunks)

            # Verificar si hay embeddings
            embedding_file = chunk_file.parent / f"{doc_name}_embeddings.npy"
            if embedding_file.exists():
                chunks_by_doc[doc_name]["has_embeddings"] = True

        return {
            "total_chunks": total_chunks,
            "chunks_by_document": chunks_by_doc,
            "num_documents_with_chunks": len(chunks_by_doc),
            "num_documents_with_embeddings": sum(
                1 for d in chunks_by_doc.values() if d["has_embeddings"]
            ),
        }

    def calculate_statistics(self, summary: Dict, chunks_analysis: Dict) -> Dict:
        """
        Calcula estadísticas completas del pipeline

        Returns:
            Dict con estadísticas para IOV1
        """
        # Identificar formatos procesados
        formats_processed = set()
        for doc in summary.get("documents", []):
            fmt = doc.get("metadata", {}).get("format", "UNKNOWN")
            formats_processed.add(fmt)

        # Distribución de chunks
        chunks_counts = [d["count"] for d in chunks_analysis["chunks_by_document"].values()]

        stats = {
            "validation_timestamp": datetime.now().isoformat(),
            "iov_criteria": {
                "required_formats": 3,
                "description": "Pipeline ETL operativo con procesamiento exitoso de ≥3 formatos documentales",
            },
            "etl_pipeline_status": {
                "total_input_documents": summary.get("total_documents", 0),
                "formats_processed": list(formats_processed),
                "num_formats": len(formats_processed),
                "meets_requirement": len(formats_processed) >= 3,
                "total_images_extracted": summary.get("total_images", 0),
            },
            "chunk_generation": {
                "total_chunks": chunks_analysis["total_chunks"],
                "documents_with_chunks": chunks_analysis["num_documents_with_chunks"],
                "chunks_per_document": {
                    "mean": round(np.mean(chunks_counts), 2) if chunks_counts else 0,
                    "median": round(np.median(chunks_counts), 2) if chunks_counts else 0,
                    "min": int(np.min(chunks_counts)) if chunks_counts else 0,
                    "max": int(np.max(chunks_counts)) if chunks_counts else 0,
                    "std": round(np.std(chunks_counts), 2) if chunks_counts else 0,
                },
                "distribution": {
                    "0-50 chunks": sum(1 for c in chunks_counts if c <= 50),
                    "51-100 chunks": sum(1 for c in chunks_counts if 50 < c <= 100),
                    "101-200 chunks": sum(1 for c in chunks_counts if 100 < c <= 200),
                    "200+ chunks": sum(1 for c in chunks_counts if c > 200),
                },
            },
            "embedding_coverage": {
                "documents_with_embeddings": chunks_analysis["num_documents_with_embeddings"],
                "coverage_percentage": round(
                    (
                        (
                            chunks_analysis["num_documents_with_embeddings"]
                            / chunks_analysis["num_documents_with_chunks"]
                            * 100
                        )
                        if chunks_analysis["num_documents_with_chunks"] > 0
                        else 0
                    ),
                    2,
                ),
            },
            "document_details": [],
        }

        # Detalles por documento
        for doc in summary.get("documents", []):
            doc_id = doc.get("doc_id", "unknown")
            doc_name = Path(doc.get("filepath", "")).stem

            chunk_info = chunks_analysis["chunks_by_document"].get(doc_name, {})

            stats["document_details"].append(
                {
                    "doc_id": doc_id,
                    "filename": Path(doc.get("filepath", "")).name,
                    "format": doc.get("metadata", {}).get("format", "UNKNOWN"),
                    "pages": doc.get("metadata", {}).get("pages", 0),
                    "chunks": chunk_info.get("count", 0),
                    "has_embeddings": chunk_info.get("has_embeddings", False),
                    "avg_chunk_length": round(chunk_info.get("avg_chunk_length", 0), 2),
                }
            )

        return stats

    def generate_visualization_data(self, stats: Dict) -> Dict:
        """
        Genera datos para visualización (gráficas austeras)

        Returns:
            Dict con datos listos para plotear
        """
        viz_data = {
            "charts": {
                "formats_processed": {
                    "type": "bar",
                    "title": "Formatos Procesados por ETL Pipeline",
                    "data": {
                        "labels": stats["etl_pipeline_status"]["formats_processed"],
                        "values": [
                            sum(1 for d in stats["document_details"] if d["format"] == fmt)
                            for fmt in stats["etl_pipeline_status"]["formats_processed"]
                        ],
                    },
                },
                "chunks_distribution": {
                    "type": "histogram",
                    "title": "Distribución de Chunks por Documento",
                    "data": {
                        "bins": list(stats["chunk_generation"]["distribution"].keys()),
                        "counts": list(stats["chunk_generation"]["distribution"].values()),
                    },
                },
                "embedding_coverage": {
                    "type": "pie",
                    "title": "Cobertura de Embeddings",
                    "data": {
                        "labels": ["Con Embeddings", "Sin Embeddings"],
                        "values": [
                            stats["embedding_coverage"]["documents_with_embeddings"],
                            stats["chunk_generation"]["documents_with_chunks"]
                            - stats["embedding_coverage"]["documents_with_embeddings"],
                        ],
                    },
                },
            },
            "summary_for_ai": {
                "total_documents": stats["etl_pipeline_status"]["total_input_documents"],
                "total_formats": stats["etl_pipeline_status"]["num_formats"],
                "total_chunks": stats["chunk_generation"]["total_chunks"],
                "avg_chunks_per_doc": stats["chunk_generation"]["chunks_per_document"]["mean"],
                "embedding_coverage": stats["embedding_coverage"]["coverage_percentage"],
                "iov1_passed": stats["etl_pipeline_status"]["meets_requirement"],
            },
        }

        return viz_data

    def validate_pipeline(self) -> Dict:
        """
        Ejecuta validación completa del pipeline ETL

        Returns:
            Dict con estadísticas completas para IOV1
        """
        print("\n" + "=" * 70)
        print("📊 IOV1: VALIDACIÓN ETL PIPELINE - PROCESAMIENTO MULTI-FORMATO")
        print("=" * 70)
        print()

        # 1. Recolectar archivos de entrada
        print("🔍 Analizando archivos de entrada...")
        input_files = self.collect_input_files()
        print(f"   ✓ Formatos detectados: {list(input_files.keys())}")

        # 2. Cargar summary de procesamiento
        print("\n📝 Cargando summary de procesamiento...")
        summary = self.load_processing_summary()
        print(f"   ✓ Documentos procesados: {summary.get('total_documents', 0)}")

        # 3. Analizar chunks
        print("\n🧩 Analizando chunks generados...")
        chunks_analysis = self.analyze_chunks()
        print(f"   ✓ Total de chunks: {chunks_analysis['total_chunks']}")
        print(f"   ✓ Documentos con embeddings: {chunks_analysis['num_documents_with_embeddings']}")

        # 4. Calcular estadísticas
        print("\n📈 Calculando estadísticas...")
        stats = self.calculate_statistics(summary, chunks_analysis)

        # 5. Generar datos de visualización
        print("\n📊 Generando datos de visualización...")
        viz_data = self.generate_visualization_data(stats)

        # Combinar resultados
        results = {"statistics": stats, "visualizations": viz_data}

        # Mostrar resumen
        print("\n" + "=" * 70)
        print("📋 RESUMEN IOV1")
        print("=" * 70)
        print(
            f"Formatos procesados: {stats['etl_pipeline_status']['num_formats']} "
            f"({', '.join(stats['etl_pipeline_status']['formats_processed'])})"
        )
        print(f"Documentos procesados: {stats['etl_pipeline_status']['total_input_documents']}")
        print(f"Total chunks: {stats['chunk_generation']['total_chunks']}")
        print(f"Cobertura embeddings: {stats['embedding_coverage']['coverage_percentage']}%")
        print()

        if stats["etl_pipeline_status"]["meets_requirement"]:
            print("✅ IOV1 CUMPLIDO: ≥3 formatos procesados exitosamente")
        else:
            print(
                f"❌ IOV1 NO CUMPLIDO: Solo {stats['etl_pipeline_status']['num_formats']} "
                f"formatos procesados (se requieren ≥3)"
            )
        print("=" * 70)
        print()

        return results


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="IOV1: Validar ETL Pipeline - Procesamiento Multi-Formato"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/INF265",
        help="Directorio con archivos de entrada (mismo que --input en main_etl.py)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directorio con documentos procesados (mismo que --output en main_etl.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/iov1_etl_validation.json",
        help="Archivo de salida JSON con estadísticas",
    )

    args = parser.parse_args()

    # Crear validador
    validator = ETLPipelineValidator(
        input_dir=Path(args.input_dir), processed_dir=Path(args.processed_dir)
    )

    # Ejecutar validación
    results = validator.validate_pipeline()

    # Guardar resultados
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 Resultados guardados en: {output_file}")
    print()
    print("📝 Para análisis con IA, usa:")
    print(f"   - statistics: métricas completas del pipeline")
    print(f"   - visualizations.summary_for_ai: resumen ejecutivo")
    print(f"   - visualizations.charts: datos para gráficas")
    print()


if __name__ == "__main__":
    main()

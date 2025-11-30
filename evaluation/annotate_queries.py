#!/usr/bin/env python
"""
Script interactivo para anotar queries con chunks relevantes.
Facilita la creación del dataset de prueba para evaluación.

USO:
    python evaluation/annotate_queries.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_all_chunks():
    """Cargar todos los chunks disponibles desde HNSW mappings"""
    config = load_config()
    base_path = Path(__file__).parent.parent
    mappings_path = base_path / config["paths"]["hnsw_mappings"]

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    return mappings["id_to_chunk"]


def search_chunks_by_keywords(chunks, keywords):
    """Buscar chunks que contengan las keywords"""
    keywords = [k.lower() for k in keywords]
    results = []

    for chunk_id, chunk_data in chunks.items():
        content_lower = chunk_data["content"].lower()

        # Contar cuántas keywords aparecen
        matches = sum(1 for kw in keywords if kw in content_lower)

        if matches > 0:
            results.append(
                {
                    "id": int(chunk_id),
                    "matches": matches,
                    "content": chunk_data["content"][:1000] + "...",
                    "file": chunk_data["file_name"],
                    "chunk_index": chunk_data.get("chunk_index", "?"),
                }
            )

    # Ordenar por número de matches
    results.sort(key=lambda x: x["matches"], reverse=True)
    return results


def display_chunks(chunks, max_display=10):
    """Mostrar chunks de forma legible"""
    print(f"\n{'='*80}")
    print(
        f"Se encontraron {len(chunks)} chunks. Mostrando los primeros {min(len(chunks), max_display)}:"
    )
    print(f"{'='*80}\n")

    for i, chunk in enumerate(chunks[:max_display], 1):
        print(f"[{i}] ID: {chunk['id']} | Matches: {chunk['matches']}")
        print(f"    Archivo: {chunk['file']} (Chunk #{chunk['chunk_index']})")
        print(f"    Contenido: {chunk['content']}")
        print()


def annotate_query(query_text, all_chunks):
    """Proceso de anotación para una query"""
    print(f"\n{'='*80}")
    print(f"QUERY: {query_text}")
    print(f"{'='*80}")

    print("\n1️⃣  Introduce keywords para buscar chunks relevantes (separadas por coma):")
    keywords_input = input("   Keywords: ").strip()

    if not keywords_input:
        print("❌ No se ingresaron keywords. Saltando query.")
        return None

    keywords = [kw.strip() for kw in keywords_input.split(",")]

    # Buscar chunks
    matching_chunks = search_chunks_by_keywords(all_chunks, keywords)

    if not matching_chunks:
        print("❌ No se encontraron chunks con esas keywords.")
        return None

    display_chunks(matching_chunks)

    print("\n2️⃣  Selecciona los IDs de chunks relevantes (separados por coma):")
    print("    Ejemplo: 45,67,89")
    relevant_ids_input = input("   IDs relevantes: ").strip()

    if not relevant_ids_input:
        print("❌ No se ingresaron IDs. Saltando query.")
        return None

    try:
        relevant_ids = [int(x.strip()) for x in relevant_ids_input.split(",")]
    except ValueError:
        print("❌ IDs inválidos. Saltando query.")
        return None

    print("\n3️⃣  Asigna scores de relevancia (0-3) a cada ID:")
    print("    0 = No relevante")
    print("    1 = Algo relevante")
    print("    2 = Relevante")
    print("    3 = Muy relevante")

    relevance_scores = {}
    for chunk_id in relevant_ids:
        while True:
            try:
                score = int(input(f"   Score para ID {chunk_id}: ").strip())
                if 0 <= score <= 3:
                    relevance_scores[str(chunk_id)] = score
                    break
                else:
                    print("   ❌ Score debe estar entre 0 y 3")
            except ValueError:
                print("   ❌ Entrada inválida")

    return {
        "query": query_text,
        "relevant_chunks": relevant_ids,
        "relevance_scores": relevance_scores,
    }


def main():
    print("🔧 HERRAMIENTA DE ANOTACIÓN DE QUERIES")
    print("=" * 80)

    # Cargar chunks
    print("\n📦 Cargando chunks disponibles...")
    all_chunks = load_all_chunks()
    print(f"✓ Cargados {len(all_chunks)} chunks")

    # Cargar template de queries
    template_path = Path(__file__).parent / "queries_test_template.json"

    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        print(f"✓ Cargadas {len(queries)} queries del template")
    else:
        print("\n❌ No se encontró queries_test_template.json")
        print(
            "   Ejecuta primero: python evaluation/evaluate_retrieval_metrics.py --generate-queries"
        )
        return 1

    # Anotar queries
    annotated_queries = []

    for i, query_data in enumerate(queries, 1):
        print(f"\n{'#'*80}")
        print(f"QUERY {i}/{len(queries)}")
        print(f"{'#'*80}")

        annotated = annotate_query(query_data["query"], all_chunks)

        if annotated:
            annotated_queries.append(annotated)
            print(f"\n✅ Query anotada ({len(annotated['relevant_chunks'])} chunks relevantes)")

        # Opción para continuar
        if i < len(queries):
            print("\n¿Continuar con la siguiente query? (s/n, default=s)")
            choice = input("   > ").strip().lower()
            if choice == "n":
                break

    # Guardar queries anotadas
    output_path = Path(__file__).parent / "queries_test_annotated.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated_queries, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"✅ {len(annotated_queries)} queries anotadas guardadas en:")
    print(f"   {output_path}")
    print(f"{'='*80}")

    print("\n📝 Siguiente paso:")
    print(f"   python evaluation/evaluate_retrieval_metrics.py --test-queries {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

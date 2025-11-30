#!/usr/bin/env python
"""
Generar queries de prueba automáticamente usando los temas y documentos del grafo.
Crea anotaciones automáticas basadas en las relaciones grafo.

USO:
    python evaluation/generate_test_queries.py --output queries_test_auto.json --num-queries 20
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import yaml
from neo4j import GraphDatabase


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_mappings():
    """Cargar mappings de chunks"""
    config = load_config()
    base_path = Path(__file__).parent.parent
    mappings_path = base_path / config["paths"]["hnsw_mappings"]

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    return mappings["id_to_chunk"]


class QueryGenerator:
    """Generador de queries de prueba basadas en el grafo"""

    def __init__(self):
        config = load_config()
        neo4j_cfg = config["neo4j"]

        self.driver = GraphDatabase.driver(
            neo4j_cfg["uri"], auth=(neo4j_cfg["user"], neo4j_cfg["password"])
        )

        self.mappings = load_mappings()

    def get_topics_with_chunks(self):
        """Obtener temas con sus chunks asociados"""
        query = """
        MATCH (t:Topic)-[:TEACHES]->(c:Concept)
        MATCH (d:Document)-[:CONTAINS]->(chunk:Chunk)
        WHERE chunk.content CONTAINS t.name OR chunk.content CONTAINS c.name
        WITH t, collect(DISTINCT chunk) as chunks
        WHERE size(chunks) >= 3
        RETURN t.name as topic, 
               t.description as description,
               [chunk IN chunks | {
                   id: toInteger(chunk.id), 
                   content: chunk.content
               }] as chunks
        ORDER BY size(chunks) DESC
        LIMIT 10
        """

        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def get_document_chunks(self):
        """Obtener documentos con sus chunks"""
        query = """
        MATCH (d:Document)-[:CONTAINS]->(chunk:Chunk)
        WITH d, collect(DISTINCT chunk) as chunks
        WHERE size(chunks) >= 3
        RETURN d.file_name as document,
               [chunk IN chunks[0..10] | {
                   id: toInteger(chunk.id),
                   content: chunk.content
               }] as chunks
        ORDER BY size(chunks) DESC
        LIMIT 5
        """

        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def generate_topic_queries(self, num_queries=10):
        """Generar queries basadas en temas"""
        topics = self.get_topics_with_chunks()
        queries = []

        for topic_data in topics[:num_queries]:
            topic = topic_data["topic"]
            description = topic_data["description"] or f"información sobre {topic}"
            chunks = topic_data["chunks"]

            # Query simple
            query_text = f"¿Qué es {topic}?"

            # IDs de chunks relevantes (todos los chunks del tema)
            relevant_ids = [chunk["id"] for chunk in chunks[:10]]

            # Scores automáticos (basados en posición)
            relevance_scores = {}
            for i, chunk_id in enumerate(relevant_ids):
                if i < 3:
                    relevance_scores[str(chunk_id)] = 3  # Muy relevante
                elif i < 6:
                    relevance_scores[str(chunk_id)] = 2  # Relevante
                else:
                    relevance_scores[str(chunk_id)] = 1  # Algo relevante

            queries.append(
                {
                    "query": query_text,
                    "topic": topic,
                    "relevant_chunks": relevant_ids,
                    "relevance_scores": relevance_scores,
                    "source": "topic_based",
                }
            )

            # Query alternativa (más específica)
            if description and len(chunks) >= 5:
                alt_query = f"Explica {description}"
                queries.append(
                    {
                        "query": alt_query,
                        "topic": topic,
                        "relevant_chunks": relevant_ids[:8],
                        "relevance_scores": {
                            k: v for k, v in relevance_scores.items() if int(k) in relevant_ids[:8]
                        },
                        "source": "topic_based_detailed",
                    }
                )

        return queries

    def generate_document_queries(self, num_queries=5):
        """Generar queries basadas en documentos"""
        documents = self.get_document_chunks()
        queries = []

        for doc_data in documents[:num_queries]:
            doc_name = doc_data["document"]
            chunks = doc_data["chunks"]

            # Extraer tema del nombre del archivo
            topic_hint = doc_name.replace("_chunks.json", "").replace("_", " ")

            query_text = f"Resume los conceptos principales de {topic_hint}"

            # IDs de chunks relevantes (primeros chunks del documento)
            relevant_ids = [chunk["id"] for chunk in chunks[:10]]

            # Scores: primeros chunks más relevantes
            relevance_scores = {}
            for i, chunk_id in enumerate(relevant_ids):
                if i < 2:
                    relevance_scores[str(chunk_id)] = 3
                elif i < 5:
                    relevance_scores[str(chunk_id)] = 2
                else:
                    relevance_scores[str(chunk_id)] = 1

            queries.append(
                {
                    "query": query_text,
                    "document": doc_name,
                    "relevant_chunks": relevant_ids,
                    "relevance_scores": relevance_scores,
                    "source": "document_based",
                }
            )

        return queries

    def generate_mixed_queries(self):
        """Queries que combinan múltiples temas"""
        topics = self.get_topics_with_chunks()
        queries = []

        # Ejemplo: comparación entre dos temas
        if len(topics) >= 2:
            topic1 = topics[0]
            topic2 = topics[1]

            query_text = f"¿Cuál es la diferencia entre {topic1['topic']} y {topic2['topic']}?"

            # Chunks relevantes de ambos temas
            relevant_ids = [c["id"] for c in topic1["chunks"][:5]] + [
                c["id"] for c in topic2["chunks"][:5]
            ]

            relevance_scores = {str(cid): 2 for cid in relevant_ids}

            queries.append(
                {
                    "query": query_text,
                    "topics": [topic1["topic"], topic2["topic"]],
                    "relevant_chunks": relevant_ids,
                    "relevance_scores": relevance_scores,
                    "source": "multi_topic",
                }
            )

        return queries

    def close(self):
        self.driver.close()


def main():
    parser = argparse.ArgumentParser(description="Generar queries de prueba automáticamente")
    parser.add_argument(
        "--output",
        default="evaluation/queries_test_auto.json",
        help="Archivo de salida para queries generadas",
    )
    parser.add_argument(
        "--num-queries", type=int, default=20, help="Número total de queries a generar"
    )
    parser.add_argument("--include-mixed", action="store_true", help="Incluir queries multi-tema")

    args = parser.parse_args()

    print("🔧 GENERADOR AUTOMÁTICO DE QUERIES DE PRUEBA")
    print("=" * 80)

    generator = QueryGenerator()

    all_queries = []

    # Queries basadas en temas
    print("\n📚 Generando queries basadas en temas...")
    topic_queries = generator.generate_topic_queries(num_queries=args.num_queries // 2)
    all_queries.extend(topic_queries)
    print(f"✓ Generadas {len(topic_queries)} queries de temas")

    # Queries basadas en documentos
    print("\n📄 Generando queries basadas en documentos...")
    doc_queries = generator.generate_document_queries(num_queries=args.num_queries // 4)
    all_queries.extend(doc_queries)
    print(f"✓ Generadas {len(doc_queries)} queries de documentos")

    # Queries mixtas (opcional)
    if args.include_mixed:
        print("\n🔀 Generando queries multi-tema...")
        mixed_queries = generator.generate_mixed_queries()
        all_queries.extend(mixed_queries)
        print(f"✓ Generadas {len(mixed_queries)} queries mixtas")

    # Limitar al número solicitado
    all_queries = all_queries[: args.num_queries]

    # Guardar
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"✅ {len(all_queries)} queries generadas y guardadas en:")
    print(f"   {output_path}")
    print(f"{'='*80}")

    # Estadísticas
    if all_queries:
        print("\n📊 ESTADÍSTICAS:")
        sources = {}
        for q in all_queries:
            src = q.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        for source, count in sources.items():
            print(f"   {source}: {count} queries")

        avg_relevant = sum(len(q["relevant_chunks"]) for q in all_queries) / len(all_queries)
        print(f"   Promedio de chunks relevantes por query: {avg_relevant:.1f}")

        print("\n📝 Siguiente paso:")
        print(f"   python evaluation/evaluate_retrieval_metrics.py --test-queries {output_path}")
    else:
        print("\n⚠️  No se generaron queries. Revisa la estructura del grafo.")

    generator.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

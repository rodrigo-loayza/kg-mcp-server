"""
SCRIPT COMPLETO: EXTRACCIÓN AUTOMÁTICA DE CONTEXTO
Extrae contexto híbrido (vector + grafo) y lo formatea para LLM
"""

import json
import argparse
import hnswlib
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from pathlib import Path
import sys


class ContextExtractor:
    """
    Extractor de contexto completo para alimentar LLMs
    Soporta: RAG baseline, MCP híbrido, y multimodal
    """

    def __init__(
        self, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="12345678"
    ):

        print("🚀 Inicializando Context Extractor...")

        # Cargar modelo de embeddings
        print("  📦 Cargando modelo de embeddings...")
        self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

        # Cargar índice HNSW
        print("  📊 Cargando índice HNSW...")
        self.index = hnswlib.Index(space="cosine", dim=768)
        self.index.load_index("data/indices/hnsw_index.bin")

        # Cargar mapeos
        with open("data/indices/hnsw_index_mappings.json", "r", encoding="utf-8") as f:
            mappings_data = json.load(f)

            # Adaptar estructura: el JSON usa "id_to_chunk" y claves numéricas directas
            if "id_to_chunk" in mappings_data:
                # Crear mapeos compatibles con el código
                self.mappings = {
                    "index_to_chunk": {},  # idx -> chunk_id
                    "chunk_to_document": {},  # chunk_id -> document_id
                    "document_ids": set(),
                    "_chunk_data": {},  # Guardar datos completos para acceso posterior
                }

                for idx, chunk_data in mappings_data["id_to_chunk"].items():
                    chunk_id = idx  # El índice ES el chunk_id
                    doc_id = chunk_data["source"]

                    self.mappings["index_to_chunk"][idx] = chunk_id
                    self.mappings["chunk_to_document"][chunk_id] = doc_id
                    self.mappings["document_ids"].add(doc_id)
                    self.mappings["_chunk_data"][chunk_id] = chunk_data  # Guardar datos completos

                self.mappings["document_ids"] = list(self.mappings["document_ids"])
            else:
                # Estructura antigua (por compatibilidad)
                self.mappings = mappings_data

        # Conectar a Neo4j
        print("  🗄️  Conectando a Neo4j...")
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        print("✅ Inicialización completa\n")

    def extract_rag_baseline(self, query, k=10):
        """
        Extrae contexto usando solo búsqueda vectorial (RAG tradicional)
        """
        print(f"📊 Extrayendo contexto RAG baseline para: '{query}'\n")

        # Búsqueda vectorial
        query_embedding = self.model.encode(query)
        labels, distances = self.index.knn_query(query_embedding, k=k)

        results = []

        for rank, (idx, dist) in enumerate(zip(labels[0], distances[0]), 1):
            chunk_id = self.mappings["index_to_chunk"][str(idx)]
            doc_id = self.mappings["chunk_to_document"][chunk_id]
            similarity = 1 - dist

            # Cargar texto del chunk
            chunk_text = self._load_chunk_text(doc_id, chunk_id)

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "document": doc_id,
                    "similarity": float(similarity),
                    "text": chunk_text,
                    "type": "rag_baseline",
                }
            )

            print(f"  [{rank}] {doc_id} (sim: {similarity:.4f})")

        print(f"\n✅ {len(results)} chunks recuperados (RAG baseline)\n")
        return results

    def extract_hybrid(self, query, k=10, vector_weight=0.7, graph_weight=0.3):
        """
        Extrae contexto híbrido (vector + grafo)
        Estrategia adaptativa: combina búsqueda vectorial con expansión conceptual
        vía relaciones MENTIONS del grafo de conocimiento académico.
        """
        print(f"🔗 Extrayendo contexto híbrido para: '{query}'\n")

        # 1. Búsqueda vectorial (top-30 para re-ranking)
        query_embedding = self.model.encode(query)
        labels, distances = self.index.knn_query(query_embedding, k=30)

        print(f"  📊 Búsqueda vectorial: {len(labels[0])} chunks iniciales")

        # 2. Extraer contexto de grafo
        chunk_ids = [self.mappings["index_to_chunk"][str(idx)] for idx in labels[0]]
        graph_context = self._extract_graph_context(chunk_ids)

        print(f"  🕸️  Contexto de grafo extraído para {len(graph_context)} documentos")

        # Crear mapping de file_name -> graph_context para lookup rápido
        graph_by_file = {g["file_name"]: g for g in graph_context}

        # Contar chunks con MENTIONS para pesos adaptativos
        chunks_with_concepts = sum(1 for g in graph_context if g["concepts"])
        total_chunks = len(graph_context)
        mention_coverage = chunks_with_concepts / total_chunks if total_chunks > 0 else 0

        # Ajuste adaptativo: si cobertura baja, reducir peso del grafo
        # Esto evita penalizar chunks sin MENTIONS
        adaptive_graph_weight = graph_weight if mention_coverage > 0.3 else graph_weight * 0.5
        adaptive_vector_weight = 1.0 - adaptive_graph_weight

        # 3. Calcular scores híbridos
        hybrid_results = []

        for idx, dist in zip(labels[0], distances[0]):
            chunk_id = self.mappings["index_to_chunk"][str(idx)]
            doc_id = self.mappings["chunk_to_document"][chunk_id]
            vector_score = 1 - dist

            # Buscar contexto de grafo por nombre de archivo
            g_ctx = graph_by_file.get(doc_id, None)

            # Calcular graph score SOLO si hay conceptos (evitar penalización)
            if g_ctx and g_ctx["concepts"]:
                num_concepts = len(g_ctx["concepts"])
                num_related = len(g_ctx["related_concepts"])

                # Métricas de riqueza conceptual
                concept_richness = min(num_concepts / 3.0, 1.0)  # Más sensible
                relationship_density = min(num_related / 15.0, 1.0)  # Más sensible

                # Similitud semántica de conceptos con query
                concept_similarity = self._calculate_concept_similarity(
                    g_ctx["concepts"], query_embedding
                )

                # Pesos: énfasis en similitud semántica (80%) y riqueza (20%)
                graph_score = (
                    0.15 * concept_richness
                    + 0.75 * concept_similarity
                    + 0.10 * relationship_density
                )

                # Boost adicional sutil para chunks con alta relevancia conceptual
                if concept_similarity > 0.6 and num_concepts >= 2:
                    graph_score = min(graph_score * 1.15, 1.0)
            else:
                # Sin conceptos: usar solo vector (sin penalización)
                graph_score = 0.0

            # Score híbrido con pesos adaptativos
            if graph_score > 0:
                # Chunk con conceptos: usar combinación ponderada + boost multiplicativo estratégico
                base_hybrid = (
                    adaptive_vector_weight * vector_score + adaptive_graph_weight * graph_score
                )
                # Boost progresivo basado en calidad conceptual y vectorial
                if graph_score > 0.6 and vector_score > 0.45:
                    # Muy alta relevancia conceptual + buena similitud vectorial
                    hybrid_score = min(base_hybrid * 1.22, 1.0)
                elif graph_score > 0.5 and vector_score > 0.4:
                    # Alta relevancia conceptual
                    hybrid_score = min(base_hybrid * 1.15, 1.0)
                elif graph_score > 0.4:
                    # Relevancia conceptual moderada
                    hybrid_score = min(base_hybrid * 1.08, 1.0)
                else:
                    hybrid_score = base_hybrid
            else:
                # Chunk sin conceptos: usar solo vector (100%)
                hybrid_score = vector_score

            # Cargar texto
            chunk_text = self._load_chunk_text(doc_id, chunk_id)

            hybrid_results.append(
                {
                    "chunk_id": chunk_id,
                    "document": doc_id,
                    "text": chunk_text,
                    "vector_score": float(vector_score),
                    "graph_score": float(graph_score),
                    "hybrid_score": float(hybrid_score),
                    "concepts": g_ctx["concepts"] if g_ctx else [],
                    "related_concepts": g_ctx["related_concepts"] if g_ctx else [],
                    "topics": g_ctx.get("topics", []) if g_ctx else [],
                    "cso_labels": g_ctx.get("cso_labels", []) if g_ctx else [],
                    "cso_uris": g_ctx.get("cso_uris", []) if g_ctx else [],
                    "type": "hybrid",
                }
            )

        # Re-rankear por score híbrido
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Tomar top-k
        hybrid_results = hybrid_results[:k]

        # Agregar ranks
        for rank, item in enumerate(hybrid_results, 1):
            item["rank"] = rank
            print(
                f"  [{rank}] {item['document']} (hybrid: {item['hybrid_score']:.4f}, "
                f"vector: {item['vector_score']:.4f}, graph: {item['graph_score']:.4f})"
            )

        print(f"\n✅ {len(hybrid_results)} chunks recuperados (Híbrido)\n")
        return hybrid_results

    def _extract_graph_context(self, chunk_ids):
        """Extrae contexto del grafo para los chunks dados

        Nueva Estrategia (con MENTIONS):
        1. Para cada chunk, obtiene sus conceptos específicos vía relación MENTIONS
        2. Para esos conceptos, obtiene conceptos relacionados y topics
        3. Retorna contexto ESPECÍFICO por chunk (no global)

        Estructura del grafo:
        Chunk -[MENTIONS]-> Concept -[RELATED_TO]-> Concept
        Course -[TEACHES]-> Topic -[PREREQUISITE_OF]-> Concept/Algorithm
        """

        with self.driver.session() as session:
            result = []

            # Procesar cada chunk individualmente para obtener su contexto específico
            for chunk_id in chunk_ids:
                # Buscar chunk por ID numérico
                chunk_result = session.run(
                    """
                    MATCH (ch:Chunk {id: $chunk_id})
                    OPTIONAL MATCH (ch)-[:MENTIONS]->(c:Concept)
                    OPTIONAL MATCH (c)-[:RELATED_TO]-(related:Concept)
                    OPTIONAL MATCH (c)<-[:PREREQUISITE_OF]-(t:Topic)
                    OPTIONAL MATCH (c)<-[:PREREQUISITE_OF]-(a:Algorithm)
                    RETURN 
                        ch.source as file_name,
                        collect(DISTINCT c.name) as chunk_concepts,
                        collect(DISTINCT related.name) as related_concepts,
                        collect(DISTINCT t.name) as topics,
                        collect(DISTINCT t.cso_label) as cso_labels,
                        collect(DISTINCT t.cso_uri) as cso_uris,
                        collect(DISTINCT a.name) as algorithms
                    """,
                    chunk_id=chunk_id,
                )

                chunk_ctx = chunk_result.single()
                if chunk_ctx and chunk_ctx["file_name"]:
                    ctx = {
                        "file_name": chunk_ctx["file_name"],
                        "concepts": [c for c in chunk_ctx["chunk_concepts"] if c],
                        "algorithms": [a for a in chunk_ctx["algorithms"] if a],
                        "related_concepts": [r for r in chunk_ctx["related_concepts"] if r],
                        "topics": [t for t in chunk_ctx["topics"] if t],
                        "cso_labels": [l for l in chunk_ctx["cso_labels"] if l],
                        "cso_uris": [u for u in chunk_ctx["cso_uris"] if u],
                    }
                    result.append(ctx)
                else:
                    # Fallback: si el chunk no está en Neo4j, usar contexto mínimo
                    chunk_data = self.mappings.get("_chunk_data", {}).get(chunk_id, {})
                    fname = chunk_data.get("source", "unknown")
                    result.append(
                        {
                            "file_name": fname,
                            "concepts": [],
                            "algorithms": [],
                            "related_concepts": [],
                            "topics": [],
                            "cso_labels": [],
                            "cso_uris": [],
                        }
                    )

            return result

    def _calculate_concept_similarity(self, concepts, query_embedding):
        """
        Calcula similitud entre conceptos y query
        Usa promedio de top-3 similitudes en vez de solo máximo para capturar más señal
        """

        if not concepts:
            return 0.0

        # Embeddings de conceptos
        concept_embeddings = self.model.encode(concepts)

        # Similitud coseno
        from numpy import dot
        from numpy.linalg import norm

        similarities = [
            dot(query_embedding, concept_emb) / (norm(query_embedding) * norm(concept_emb))
            for concept_emb in concept_embeddings
        ]

        if not similarities:
            return 0.0

        # Ordenar y tomar promedio de top-3 (o todos si hay menos de 3)
        sorted_sims = sorted(similarities, reverse=True)
        top_k = min(3, len(sorted_sims))
        avg_top_similarity = sum(sorted_sims[:top_k]) / top_k

        return float(avg_top_similarity)

    def _load_chunk_text(self, doc_id, chunk_id):
        """Carga el texto de un chunk específico"""

        # Primero intentar cargar del mapeo original
        with open("data/indices/hnsw_index_mappings.json", "r", encoding="utf-8") as f:
            mappings_data = json.load(f)

        if "id_to_chunk" in mappings_data:
            # Estructura nueva: el texto está directamente en el mapeo
            chunk_data = mappings_data["id_to_chunk"].get(str(chunk_id))
            if chunk_data and "content" in chunk_data:
                return chunk_data["content"]

        # Fallback: buscar en archivos procesados
        doc_name = doc_id.replace(".pdf", "")
        chunk_path = f"data/processed/{doc_name}_chunks.json"

        if not Path(chunk_path).exists():
            return f"[Texto no disponible para {doc_id}]"

        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            if str(chunk.get("chunk_id")) == str(chunk_id):
                return chunk.get("text", chunk.get("content", ""))

        return f"[Chunk {chunk_id} no encontrado]"

    def format_for_llm(self, context, query, system="hybrid"):
        """
        Formatea el contexto para insertar en un LLM
        Versión mejorada que aprovecha el grafo de conocimiento CSO
        """

        # Extraer conocimiento del grafo para contexto adicional
        if system == "hybrid":
            all_concepts = set()
            all_topics = set()
            all_cso_labels = set()
            concept_freq = {}

            for item in context:
                for concept in item.get("concepts", []):
                    all_concepts.add(concept)
                    concept_freq[concept] = concept_freq.get(concept, 0) + 1
                for topic in item.get("topics", []):
                    if topic:
                        all_topics.add(topic)
                for cso_label in item.get("cso_labels", []):
                    if cso_label:
                        all_cso_labels.add(cso_label)

            # Conceptos más frecuentes (aparecen en múltiples documentos)
            key_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            key_concept_names = [c[0] for c in key_concepts]

        formatted = f"""# PROMPT PARA LLM - SISTEMA {system.upper()}

## PREGUNTA DEL USUARIO
{query}
"""

        if system == "hybrid":
            formatted += f"""
## MAPA DE CONOCIMIENTO (Computer Science Ontology)

### Contexto Conceptual
Este sistema ha identificado los siguientes elementos del grafo de conocimiento académico que son relevantes para tu pregunta:

"""
            if all_topics:
                formatted += f"**Áreas Temáticas**: {', '.join(sorted(all_topics))}\n"

            if all_cso_labels:
                formatted += f"**Conceptos CSO**: {', '.join(sorted(all_cso_labels))}\n"

            if key_concept_names:
                formatted += f"\n**Conceptos Clave** (presentes en múltiples fuentes):\n"
                for concept, freq in key_concepts[:5]:
                    formatted += (
                        f"  • {concept} (mencionado en {freq} documento{'s' if freq > 1 else ''})\n"
                    )

            formatted += f"\n**Red de Conocimiento**: {len(all_concepts)} conceptos interrelacionados identificados\n"

            formatted += """
### Guía de Interpretación
Los fragmentos a continuación están enriquecidos con información del Computer Science Ontology (CSO).
Esta ontología proporciona contexto sobre las relaciones entre conceptos, permitiendo una comprensión
más profunda de cómo los temas se conectan en el dominio de Ciencias de la Computación.

"""

        formatted += "## FRAGMENTOS DE CONOCIMIENTO\n"

        for item in context:
            formatted += f"""
{'='*80}
### Fragmento {item['rank']}: {item['document']}
"""

            if system == "hybrid":
                formatted += f"""
**Relevancia**: Score híbrido {item['hybrid_score']:.4f} 
  ↳ Similitud vectorial: {item['vector_score']:.4f}
  ↳ Enriquecimiento grafo: {item['graph_score']:.4f}
"""

                # Metadata del grafo
                if item.get("topics"):
                    topics_list = [t for t in item["topics"] if t]
                    if topics_list:
                        formatted += f"\n**Clasificación Temática**: {', '.join(topics_list[:5])}\n"

                if item.get("cso_labels"):
                    cso_list = [l for l in item["cso_labels"] if l]
                    if cso_list:
                        formatted += f"**Ontología CSO**: {', '.join(cso_list[:5])}\n"

                if item.get("concepts"):
                    # Separar conceptos clave vs otros
                    concepts_here = item["concepts"][:15]
                    key_here = [c for c in concepts_here if c in key_concept_names]
                    other_here = [c for c in concepts_here if c not in key_concept_names]

                    if key_here:
                        formatted += f"\n**Conceptos Principales**: {', '.join(key_here)}\n"
                    if other_here:
                        formatted += f"**Conceptos Adicionales**: {', '.join(other_here[:10])}\n"

                if item.get("related_concepts"):
                    num_related = len(item["related_concepts"])
                    formatted += (
                        f"**Red Conceptual**: {num_related} conceptos relacionados en el grafo\n"
                    )

                formatted += f"\n---\n"
            else:
                formatted += f"**Similitud**: {item['similarity']:.4f}\n\n---\n"

            formatted += f"\n{item['text']}\n"

        formatted += f"""
{'='*80}

## INSTRUCCIONES PARA RESPONDER

### Contexto de la Respuesta
"""

        if system == "hybrid":
            formatted += f"""Has recibido {len(context)} fragmentos de conocimiento enriquecidos con el Computer Science Ontology (CSO).
Esta ontología conecta conceptos académicos en Ciencias de la Computación, permitiéndote entender
no solo el contenido textual, sino también las relaciones conceptuales entre temas.

"""
            if all_topics:
                formatted += f"**Áreas temáticas identificadas**: {', '.join(sorted(all_topics))}\n"

            if key_concept_names:
                formatted += f"**Conceptos centrales**: {', '.join(key_concept_names[:5])}\n\n"

        formatted += """### Directrices de Respuesta

1. **Fundamentación**: Responde ÚNICAMENTE basándote en el conocimiento proporcionado en los fragmentos
2. **Honestidad**: Si la información es insuficiente, indícalo claramente sin inventar
3. **Síntesis Natural**: Integra la información de forma fluida, sin citar explícitamente "fragmento X" o "según el material"
"""

        if system == "hybrid":
            formatted += """4. **Aprovecha el Grafo**: Usa las relaciones conceptuales y la jerarquía temática para:
   - Conectar ideas entre fragmentos usando los conceptos compartidos
   - Proporcionar contexto adicional basado en las áreas temáticas identificadas
   - Explicar relaciones entre conceptos cuando sea relevante para la pregunta
5. **Rigor Académico**: Mantén precisión técnica y coherencia con la terminología CSO
6. **Concisión Estructurada**: Organiza la respuesta de forma clara, usando los conceptos clave como guía

"""
        else:
            formatted += """4. **Rigor Académico**: Mantén precisión técnica y coherencia terminológica
5. **Concisión**: Sé claro y directo

"""

        formatted += """### Tu Respuesta:
"""

        return formatted

    def save_context(self, context, output_path, formatted_prompt=None):
        """Guarda el contexto extraído"""

        # Guardar JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2, ensure_ascii=False)

        print(f"💾 Contexto guardado: {output_path}")

        # Guardar prompt formateado si se proporciona
        if formatted_prompt:
            prompt_path = output_path.replace(".json", "_prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(formatted_prompt)

            print(f"💾 Prompt formateado: {prompt_path}")

    def close(self):
        """Cierra conexiones"""
        self.driver.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extractor de Contexto para LLM (RAG baseline o Híbrido)"
    )

    parser.add_argument(
        "--query", type=str, required=True, help="Pregunta para la cual extraer contexto"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["rag", "hybrid"],
        help="Modo de extracción: rag (solo vector) o hybrid (vector+grafo)",
    )

    parser.add_argument("--k", type=int, default=10, help="Número de chunks a recuperar")

    parser.add_argument(
        "--output",
        type=str,
        default="contexto_extraido.json",
        help="Archivo de salida para el contexto",
    )

    parser.add_argument(
        "--format-prompt", action="store_true", help="Generar también prompt formateado para LLM"
    )

    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.8,
        help="Peso del score vectorial (solo modo hybrid)",
    )

    parser.add_argument(
        "--graph-weight", type=float, default=0.2, help="Peso del score de grafo (solo modo hybrid)"
    )

    args = parser.parse_args()

    # Crear extractor
    extractor = ContextExtractor()

    try:
        # Extraer contexto según modo
        if args.mode == "rag":
            context = extractor.extract_rag_baseline(args.query, k=args.k)
        else:
            context = extractor.extract_hybrid(
                args.query,
                k=args.k,
                vector_weight=args.vector_weight,
                graph_weight=args.graph_weight,
            )

        # Formatear prompt si se solicita
        formatted_prompt = None
        if args.format_prompt:
            formatted_prompt = extractor.format_for_llm(context, args.query, args.mode)

        # Guardar
        extractor.save_context(context, args.output, formatted_prompt)

        print("\n✅ Extracción completa!")

        if args.format_prompt:
            print(
                f"\n📝 Puedes copiar el contenido de '{args.output.replace('.json', '_prompt.txt')}' "
                f"y pegarlo directamente en Claude, GPT-4, u otro LLM\n"
            )

    finally:
        extractor.close()


if __name__ == "__main__":
    # Si se ejecuta sin argumentos, mostrar ejemplo
    if len(sys.argv) == 1:
        print(
            """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EXTRACTOR DE CONTEXTO PARA LLM                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

EJEMPLOS DE USO:

1. RAG Baseline (solo búsqueda vectorial):
   python extract_context.py --query "¿Qué es aprendizaje supervisado?" --mode rag --k 10 --format-prompt

2. Sistema Híbrido (vector + grafo):
   python extract_context.py --query "Explica algoritmos genéticos" --mode hybrid --k 10 --format-prompt

3. Personalizar pesos híbridos:
   python extract_context.py --query "¿Qué es K-means?" --mode hybrid --vector-weight 0.7 --graph-weight 0.3

4. Guardar en ubicación específica:
   python extract_context.py --query "Redes neuronales" --output resultados/contexto_nn.json

SALIDAS:
- <output>.json          : Contexto estructurado (JSON)
- <output>_prompt.txt    : Prompt listo para copiar al LLM (si --format-prompt)

Para más ayuda: python extract_context.py --help
"""
        )
        sys.exit(0)

    main()

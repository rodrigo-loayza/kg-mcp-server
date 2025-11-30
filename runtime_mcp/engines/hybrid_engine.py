# hybrid_engine.py
"""
🔍 Motor de Recuperación Híbrida (Vector + Graph)
Combina búsqueda vectorial HNSW con traversal del grafo de conocimiento Neo4j

VERSIÓN CORREGIDA - Compatible con estructura real de chunks

Responsabilidades:
- Búsqueda vectorial eficiente (HNSW)
- Expansión de contexto con grafo (Neo4j)
- Fusion scoring
- Enriquecimiento académico

Autor: Rodrigo Cárdenas
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import hnswlib
from neo4j import GraphDatabase


@dataclass
class RetrievalResult:
    """Resultado de búsqueda enriquecido con contexto del grafo"""

    chunk_id: str
    content: str
    relevance_score: float  # Renombrado de vector_score para consistencia
    doc_id: str
    enriched_context: Dict
    metadata: Dict


class HybridEngine:
    """Motor híbrido Vector + Graph para recuperación académica"""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        dimension: int = 768,
        preloaded_model=None,
    ):
        """
        Inicializa motor híbrido con soporte multilingüe

        Args:
            neo4j_uri: URI de Neo4j
            neo4j_user: Usuario Neo4j
            neo4j_password: Password Neo4j
            embedding_model: Modelo embeddings (default: multilingüe español-inglés)
            dimension: 768 para modelo multilingüe, 384 para modelos simples
            preloaded_model: Modelo SentenceTransformer ya cargado (opcional)
        """
        # Conexión Neo4j
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Búsqueda vectorial (lazy loading o pre-cargado)
        self.embedding_model_name = embedding_model
        self.model = preloaded_model  # Usar modelo pre-cargado si se proporciona
        self.dimension = dimension
        self.index = hnswlib.Index(space="cosine", dim=dimension)
        self.index_loaded = False

        # Mappings
        self.id_to_chunk = {}
        self.chunk_counter = 0

    def _ensure_model_loaded(self):
        """Carga el modelo de embeddings si no está cargado (lazy loading)"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.embedding_model_name)

    def close(self):
        """Cierra conexión a Neo4j"""
        self.driver.close()

    def load_index(self, index_path: Path):
        """
        Carga índice HNSW pre-construido

        Args:
            index_path: Ruta al archivo hnsw_index.bin
        """
        print(f"\n📥 Cargando índice HNSW desde: {index_path}")

        # Cargar índice
        self.index.load_index(str(index_path))

        # Cargar mappings
        mapping_file = index_path.parent / f"{index_path.stem}_mappings.json"

        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.id_to_chunk = {int(k): v for k, v in data["id_to_chunk"].items()}
            self.chunk_counter = data["chunk_counter"]

        # Configurar ef de búsqueda
        self.index.set_ef(50)
        self.index_loaded = True

        print(f"✅ Índice cargado: {self.chunk_counter} chunks")

    def vector_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Búsqueda vectorial pura en HNSW

        Args:
            query: Texto de consulta
            k: Número de resultados

        Returns:
            Lista de dicts con chunk_id, content, score, doc_id, metadata
        """
        if not self.index_loaded:
            print("❌ Índice no cargado")
            return []

        # Cargar modelo si no está cargado (lazy loading)
        self._ensure_model_loaded()

        # Generar embedding de query
        query_embedding = self.model.encode([query])[0]

        # Buscar en HNSW
        labels, distances = self.index.knn_query(query_embedding, k=k)

        # Construir resultados con estructura real de chunks
        results = []
        for label, distance in zip(labels[0], distances[0]):
            chunk = self.id_to_chunk[label]

            # Usar el label HNSW como chunk_id (int)
            chunk_id = str(label)  # Convertir a string para compatibilidad con JSON

            # Generar doc_id basado en el source
            source = chunk.get("source", chunk.get("file_name", "unknown"))
            doc_id = source

            results.append(
                {
                    "chunk_id": chunk_id,
                    "content": chunk.get("content", ""),
                    "score": float(1 - distance),  # Convertir distancia a similitud
                    "doc_id": doc_id,
                    "metadata": {
                        "file_name": chunk.get("file_name", ""),
                        "file_path": chunk.get("file_path", ""),
                        "relative_path": chunk.get("relative_path", ""),
                        "course_code": chunk.get("course_code", ""),
                        "start_word": chunk.get("start_word", 0),
                        "end_word": chunk.get("end_word", 0),
                        "total_words": chunk.get("total_words", 0),
                        "file_type": chunk.get("file_type", ""),
                    },
                }
            )

        return results

    def graph_expand(self, doc_id: str, max_hops: int = 2) -> Dict:
        """
        Expande contexto usando el grafo de conocimiento

        Args:
            doc_id: ID del documento (source/file_name)
            max_hops: Profundidad máxima de traversal

        Returns:
            Dict con información enriquecida del grafo
        """
        with self.driver.session() as session:
            # Query Cypher para expandir desde Document
            # Usar name o path en lugar de id
            query = """
            MATCH (d:Document)
            WHERE d.name CONTAINS $doc_id OR d.path CONTAINS $doc_id
            OPTIONAL MATCH (d)-[:CONTAINS]->(c:Concept)
            OPTIONAL MATCH (c)-[:RELATED_TO]-(rc:Concept)
            OPTIONAL MATCH (d)<-[:TEACHES]-(course:Course)
            OPTIONAL MATCH (course)-[:TEACHES]->(t:Topic)
            OPTIONAL MATCH (d)-[:CITES]->(r:Reference)
            
            RETURN 
                d as document,
                collect(DISTINCT c) as concepts,
                collect(DISTINCT rc) as related_concepts,
                collect(DISTINCT course) as courses,
                collect(DISTINCT t) as topics,
                collect(DISTINCT r) as references
            LIMIT 1
            """

            result = session.run(query, doc_id=doc_id)
            record = result.single()

            if not record:
                return {}

            # Estructurar contexto enriquecido
            enriched = {
                "document": self._node_to_dict(record["document"]),
                "concepts": [self._node_to_dict(n) for n in record["concepts"] if n],
                "related_concepts": [
                    self._node_to_dict(n) for n in record["related_concepts"] if n
                ],
                "courses": [self._node_to_dict(n) for n in record["courses"] if n],
                "topics": [self._node_to_dict(n) for n in record["topics"] if n],
                "references": [self._node_to_dict(n) for n in record["references"] if n],
            }

            return enriched

    def _node_to_dict(self, node) -> Dict:
        """Convierte nodo de Neo4j a diccionario"""
        if node is None:
            return {}
        return dict(node)

    def _find_documents_by_concepts(self, concept_names: List[str], limit: int = 3) -> List[str]:
        """
        Encuentra documentos que contienen los conceptos especificados

        Args:
            concept_names: Lista de nombres de conceptos
            limit: Máximo número de documentos a retornar

        Returns:
            Lista de nombres de documentos
        """
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $concepts as concept_name
                MATCH (d:Document)-[:CONTAINS]->(c:Concept)
                WHERE c.name = concept_name
                RETURN DISTINCT d.name as doc_name
                LIMIT $limit
            """,
                concepts=concept_names,
                limit=limit,
            )

            return [record["doc_name"] for record in result]

    def _get_chunks_from_document(self, doc_name: str, limit: int = 2) -> List[str]:
        """
        Obtiene IDs de chunks de un documento específico

        Args:
            doc_name: Nombre del documento
            limit: Máximo número de chunks a retornar

        Returns:
            Lista de chunk IDs (como strings)
        """
        chunk_ids = []
        for chunk_id, chunk_data in self.id_to_chunk.items():
            source = chunk_data.get("source", chunk_data.get("file_name", ""))
            if source == doc_name:
                chunk_ids.append(str(chunk_id))
                if len(chunk_ids) >= limit:
                    break
        return chunk_ids

    def hybrid_search(
        self, query: str, k: int = 5, expand_graph: bool = True
    ) -> List[RetrievalResult]:
        """
        Búsqueda híbrida: Vector (HNSW) + Graph expansion

        Args:
            query: Texto de consulta
            k: Número de resultados finales
            expand_graph: Si True, expande con conceptos del grafo

        Returns:
            Lista de Retrieval Result con contexto enriquecido y expandido
        """
        # 1. Búsqueda vectorial completa
        vector_results = self.vector_search(query, k=k * 3)

        if not expand_graph:
            return [
                RetrievalResult(
                    chunk_id=vr["chunk_id"],
                    content=vr["content"],
                    relevance_score=vr["score"],
                    doc_id=vr["doc_id"],
                    enriched_context={},
                    metadata=vr["metadata"],
                )
                for vr in vector_results[:k]
            ]

        # 2. Re-ranking con graph scoring
        query_embedding = self.model.encode([query])[0]
        scored_results = []

        for vr in vector_results:
            # Enriquecer con contexto del grafo
            enriched_context = self.graph_expand(vr["doc_id"], max_hops=2)

            # Calcular graph score
            graph_score = self._calculate_graph_score(enriched_context, query_embedding)

            # Combinar scores: 80% vector + 20% graph
            combined_score = 0.80 * vr["score"] + 0.20 * graph_score

            scored_results.append(
                (
                    combined_score,
                    RetrievalResult(
                        chunk_id=vr["chunk_id"],
                        content=vr["content"],
                        relevance_score=combined_score,
                        doc_id=vr["doc_id"],
                        enriched_context=enriched_context,
                        metadata=vr["metadata"],
                    ),
                )
            )

        # 3. Ordenar por combined score y tomar top-k
        scored_results.sort(key=lambda x: x[0], reverse=True)

        return [result for _, result in scored_results[:k]]

    def _rerank_with_graph(
        self, results: List[RetrievalResult], graph_contexts: List[Dict], query: str
    ) -> List[RetrievalResult]:
        """
        Re-rankea resultados usando información del grafo de conocimiento

        Args:
            results: Lista de resultados con scores vectoriales
            graph_contexts: Contextos del grafo para cada resultado
            query: Query original

        Returns:
            Lista de resultados re-rankeada
        """
        # Cargar modelo si no está cargado
        self._ensure_model_loaded()

        # Generar embedding de la query
        query_embedding = self.model.encode([query])[0]

        # Calcular scores combinados
        scored_results = []

        for result, graph_ctx in zip(results, graph_contexts):
            vector_score = result.relevance_score

            # Calcular graph score basado en riqueza del contexto
            graph_score = self._calculate_graph_score(graph_ctx, query_embedding)

            # Combinar scores: 85% vector + 15% graph (priorizar similitud vectorial)
            combined_score = 0.85 * vector_score + 0.15 * graph_score

            # Crear nuevo resultado con score combinado
            scored_results.append((combined_score, result))

        # Ordenar por score combinado (descendente)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Actualizar relevance_score con el score combinado
        reranked = []
        for combined_score, result in scored_results:
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    relevance_score=combined_score,
                    doc_id=result.doc_id,
                    enriched_context=result.enriched_context,
                    metadata=result.metadata,
                )
            )

        return reranked

    def _calculate_graph_score(self, graph_context: Dict, query_embedding) -> float:
        """
        Calcula score basado en la riqueza del contexto del grafo

        Args:
            graph_context: Diccionario con información del grafo
            query_embedding: Embedding de la query para comparación semántica

        Returns:
            Score normalizado [0, 1]
        """
        score = 0.0

        # 1. Bonus por conceptos (más conceptos = mayor riqueza semántica)
        concepts = graph_context.get("concepts", [])
        num_concepts = len([c for c in concepts if c])
        if num_concepts > 0:
            # Normalizar: hasta 8 conceptos = +0.5
            score += min(num_concepts / 8.0, 1.0) * 0.5

            # Bonus MAYOR si hay match semántico con concepts
            if concepts:
                concept_texts = [c.get("name", "") for c in concepts if c and c.get("name")]
                if concept_texts:
                    concept_embeddings = self.model.encode(concept_texts)
                    # Similitud máxima con algún concepto
                    from numpy import dot
                    from numpy.linalg import norm

                    similarities = [
                        dot(query_embedding, c_emb) / (norm(query_embedding) * norm(c_emb))
                        for c_emb in concept_embeddings
                    ]
                    max_sim = max(similarities) if similarities else 0.0
                    # Aumentar peso si hay match semántico fuerte
                    score += max_sim * 0.4

        # 2. Bonus MENOR por conceptos relacionados (breadth, menos importante)
        related_concepts = graph_context.get("related_concepts", [])
        num_related = len([c for c in related_concepts if c])
        if num_related > 0:
            # Reducir impacto: hasta 20 relacionados = +0.1
            score += min(num_related / 20.0, 1.0) * 0.1

        # 3. Bonus por topics (estructura curricular)
        topics = graph_context.get("topics", [])
        if len(topics) > 0:
            score += min(len(topics) / 3.0, 1.0) * 0.15

        # 4. Bonus por cursos (contexto académico)
        courses = graph_context.get("courses", [])
        if len(courses) > 0:
            score += 0.15

        # Normalizar a [0, 1]
        return min(score, 1.0)

    def format_for_llm(self, results: List[RetrievalResult]) -> str:
        """
        Formatea resultados para prompt de LLM

        Args:
            results: Lista de RetrievalResult

        Returns:
            String formateado para contexto del LLM
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            # Chunk principal
            context_parts.append(f"[DOCUMENTO {i}]")
            context_parts.append(f"Fuente: {result.metadata.get('file_name', 'N/A')}")
            context_parts.append(f"Score: {result.relevance_score:.4f}")
            context_parts.append(result.content)

            # Contexto enriquecido del grafo
            enriched = result.enriched_context

            if enriched.get("courses"):
                course = enriched["courses"][0]
                context_parts.append(f"\n[CONTEXTO] Curso: {course.get('name', 'N/A')}")

            if enriched.get("topics"):
                topics = [t.get("name", "") for t in enriched["topics"][:3]]
                if topics:
                    context_parts.append(f"[CONTEXTO] Temas relacionados: {', '.join(topics)}")

            if enriched.get("concepts"):
                concepts = [c.get("name", "") for c in enriched["concepts"][:5]]
                if concepts:
                    context_parts.append(f"[CONTEXTO] Conceptos: {', '.join(concepts)}")

            if enriched.get("references"):
                refs = [
                    f"{r.get('authors', 'N/A')} ({r.get('year', 'N/A')})"
                    for r in enriched["references"][:2]
                ]
                if refs:
                    context_parts.append(f"[CONTEXTO] Referencias: {', '.join(refs)}")

            context_parts.append("\n" + "-" * 50 + "\n")

        return "\n".join(context_parts)


def demo():
    """Demo del motor híbrido"""
    print("\n" + "=" * 70)
    print("🔍 DEMO: MOTOR DE RECUPERACIÓN HÍBRIDA")
    print("=" * 70)

    # Inicializar engine
    engine = HybridEngine(
        neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="12345678"
    )

    try:
        # Cargar índice pre-construido
        engine.load_index(Path("data/indices/hnsw_index.bin"))

        # Consultas de prueba
        queries = [
            "¿Cómo funciona PSO?",
            "¿Qué es la velocidad de partícula?",
            "Explica algoritmos genéticos",
        ]

        for query in queries:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}\n")

            # Búsqueda híbrida
            results = engine.hybrid_search(query, k=3)

            # Formatear para LLM
            llm_context = engine.format_for_llm(results)
            print(llm_context)

            # Estadísticas de enriquecimiento
            print("\n📊 Enriquecimiento:")
            for i, r in enumerate(results, 1):
                ec = r.enriched_context
                print(f"   Resultado {i}:")
                print(f"      Conceptos: {len(ec.get('concepts', []))}")
                print(f"      Temas: {len(ec.get('topics', []))}")
                print(f"      Referencias: {len(ec.get('references', []))}")

    finally:
        engine.close()


if __name__ == "__main__":
    demo()

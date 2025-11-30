# context_enricher.py
"""
📚 Context Enricher - Enriquecimiento de Contexto Académico
Capa adicional sobre HybridEngine para enriquecimiento específico académico

Responsabilidades:
- Filtrar por nivel académico
- Identificar prerequisitos
- Detectar temas relacionados
- Enriquecer con metadata pedagógica
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Agregar directorio raíz al path para importar desde offline_etl
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from runtime_mcp.engines.hybrid_engine import HybridEngine, RetrievalResult
from offline_etl.models.academic_kg_model import AcademicLevel


@dataclass
class AcademicContext:
    """Contexto académico enriquecido para una consulta"""

    query: str
    results: List[RetrievalResult]
    nivel_academico: str
    conceptos_principales: List[Dict]
    temas_relacionados: List[Dict]
    prerequisitos: List[Dict]
    referencias_sugeridas: List[Dict]


class ContextEnricher:
    """Enriquecedor de contexto académico"""

    def __init__(self, hybrid_engine: HybridEngine):
        """
        Inicializa enricher con motor híbrido

        Args:
            hybrid_engine: Motor híbrido ya inicializado
        """
        self.engine = hybrid_engine
        self.driver = hybrid_engine.driver

    def enrich_for_level(
        self, query: str, nivel_academico: str = "intermedio", k: int = 5
    ) -> AcademicContext:
        """
        Enriquece contexto según nivel académico

        Args:
            query: Consulta del estudiante
            nivel_academico: básico, intermedio, avanzado
            k: Número de resultados

        Returns:
            AcademicContext con resultados enriquecidos
        """
        # 1. Búsqueda híbrida base
        results = self.engine.hybrid_search(query, k=k, expand_graph=True)

        # 2. Filtrar por nivel (si hay metadata de nivel)
        level_filtered_results = self._filter_by_level(results, nivel_academico)

        # 3. Extraer conceptos principales
        conceptos = self._extract_main_concepts(level_filtered_results)

        # 4. Identificar temas relacionados
        temas = self._extract_related_topics(level_filtered_results)

        # 5. Detectar prerequisitos
        prerequisitos = self._identify_prerequisites(temas, nivel_academico)

        # 6. Sugerir referencias
        referencias = self._suggest_references(level_filtered_results)

        return AcademicContext(
            query=query,
            results=level_filtered_results,
            nivel_academico=nivel_academico,
            conceptos_principales=conceptos,
            temas_relacionados=temas,
            prerequisitos=prerequisitos,
            referencias_sugeridas=referencias,
        )

    def _filter_by_level(
        self, results: List[RetrievalResult], target_level: str
    ) -> List[RetrievalResult]:
        """
        Filtra resultados por nivel académico

        Args:
            results: Resultados de búsqueda
            target_level: Nivel objetivo (básico, intermedio, avanzado)

        Returns:
            Resultados filtrados
        """
        # Mapeo de niveles a prioridad
        level_priority = {
            "básico": ["básico", "intermedio"],
            "intermedio": ["intermedio", "básico", "avanzado"],
            "avanzado": ["avanzado", "intermedio"],
        }

        preferred_levels = level_priority.get(target_level, ["intermedio"])

        # Filtrar y reordenar por nivel
        filtered = []
        for result in results:
            metadata = result.metadata
            doc_level = metadata.get("level", "intermedio")

            # Agregar con prioridad según nivel
            if doc_level in preferred_levels:
                result.metadata["level_match"] = preferred_levels.index(doc_level)
                filtered.append(result)

        # Ordenar por level_match y luego por vector_score
        filtered.sort(key=lambda r: (r.metadata.get("level_match", 10), -r.vector_score))

        return filtered

    def _extract_main_concepts(self, results: List[RetrievalResult]) -> List[Dict]:
        """
        Extrae conceptos principales de los resultados

        Args:
            results: Resultados enriquecidos

        Returns:
            Lista de conceptos únicos con frecuencia
        """
        concept_freq = {}

        for result in results:
            enriched = result.enriched_context

            # Concepts del documento
            for concept in enriched.get("concepts", []):
                concept_name = concept.get("name", "")
                if concept_name:
                    if concept_name not in concept_freq:
                        concept_freq[concept_name] = {
                            "name": concept_name,
                            "id": concept.get("id", ""),
                            "level": concept.get("level", "intermedio"),
                            "frequency": 0,
                        }
                    concept_freq[concept_name]["frequency"] += 1

            # Related concepts
            for concept in enriched.get("related_concepts", []):
                concept_name = concept.get("name", "")
                if concept_name and concept_name not in concept_freq:
                    concept_freq[concept_name] = {
                        "name": concept_name,
                        "id": concept.get("id", ""),
                        "level": concept.get("level", "intermedio"),
                        "frequency": 0,
                    }

        # Ordenar por frecuencia
        concepts = sorted(concept_freq.values(), key=lambda x: x["frequency"], reverse=True)

        return concepts[:10]  # Top 10

    def _extract_related_topics(self, results: List[RetrievalResult]) -> List[Dict]:
        """
        Extrae temas relacionados de los resultados

        Args:
            results: Resultados enriquecidos

        Returns:
            Lista de temas únicos
        """
        topic_set = {}

        for result in results:
            enriched = result.enriched_context

            for topic in enriched.get("topics", []):
                topic_id = topic.get("id", "")
                if topic_id and topic_id not in topic_set:
                    topic_set[topic_id] = {
                        "id": topic_id,
                        "name": topic.get("name", ""),
                        "level": topic.get("level", "intermedio"),
                        "area": topic.get("area", "general"),
                    }

        return list(topic_set.values())

    def _identify_prerequisites(self, topics: List[Dict], current_level: str) -> List[Dict]:
        """
        Identifica prerequisitos necesarios para los temas

        Args:
            topics: Lista de temas relacionados
            current_level: Nivel académico actual

        Returns:
            Lista de prerequisitos
        """
        if not topics:
            return []

        # Query Cypher para encontrar prerequisitos
        with self.driver.session() as session:
            topic_ids = [t["id"] for t in topics]

            query = """
            MATCH (topic:Topic)
            WHERE topic.id IN $topic_ids
            MATCH (course:Course)-[:TEACHES]->(topic)
            OPTIONAL MATCH (course)-[:REQUIRES]->(prereq:Course)
            OPTIONAL MATCH (prereq)-[:TEACHES]->(prereq_topic:Topic)
            
            RETURN DISTINCT prereq_topic
            """

            result = session.run(query, topic_ids=topic_ids)

            prerequisites = []
            for record in result:
                prereq_topic = record["prereq_topic"]
                if prereq_topic:
                    prerequisites.append(
                        {
                            "id": prereq_topic.get("id", ""),
                            "name": prereq_topic.get("name", ""),
                            "level": prereq_topic.get("level", "básico"),
                        }
                    )

            return prerequisites

    def _suggest_references(self, results: List[RetrievalResult]) -> List[Dict]:
        """
        Sugiere referencias bibliográficas relevantes

        Args:
            results: Resultados enriquecidos

        Returns:
            Lista de referencias únicas
        """
        ref_set = {}

        for result in results:
            enriched = result.enriched_context

            for ref in enriched.get("references", []):
                ref_id = ref.get("id", "")
                if ref_id and ref_id not in ref_set:
                    ref_set[ref_id] = {
                        "id": ref_id,
                        "title": ref.get("title", ""),
                        "authors": ref.get("authors", ""),
                        "year": ref.get("year", ""),
                        "type": ref.get("type", "book"),
                    }

        # Ordenar por año (más reciente primero)
        references = sorted(ref_set.values(), key=lambda x: x.get("year", 0), reverse=True)

        return references[:5]  # Top 5

    def format_academic_context(self, context: AcademicContext) -> str:
        """
        Formatea contexto académico para el LLM

        Args:
            context: AcademicContext enriquecido

        Returns:
            String formateado para prompt del LLM
        """
        parts = []

        # Header
        parts.append(f"=== CONTEXTO ACADÉMICO ===")
        parts.append(f"Query: {context.query}")
        parts.append(f"Nivel: {context.nivel_academico}")
        parts.append("")

        # Conceptos principales
        if context.conceptos_principales:
            parts.append("📚 CONCEPTOS PRINCIPALES:")
            for concept in context.conceptos_principales[:5]:
                parts.append(f"   • {concept['name']} (nivel: {concept['level']})")
            parts.append("")

        # Temas relacionados
        if context.temas_relacionados:
            parts.append("🎯 TEMAS RELACIONADOS:")
            for tema in context.temas_relacionados[:3]:
                parts.append(f"   • {tema['name']}")
            parts.append("")

        # Prerequisitos
        if context.prerequisitos:
            parts.append("⚠️  PREREQUISITOS RECOMENDADOS:")
            for prereq in context.prerequisitos[:3]:
                parts.append(f"   • {prereq['name']}")
            parts.append("")

        # Documentos
        parts.append("📄 DOCUMENTOS RELEVANTES:")
        for i, result in enumerate(context.results[:3], 1):
            parts.append(f"\n[DOCUMENTO {i}] (Score: {result.vector_score:.3f})")
            parts.append(result.content[:300] + "...")

        parts.append("")

        # Referencias
        if context.referencias_sugeridas:
            parts.append("📖 REFERENCIAS SUGERIDAS:")
            for ref in context.referencias_sugeridas[:3]:
                parts.append(f"   • {ref['title']} - {ref['authors']} ({ref['year']})")

        return "\n".join(parts)


def demo():
    """Demo del context enricher"""
    from engines.hybrid_engine import HybridEngine
    from pathlib import Path

    print("\n" + "=" * 70)
    print("📚 DEMO: CONTEXT ENRICHER")
    print("=" * 70)

    # Inicializar motor híbrido
    engine = HybridEngine(
        neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="12345678"
    )

    try:
        # Cargar índice
        engine.load_index(Path("data/indices/hnsw_index.bin"))

        # Crear enricher
        enricher = ContextEnricher(engine)

        # Consulta de prueba
        query = "¿Cómo funciona el algoritmo PSO?"

        print(f"\n🔍 Consulta: {query}")
        print(f"📊 Nivel: intermedio")

        # Enriquecer contexto
        context = enricher.enrich_for_level(query, nivel_academico="intermedio", k=5)

        # Formatear y mostrar
        formatted = enricher.format_academic_context(context)
        print(f"\n{formatted}")

    finally:
        engine.close()


if __name__ == "__main__":
    demo()

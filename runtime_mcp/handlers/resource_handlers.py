# resource_handlers.py
"""
📚 Resource Handlers - Implementación de Resources MCP Académicos
Según especificación MCP: Application-controlled, fuentes de datos para el LLM

Resources implementados:
1. course_catalog - Catálogo de cursos disponibles
2. document_list - Lista de documentos procesados
3. kg_statistics - Estadísticas del knowledge graph

Autor: Rodrigo Cárdenas
Basado en: Model Context Protocol Specification
"""

from typing import List, Dict, Any
from pathlib import Path
import json
from mcp.types import Resource, TextContent
from neo4j import GraphDatabase


class AcademicResourceHandlers:
    """Handlers para resources MCP académicos"""

    def __init__(
        self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, processed_docs_dir: Path
    ):
        """
        Inicializa handlers con conexión a Neo4j y directorio de docs

        Args:
            neo4j_uri: URI de Neo4j
            neo4j_user: Usuario Neo4j
            neo4j_password: Password Neo4j
            processed_docs_dir: Directorio con documentos procesados
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.processed_docs_dir = Path(processed_docs_dir)

    def close(self):
        """Cierra conexión a Neo4j"""
        self.driver.close()

    def get_resource_definitions(self) -> List[Resource]:
        """
        Retorna definiciones de resources MCP

        Returns:
            Lista de Resource según especificación MCP
        """
        return [
            Resource(
                uri="academic://course_catalog",
                name="Catálogo de Cursos",
                description=(
                    "Catálogo completo de cursos disponibles en el knowledge graph. "
                    "Incluye información sobre código, nombre, créditos, nivel académico, "
                    "prerequisitos y temas enseñados."
                ),
                mimeType="application/json",
            ),
            Resource(
                uri="academic://document_list",
                name="Lista de Documentos",
                description=(
                    "Lista de todos los documentos académicos procesados y disponibles "
                    "para consulta. Incluye PDFs, notebooks, presentaciones, etc."
                ),
                mimeType="application/json",
            ),
            Resource(
                uri="academic://kg_statistics",
                name="Estadísticas del Knowledge Graph",
                description=(
                    "Estadísticas detalladas del knowledge graph académico: "
                    "número de nodos por tipo, relaciones, cobertura de temas, etc."
                ),
                mimeType="application/json",
            ),
        ]

    async def handle_course_catalog(self) -> List[TextContent]:
        """
        Handler para resource: course_catalog

        Returns:
            Lista con TextContent conteniendo catálogo de cursos
        """
        try:
            with self.driver.session() as session:
                # Query para obtener todos los cursos con sus detalles
                query = """
                MATCH (c:Course)
                OPTIONAL MATCH (c)-[:TEACHES]->(t:Topic)
                OPTIONAL MATCH (c)-[:REQUIRES]->(prereq:Course)
                
                RETURN 
                    c.id as id,
                    c.code as code,
                    c.name as name,
                    c.credits as credits,
                    c.level as level,
                    c.area as area,
                    collect(DISTINCT t.name) as topics,
                    collect(DISTINCT prereq.code) as prerequisites
                ORDER BY c.code
                """

                result = session.run(query)

                courses = []
                for record in result:
                    courses.append(
                        {
                            "id": record["id"],
                            "code": record["code"],
                            "name": record["name"],
                            "credits": record["credits"],
                            "level": record["level"],
                            "area": record["area"],
                            "topics_count": len([t for t in record["topics"] if t]),
                            "prerequisites": [p for p in record["prerequisites"] if p],
                        }
                    )

                # Formatear como JSON legible
                catalog = {
                    "total_courses": len(courses),
                    "courses": courses,
                }

                catalog_json = json.dumps(catalog, indent=2, ensure_ascii=False)

                return [
                    TextContent(
                        type="text",
                        text=f"# Catálogo de Cursos\n\n```json\n{catalog_json}\n```",
                    )
                ]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error al cargar catálogo de cursos: {str(e)}",
                )
            ]

    async def handle_document_list(self) -> List[TextContent]:
        """
        Handler para resource: document_list

        Returns:
            Lista con TextContent conteniendo lista de documentos
        """
        try:
            with self.driver.session() as session:
                # Query para obtener todos los documentos
                query = """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:Concept)
                
                RETURN 
                    d.id as id,
                    d.name as name,
                    d.type as type,
                    d.course_code as course_code,
                    count(DISTINCT c) as concepts_count
                ORDER BY d.name
                """

                result = session.run(query)

                documents = []
                for record in result:
                    documents.append(
                        {
                            "id": record["id"],
                            "name": record["name"],
                            "type": record["type"],
                            "course_code": record["course_code"],
                            "concepts_count": record["concepts_count"],
                        }
                    )

                # Contar chunks procesados
                chunk_files = list(self.processed_docs_dir.glob("*_chunks.json"))
                total_chunks = 0
                for chunk_file in chunk_files:
                    with open(chunk_file, "r") as f:
                        chunks = json.load(f)
                        total_chunks += len(chunks)

                # Formatear como JSON
                doc_list = {
                    "total_documents": len(documents),
                    "total_chunks": total_chunks,
                    "documents": documents,
                }

                doc_list_json = json.dumps(doc_list, indent=2, ensure_ascii=False)

                return [
                    TextContent(
                        type="text",
                        text=f"# Lista de Documentos\n\n```json\n{doc_list_json}\n```",
                    )
                ]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error al cargar lista de documentos: {str(e)}",
                )
            ]

    async def handle_kg_statistics(self) -> List[TextContent]:
        """
        Handler para resource: kg_statistics

        Returns:
            Lista con TextContent conteniendo estadísticas del KG
        """
        try:
            with self.driver.session() as session:
                # Contar nodos por tipo
                nodes_query = """
                MATCH (n)
                RETURN labels(n)[0] as type, count(n) as count
                ORDER BY count DESC
                """

                nodes_result = session.run(nodes_query)
                nodes_by_type = {record["type"]: record["count"] for record in nodes_result}

                # Contar relaciones por tipo
                rels_query = """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
                """

                rels_result = session.run(rels_query)
                rels_by_type = {record["type"]: record["count"] for record in rels_result}

                # Estadísticas generales
                total_nodes = sum(nodes_by_type.values())
                total_rels = sum(rels_by_type.values())

                # Formatear estadísticas
                stats = {
                    "summary": {
                        "total_nodes": total_nodes,
                        "total_relations": total_rels,
                        "node_types": len(nodes_by_type),
                        "relation_types": len(rels_by_type),
                    },
                    "nodes_by_type": nodes_by_type,
                    "relations_by_type": rels_by_type,
                }

                stats_json = json.dumps(stats, indent=2, ensure_ascii=False)

                return [
                    TextContent(
                        type="text",
                        text=f"# Estadísticas del Knowledge Graph\n\n```json\n{stats_json}\n```",
                    )
                ]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error al cargar estadísticas: {str(e)}",
                )
            ]

    async def route_resource_call(self, uri: str) -> List[TextContent]:
        """
        Rutea llamada a resource al handler correspondiente

        Args:
            uri: URI del resource (ej: academic://course_catalog)

        Returns:
            Lista de TextContent con datos del resource
        """
        handlers = {
            "academic://course_catalog": self.handle_course_catalog,
            "academic://document_list": self.handle_document_list,
            "academic://kg_statistics": self.handle_kg_statistics,
        }

        handler = handlers.get(uri)

        if not handler:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Resource '{uri}' no reconocido.",
                )
            ]

        return await handler()

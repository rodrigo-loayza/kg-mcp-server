# tool_handlers.py
"""
🛠️ Tool Handlers - Implementación de Tools MCP Académicos
Según especificación MCP: Model-controlled, ejecutables por el LLM

Tools implementados:
1. consulta_conceptual - Búsqueda conceptual en materiales académicos
2. analisis_codigo - Análisis de código con ejemplos similares
3. navegacion_prerequisitos - Navegación de prerequisitos temáticos

Autor: Rodrigo Cárdenas
Basado en: Model Context Protocol Specification
"""

from typing import Any, Dict, List
import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.types import Tool, TextContent, EmbeddedResource
from runtime_mcp.engines.hybrid_engine import HybridEngine
from runtime_mcp.engines.context_enricher import ContextEnricher


class AcademicToolHandlers:
    """Handlers para tools MCP académicos"""

    def __init__(self, hybrid_engine: HybridEngine, context_enricher: ContextEnricher):
        """
        Inicializa handlers con motores de búsqueda

        Args:
            hybrid_engine: Motor de búsqueda híbrida
            context_enricher: Enriquecedor de contexto académico
        """
        self.engine = hybrid_engine
        self.enricher = context_enricher

    def get_tool_definitions(self) -> List[Tool]:
        """
        Retorna definiciones de tools MCP

        Returns:
            Lista de Tool según especificación MCP
        """
        return [
            Tool(
                name="consulta_conceptual",
                description=(
                    "Busca explicaciones conceptuales en materiales académicos del curso. "
                    "Ideal para preguntas teóricas sobre algoritmos, estructuras de datos, "
                    "conceptos de computación, etc. Retorna contexto enriquecido con "
                    "conceptos relacionados, prerequisitos y referencias."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Pregunta o concepto a buscar en materiales académicos",
                        },
                        "nivel_academico": {
                            "type": "string",
                            "enum": ["básico", "intermedio", "avanzado"],
                            "description": "Nivel académico del estudiante para filtrar contenido apropiado",
                            "default": "intermedio",
                        },
                        "k_results": {
                            "type": "integer",
                            "description": "Número de documentos relevantes a retornar",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="analisis_codigo",
                description=(
                    "Analiza fragmentos de código y encuentra implementaciones similares "
                    "en los materiales académicos del curso. Útil para entender patrones "
                    "de código, comparar implementaciones, o encontrar ejemplos relacionados."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code_fragment": {
                            "type": "string",
                            "description": "Fragmento de código a analizar (Python, Java, C++, etc.)",
                        },
                        "language": {
                            "type": "string",
                            "description": "Lenguaje de programación del código",
                            "default": "python",
                        },
                        "context": {
                            "type": "string",
                            "description": "Contexto adicional sobre qué se busca entender del código",
                        },
                    },
                    "required": ["code_fragment"],
                },
            ),
            Tool(
                name="navegacion_prerequisitos",
                description=(
                    "Navega la jerarquía de prerequisitos de un tema académico. "
                    "Identifica qué conceptos o temas deben dominarse antes de estudiar "
                    "un tema específico. Útil para planificar el orden de estudio."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tema": {
                            "type": "string",
                            "description": "Nombre del tema o concepto para explorar prerequisitos",
                        },
                        "profundidad": {
                            "type": "integer",
                            "description": "Niveles de prerequisitos a explorar (1-3)",
                            "default": 2,
                            "minimum": 1,
                            "maximum": 3,
                        },
                    },
                    "required": ["tema"],
                },
            ),
        ]

    async def handle_consulta_conceptual(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Handler para tool: consulta_conceptual

        Args:
            arguments: Dict con query, nivel_academico, k_results

        Returns:
            Lista de TextContent con resultados
        """
        query = arguments.get("query", "")
        nivel_academico = arguments.get("nivel_academico", "intermedio")
        k_results = arguments.get("k_results", 5)

        # Validar query
        if not query:
            return [
                TextContent(
                    type="text",
                    text="Error: Se requiere una query para buscar.",
                )
            ]

        try:
            # Enriquecer contexto con nivel académico
            context = self.enricher.enrich_for_level(
                query=query, nivel_academico=nivel_academico, k=k_results
            )

            # Formatear contexto académico
            formatted_context = self.enricher.format_academic_context(context)

            # Construir respuesta
            response_parts = [
                f"# Consulta Conceptual: {query}",
                f"*Nivel académico: {nivel_academico}*",
                "",
                formatted_context,
                "",
                f"---",
                f"*Resultados basados en {len(context.results)} documentos del curso*",
            ]

            return [
                TextContent(
                    type="text",
                    text="\n".join(response_parts),
                )
            ]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error al procesar consulta: {str(e)}",
                )
            ]

    async def handle_analisis_codigo(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Handler para tool: analisis_codigo

        Args:
            arguments: Dict con code_fragment, language, context

        Returns:
            Lista de TextContent con análisis
        """
        code_fragment = arguments.get("code_fragment", "")
        language = arguments.get("language", "python")
        context_desc = arguments.get("context", "")

        if not code_fragment:
            return [
                TextContent(
                    type="text",
                    text="Error: Se requiere un fragmento de código para analizar.",
                )
            ]

        try:
            # Construir query basada en código
            query = f"código {language}: {code_fragment}"
            if context_desc:
                query += f" {context_desc}"

            # Buscar implementaciones similares
            results = self.engine.hybrid_search(query, k=5, expand_graph=True)

            # Formatear resultados
            response_parts = [
                f"# Análisis de Código ({language})",
                "",
                "## Código analizado:",
                f"```{language}",
                code_fragment,
                "```",
                "",
            ]

            if context_desc:
                response_parts.extend(
                    [
                        f"## Contexto:",
                        context_desc,
                        "",
                    ]
                )

            response_parts.append("## Implementaciones similares encontradas:")
            response_parts.append("")

            for i, result in enumerate(results[:3], 1):
                response_parts.extend(
                    [
                        f"### Ejemplo {i} (Similitud: {result.vector_score:.2%})",
                        result.content[:400] + "...",
                        "",
                    ]
                )

                # Agregar conceptos relacionados si existen
                enriched = result.enriched_context
                if enriched.get("concepts"):
                    concepts = [c.get("name", "") for c in enriched["concepts"][:3]]
                    response_parts.append(f"*Conceptos: {', '.join(concepts)}*")
                    response_parts.append("")

            return [
                TextContent(
                    type="text",
                    text="\n".join(response_parts),
                )
            ]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error al analizar código: {str(e)}",
                )
            ]

    async def handle_navegacion_prerequisitos(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Handler para tool: navegacion_prerequisitos

        Args:
            arguments: Dict con tema, profundidad

        Returns:
            Lista de TextContent con prerequisitos
        """
        tema = arguments.get("tema", "")
        profundidad = arguments.get("profundidad", 2)

        if not tema:
            return [
                TextContent(
                    type="text",
                    text="Error: Se requiere un tema para explorar prerequisitos.",
                )
            ]

        try:
            # Buscar tema en el grafo
            driver = self.engine.driver

            with driver.session() as session:
                # Query para encontrar tema y sus prerequisitos
                query = """
                MATCH (topic:Topic)
                WHERE toLower(topic.name) CONTAINS toLower($tema)
                
                OPTIONAL MATCH (course:Course)-[:TEACHES]->(topic)
                OPTIONAL MATCH (course)-[:REQUIRES*1..3]->(prereq_course:Course)
                OPTIONAL MATCH (prereq_course)-[:TEACHES]->(prereq_topic:Topic)
                
                RETURN 
                    topic,
                    course,
                    collect(DISTINCT prereq_topic) as prerequisitos
                LIMIT 1
                """

                result = session.run(query, tema=tema)
                record = result.single()

                if not record or not record["topic"]:
                    return [
                        TextContent(
                            type="text",
                            text=f"No se encontró el tema '{tema}' en el grafo de conocimiento.",
                        )
                    ]

                topic = dict(record["topic"])
                course = dict(record["course"]) if record["course"] else {}
                prerequisitos = [dict(p) for p in record["prerequisitos"] if p]

                # Construir respuesta
                response_parts = [
                    f"# Prerequisitos: {topic.get('name', tema)}",
                    "",
                ]

                if course:
                    response_parts.extend(
                        [
                            f"## Curso:",
                            f"**{course.get('name', 'N/A')}** ({course.get('code', 'N/A')})",
                            "",
                        ]
                    )

                response_parts.extend(
                    [
                        f"## Nivel del tema:",
                        f"{topic.get('level', 'intermedio').title()}",
                        "",
                    ]
                )

                if prerequisitos:
                    response_parts.extend(
                        [
                            f"## Prerequisitos recomendados:",
                            "",
                        ]
                    )

                    for i, prereq in enumerate(prerequisitos, 1):
                        prereq_name = prereq.get("name", "N/A")
                        prereq_level = prereq.get("level", "básico")
                        response_parts.append(f"{i}. **{prereq_name}** (Nivel: {prereq_level})")
                else:
                    response_parts.extend(
                        [
                            "## Prerequisitos:",
                            "No se identificaron prerequisitos específicos para este tema.",
                            "",
                        ]
                    )

                response_parts.extend(
                    [
                        "",
                        "---",
                        "*Información basada en el grafo de conocimiento del curso*",
                    ]
                )

                return [
                    TextContent(
                        type="text",
                        text="\n".join(response_parts),
                    )
                ]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error al navegar prerequisitos: {str(e)}",
                )
            ]

    async def route_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Rutea llamada a tool al handler correspondiente

        Args:
            tool_name: Nombre del tool a ejecutar
            arguments: Argumentos del tool

        Returns:
            Lista de TextContent con resultados
        """
        handlers = {
            "consulta_conceptual": self.handle_consulta_conceptual,
            "analisis_codigo": self.handle_analisis_codigo,
            "navegacion_prerequisitos": self.handle_navegacion_prerequisitos,
        }

        handler = handlers.get(tool_name)

        if not handler:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Tool '{tool_name}' no reconocido.",
                )
            ]

        return await handler(arguments)

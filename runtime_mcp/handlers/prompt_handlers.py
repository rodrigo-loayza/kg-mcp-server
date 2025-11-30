# prompt_handlers.py
"""
💬 Prompt Handlers - Implementación de Prompts MCP Académicos
Según especificación MCP: User-controlled, templates predefinidos

Prompts implementados:
1. explicacion_conceptual - Template para explicar conceptos
2. resolucion_problema - Template para resolver problemas algorítmicos
3. revision_codigo - Template para revisar código
4. planificacion_estudio - Template para planificar estudio

Autor: Rodrigo Cárdenas
Basado en: Model Context Protocol Specification
"""

from typing import List, Dict, Any
from mcp.types import Prompt, PromptMessage, TextContent


class AcademicPromptHandlers:
    """Handlers para prompts MCP académicos"""

    def get_prompt_definitions(self) -> List[Prompt]:
        """
        Retorna definiciones de prompts MCP

        Returns:
            Lista de Prompt según especificación MCP
        """
        return [
            Prompt(
                name="explicacion_conceptual",
                description=(
                    "Template para solicitar explicaciones detalladas de conceptos "
                    "académicos de ciencias de la computación. Ajusta nivel de detalle "
                    "según el nivel académico del estudiante."
                ),
                arguments=[
                    {
                        "name": "concepto",
                        "description": "Concepto a explicar",
                        "required": True,
                    },
                    {
                        "name": "nivel",
                        "description": "Nivel académico (básico, intermedio, avanzado)",
                        "required": False,
                    },
                ],
            ),
            Prompt(
                name="resolucion_problema",
                description=(
                    "Template para resolver problemas algorítmicos paso a paso. "
                    "Incluye análisis de complejidad, casos edge, y optimizaciones."
                ),
                arguments=[
                    {
                        "name": "problema",
                        "description": "Descripción del problema a resolver",
                        "required": True,
                    },
                    {
                        "name": "restricciones",
                        "description": "Restricciones del problema",
                        "required": False,
                    },
                ],
            ),
            Prompt(
                name="revision_codigo",
                description=(
                    "Template para revisar código académico. Analiza correctitud, "
                    "eficiencia, estilo y buenas prácticas."
                ),
                arguments=[
                    {
                        "name": "codigo",
                        "description": "Código a revisar",
                        "required": True,
                    },
                    {
                        "name": "lenguaje",
                        "description": "Lenguaje de programación",
                        "required": False,
                    },
                ],
            ),
            Prompt(
                name="planificacion_estudio",
                description=(
                    "Template para crear un plan de estudio personalizado basado "
                    "en temas, prerequisitos y nivel académico."
                ),
                arguments=[
                    {
                        "name": "tema_objetivo",
                        "description": "Tema que se quiere dominar",
                        "required": True,
                    },
                    {
                        "name": "tiempo_disponible",
                        "description": "Tiempo disponible (ej: 2 semanas)",
                        "required": False,
                    },
                ],
            ),
        ]

    async def handle_explicacion_conceptual(self, arguments: Dict[str, Any]) -> PromptMessage:
        """
        Handler para prompt: explicacion_conceptual

        Args:
            arguments: Dict con concepto, nivel

        Returns:
            PromptMessage con template de explicación
        """
        concepto = arguments.get("concepto", "")
        nivel = arguments.get("nivel", "intermedio")

        # Ajustar profundidad según nivel
        profundidad_map = {
            "básico": "una explicación introductoria y accesible",
            "intermedio": "una explicación detallada con ejemplos",
            "avanzado": "un análisis profundo incluyendo teoría y aplicaciones avanzadas",
        }

        profundidad = profundidad_map.get(nivel, profundidad_map["intermedio"])

        prompt_text = f"""Por favor, explica el concepto "{concepto}" en el contexto de ciencias de la computación.

Nivel académico: {nivel}
Proporciona {profundidad}.

La explicación debe incluir:
1. **Definición clara**: ¿Qué es {concepto}?
2. **Contexto**: ¿Cuándo y por qué se usa?
3. **Ejemplos**: Casos concretos de aplicación
4. **Relaciones**: Conceptos relacionados o prerequisitos
5. **Consideraciones**: Ventajas, limitaciones o trade-offs

Usa los documentos académicos del curso como referencia cuando sea apropiado."""

        return PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=prompt_text,
            ),
        )

    async def handle_resolucion_problema(self, arguments: Dict[str, Any]) -> PromptMessage:
        """
        Handler para prompt: resolucion_problema

        Args:
            arguments: Dict con problema, restricciones

        Returns:
            PromptMessage con template de resolución
        """
        problema = arguments.get("problema", "")
        restricciones = arguments.get("restricciones", "No especificadas")

        prompt_text = f"""Ayúdame a resolver el siguiente problema algorítmico paso a paso:

**Problema:**
{problema}

**Restricciones:**
{restricciones}

Por favor, proporciona:

1. **Comprensión del problema**:
   - Reformula el problema con tus propias palabras
   - Identifica inputs, outputs y casos edge

2. **Enfoque de solución**:
   - Propón al menos un enfoque algorítmico
   - Explica la intuición detrás del enfoque

3. **Implementación**:
   - Pseudocódigo o código comentado
   - Manejo de casos especiales

4. **Análisis de complejidad**:
   - Complejidad temporal O(?)
   - Complejidad espacial O(?)

5. **Optimizaciones** (si aplica):
   - ¿Se puede mejorar el enfoque?
   - Trade-offs de la solución

6. **Testing**:
   - Casos de prueba sugeridos
   - Casos edge a considerar

Usa ejemplos de los materiales del curso si son relevantes."""

        return PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=prompt_text,
            ),
        )

    async def handle_revision_codigo(self, arguments: Dict[str, Any]) -> PromptMessage:
        """
        Handler para prompt: revision_codigo

        Args:
            arguments: Dict con codigo, lenguaje

        Returns:
            PromptMessage con template de revisión
        """
        codigo = arguments.get("codigo", "")
        lenguaje = arguments.get("lenguaje", "Python")

        prompt_text = f"""Por favor, revisa el siguiente código {lenguaje}:

```{lenguaje.lower()}
{codigo}
```

Proporciona una revisión estructurada cubriendo:

1. **Correctitud**:
   - ¿El código hace lo que debe hacer?
   - ¿Hay bugs evidentes o casos edge no manejados?

2. **Eficiencia**:
   - ¿Cuál es la complejidad temporal y espacial?
   - ¿Se puede optimizar?

3. **Estilo y legibilidad**:
   - ¿Sigue convenciones del lenguaje?
   - ¿Nombres de variables son descriptivos?
   - ¿Está bien comentado?

4. **Buenas prácticas**:
   - ¿Usa estructuras de datos apropiadas?
   - ¿Maneja errores adecuadamente?
   - ¿Es mantenible?

5. **Sugerencias de mejora**:
   - Refactorización recomendada
   - Código mejorado (si aplica)

Compara con patrones vistos en los materiales del curso si es relevante."""

        return PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=prompt_text,
            ),
        )

    async def handle_planificacion_estudio(self, arguments: Dict[str, Any]) -> PromptMessage:
        """
        Handler para prompt: planificacion_estudio

        Args:
            arguments: Dict con tema_objetivo, tiempo_disponible

        Returns:
            PromptMessage con template de planificación
        """
        tema_objetivo = arguments.get("tema_objetivo", "")
        tiempo_disponible = arguments.get("tiempo_disponible", "No especificado")

        prompt_text = f"""Ayúdame a crear un plan de estudio para dominar el siguiente tema:

**Tema objetivo:** {tema_objetivo}
**Tiempo disponible:** {tiempo_disponible}

Por favor, genera un plan de estudio estructurado que incluya:

1. **Prerequisitos**:
   - ¿Qué conceptos debo dominar primero?
   - Orden sugerido de prerequisitos

2. **Desglose del tema**:
   - Subtemas o componentes principales
   - Orden lógico de aprendizaje

3. **Recursos del curso**:
   - Documentos relevantes a revisar
   - Ejercicios o ejemplos de código sugeridos

4. **Plan temporal** (si se especificó tiempo):
   - Distribución por días/semanas
   - Milestones de aprendizaje

5. **Evaluación**:
   - ¿Cómo verificar que domino el tema?
   - Ejercicios de autoevaluación

6. **Conexiones**:
   - ¿Cómo se relaciona con otros temas del curso?
   - Aplicaciones prácticas

Basa las recomendaciones en la estructura del curso y los materiales disponibles."""

        return PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=prompt_text,
            ),
        )

    async def route_prompt_call(self, prompt_name: str, arguments: Dict[str, Any]) -> PromptMessage:
        """
        Rutea llamada a prompt al handler correspondiente

        Args:
            prompt_name: Nombre del prompt
            arguments: Argumentos del prompt

        Returns:
            PromptMessage con template expandido
        """
        handlers = {
            "explicacion_conceptual": self.handle_explicacion_conceptual,
            "resolucion_problema": self.handle_resolucion_problema,
            "revision_codigo": self.handle_revision_codigo,
            "planificacion_estudio": self.handle_planificacion_estudio,
        }

        handler = handlers.get(prompt_name)

        if not handler:
            # Retornar prompt de error
            return PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Error: Prompt '{prompt_name}' no reconocido.",
                ),
            )

        return await handler(arguments)

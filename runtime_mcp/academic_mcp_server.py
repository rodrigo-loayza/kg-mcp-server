# academic_mcp_server.py
import sys
import os

# --- 🛡️ ZONA DE SEGURIDAD ---
# Redirigimos stdout a stderr para proteger el canal MCP.
original_stdout = sys.stdout
sys.stdout = sys.stderr
# ----------------------------

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Configurar logging a ARCHIVO (para depuración post-mortem)
base_path = Path(__file__).resolve().parent.parent
log_file_path = base_path / "mcp_debug.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("AcademicMCP")


# ===== MOCK HANDLERS (usados mientras carga el sistema real) =====
class MockToolHandlers:
    """Handlers mock que cargan el sistema real cuando se usan"""

    def __init__(self, server_instance):
        self.server = server_instance
        self.loading = False

    def get_tool_definitions(self):
        from mcp.types import Tool

        return [
            Tool(
                name="inicializar_sistema",
                description="🚀 Inicializa el sistema académico. EJECUTA ESTO PRIMERO. Tarda ~30 segundos.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="estado_sistema",
                description="Verifica el estado del sistema académico",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    async def route_tool_call(self, name, arguments):
        from mcp.types import TextContent

        if name == "inicializar_sistema":
            if self.server.is_ready:
                return [
                    TextContent(
                        type="text",
                        text="✅ Sistema ya está inicializado y listo para usar.",
                    )
                ]

            if self.loading:
                return [
                    TextContent(
                        type="text",
                        text="⏳ Ya se está cargando... espera unos segundos.",
                    )
                ]

            self.loading = True
            logger.info("🚀 Iniciando carga de recursos desde herramienta...")

            # Cargar recursos síncronamente
            self.server._sync_load_resources()

            self.loading = False

            if self.server.is_ready:
                return [
                    TextContent(
                        type="text",
                        text="✅ Sistema inicializado correctamente. Ahora puedes usar todas las herramientas académicas.",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text="❌ Error al inicializar. Revisa los logs en mcp_debug.log",
                    )
                ]

        if name == "estado_sistema":
            if self.server.is_ready:
                return [
                    TextContent(
                        type="text",
                        text="✅ Sistema completamente inicializado y listo para usar.",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text="⏳ Sistema no inicializado. Ejecuta 'inicializar_sistema' primero.",
                    )
                ]

        return [
            TextContent(
                type="text",
                text="⚠️ Primero ejecuta 'inicializar_sistema' para cargar el sistema académico.",
            )
        ]


class MockResourceHandlers:
    """Resource handlers mock"""

    def get_resource_definitions(self):
        return []

    async def route_resource_call(self, uri):
        from mcp.types import TextContent

        return [TextContent(type="text", text="Cargando...")]


class MockPromptHandlers:
    """Prompt handlers mock"""

    def get_prompt_definitions(self):
        return []

    async def route_prompt_call(self, name, arguments):
        from mcp.types import PromptMessage, TextContent

        return PromptMessage(role="user", content=TextContent(type="text", text="Cargando..."))


class AcademicMCPServer:
    def __init__(self):
        logger.info("🏁 Iniciando constructor del servidor...")

        # 1. Configuración (Rápido)
        config_path = base_path / "config" / "config.yaml"
        if not config_path.exists():
            logger.critical(f"❌ No encuentro config en: {config_path}")
            raise FileNotFoundError(f"Missing config: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.server = Server(self.config["mcp_server"]["name"])

        # Estado inicial - usar handlers mock hasta que cargue
        self.is_ready = False
        self.hybrid_engine = None

        # Crear handlers mock inmediatamente (pasando referencia a self)
        self.tool_handlers = MockToolHandlers(self)
        self.resource_handlers = MockResourceHandlers()
        self.prompt_handlers = MockPromptHandlers()

        # Registrar capabilities (solo definiciones)
        self._register_capabilities()
        logger.info("✅ Servidor instanciado con handlers mock. Esperando conexión...")

    def _sync_load_resources(self):
        """
        Carga síncrona de recursos pesados.
        """
        try:
            logger.info("🏎️ Iniciando imports pesados...")

            # Asegurar que el path esté configurado
            if str(base_path) not in sys.path:
                sys.path.insert(0, str(base_path))
                logger.info(f"   📁 Path agregado: {base_path}")

            # --- IMPORTS PESADOS ---
            try:
                logger.info("   [1/5] Importando HybridEngine (PyTorch)...")
                from runtime_mcp.engines.hybrid_engine import HybridEngine

                logger.info("   ✅ HybridEngine OK")
            except Exception as e:
                logger.error(f"   ❌ Error en HybridEngine: {e}", exc_info=True)
                raise

            try:
                logger.info("   [2/5] Importando ContextEnricher (SentenceTransformers)...")
                from runtime_mcp.engines.context_enricher import ContextEnricher

                logger.info("   ✅ ContextEnricher OK")
            except Exception as e:
                logger.error(f"   ❌ Error en ContextEnricher: {e}", exc_info=True)
                raise

            try:
                logger.info("   [3/5] Importando AcademicToolHandlers...")
                from runtime_mcp.handlers.tool_handlers import AcademicToolHandlers

                logger.info("   ✅ AcademicToolHandlers OK")
            except Exception as e:
                logger.error(f"   ❌ Error en AcademicToolHandlers: {e}", exc_info=True)
                raise

            try:
                logger.info("   [4/5] Importando AcademicResourceHandlers...")
                from runtime_mcp.handlers.resource_handlers import AcademicResourceHandlers

                logger.info("   ✅ AcademicResourceHandlers OK")
            except Exception as e:
                logger.error(f"   ❌ Error en AcademicResourceHandlers: {e}", exc_info=True)
                raise

            try:
                logger.info("   [5/5] Importando AcademicPromptHandlers...")
                from runtime_mcp.handlers.prompt_handlers import AcademicPromptHandlers

                logger.info("   ✅ AcademicPromptHandlers OK")
            except Exception as e:
                logger.error(f"   ❌ Error en AcademicPromptHandlers: {e}", exc_info=True)
                raise

            logger.info("✅ Todos los imports completados")

            # --- Configurar Motores ---
            neo4j_config = self.config["neo4j"]
            embeddings_config = self.config["embeddings"]

            logger.info("🔌 Conectando Neo4j y cargando Embeddings...")
            self.hybrid_engine = HybridEngine(
                neo4j_uri=neo4j_config["uri"],
                neo4j_user=neo4j_config["user"],
                neo4j_password=neo4j_config["password"],
                embedding_model=embeddings_config["model"],
                dimension=embeddings_config["dimension"],
            )

            logger.info("✅ Motor híbrido inicializado")

            # Cargar índice
            hnsw_path = base_path / self.config["paths"]["hnsw_index"]
            if hnsw_path.exists():
                logger.info(f"📥 Cargando índice HNSW desde {hnsw_path}...")
                self.hybrid_engine.load_index(hnsw_path)
                logger.info("✅ Índice HNSW cargado")
            else:
                logger.warning(f"⚠️ No se encontró índice HNSW en {hnsw_path}")

            context_enricher = ContextEnricher(self.hybrid_engine)
            logger.info("✅ Context enricher creado")

            # --- Configurar Handlers ---
            self.tool_handlers = AcademicToolHandlers(
                hybrid_engine=self.hybrid_engine, context_enricher=context_enricher
            )

            processed_dir = base_path / self.config["paths"]["processed_dir"]
            self.resource_handlers = AcademicResourceHandlers(
                neo4j_uri=neo4j_config["uri"],
                neo4j_user=neo4j_config["user"],
                neo4j_password=neo4j_config["password"],
                processed_docs_dir=processed_dir,
            )

            self.prompt_handlers = AcademicPromptHandlers()

            logger.info("✅ Handlers configurados")
            logger.info("🎉 Sistema completamente inicializado")

            self.is_ready = True

        except Exception as e:
            logger.error(f"❌ Error en carga síncrona: {e}", exc_info=True)
            self.is_ready = False

    def _register_capabilities(self):
        # Tools
        @self.server.list_tools()
        async def handle_list_tools():
            # Siempre retornar las herramientas actuales (mock o reales)
            return self.tool_handlers.get_tool_definitions()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Any):
            # No esperar - simplemente ejecutar con handlers actuales
            return await self.tool_handlers.route_tool_call(name, arguments)

        # Resources
        @self.server.list_resources()
        async def handle_list_resources():
            return self.resource_handlers.get_resource_definitions()

        @self.server.read_resource()
        async def handle_read_resource(uri: str):
            return await self.resource_handlers.route_resource_call(uri)

        # Prompts
        @self.server.list_prompts()
        async def handle_list_prompts():
            return self.prompt_handlers.get_prompt_definitions()

        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: Any):
            return await self.prompt_handlers.route_prompt_call(name, arguments)

    async def run(self):
        """Inicia el servidor MCP - responde inmediatamente, carga recursos on-demand"""
        logger.info("🚀 Iniciando servidor MCP (handlers mock, carga lazy)...")

        # NO cargar recursos aquí - solo iniciar el servidor
        # Los recursos se cargarán cuando se use 'inicializar_sistema'

        async with stdio_server() as (read, write):
            init_options = self.server.create_initialization_options()
            logger.info("📡 Servidor STDIO listo, esperando conexión...")
            await self.server.run(read, write, init_options)


if __name__ == "__main__":
    try:
        server = AcademicMCPServer()
        asyncio.run(server.run())
    except Exception as e:
        logger.critical(f"❌ Error en main: {e}", exc_info=True)

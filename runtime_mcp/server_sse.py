#!/usr/bin/env python
"""
Servidor MCP Académico con transporte SSE.
Ejecuta PRIMERO este script, luego abre Claude Desktop.

USO:
  python runtime_mcp/server_sse.py

Luego configura Claude Desktop con:
{
  "mcpServers": {
    "academic-cs": {
      "url": "http://localhost:8765/sse"
    }
  }
}
"""
import sys
import os
import time

# Configurar path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)


def log(msg):
    """Log message to stderr"""
    print(msg, file=sys.stderr, flush=True)


log("\n" + "=" * 60)
log("🎓 SERVIDOR MCP ACADÉMICO (SSE)")
log("=" * 60 + "\n")

# ===== PRECARGA DE MÓDULOS =====
log("⏳ Precargando módulos (tarda ~35 segundos)...\n")
t0 = time.time()

log("  [1/6] numpy...")
import numpy as np

log("         ✅")

log("  [2/6] PyTorch...")
import torch

log(f"         ✅ (v{torch.__version__})")

log("  [3/6] SentenceTransformers...")
from sentence_transformers import SentenceTransformer

log("         ✅")

log("  [4/6] Neo4j...")
from neo4j import GraphDatabase

log("         ✅")

log("  [5/6] hnswlib...")
import hnswlib

log("         ✅")

log("  [6/6] Módulos proyecto...")
from runtime_mcp.engines.hybrid_engine import HybridEngine
from runtime_mcp.engines.context_enricher import ContextEnricher
from runtime_mcp.handlers.tool_handlers import AcademicToolHandlers
from runtime_mcp.handlers.resource_handlers import AcademicResourceHandlers
from runtime_mcp.handlers.prompt_handlers import AcademicPromptHandlers

log("         ✅")

log(f"\n✅ Precarga completa en {time.time()-t0:.1f}s\n")

# ===== SERVIDOR MCP =====
import asyncio
import logging
import yaml
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse, Response
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MCP-SSE")

PORT = 8765

# Cargar config
config_path = Path(base_path) / "config" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Crear servidor MCP
mcp_server = Server(config["mcp_server"]["name"])

# ===== INICIALIZAR RECURSOS =====
log("🔌 Inicializando recursos...\n")

neo4j_cfg = config["neo4j"]
emb_cfg = config["embeddings"]

hybrid_engine = HybridEngine(
    neo4j_uri=neo4j_cfg["uri"],
    neo4j_user=neo4j_cfg["user"],
    neo4j_password=neo4j_cfg["password"],
    embedding_model=emb_cfg["model"],
    dimension=emb_cfg["dimension"],
)

hnsw_path = Path(base_path) / config["paths"]["hnsw_index"]
if hnsw_path.exists():
    log("  📥 Cargando índice HNSW...")
    hybrid_engine.load_index(hnsw_path)
    log("     ✅")

context_enricher = ContextEnricher(hybrid_engine)

tool_handlers = AcademicToolHandlers(hybrid_engine=hybrid_engine, context_enricher=context_enricher)

processed_dir = Path(base_path) / config["paths"]["processed_dir"]
resource_handlers = AcademicResourceHandlers(
    neo4j_uri=neo4j_cfg["uri"],
    neo4j_user=neo4j_cfg["user"],
    neo4j_password=neo4j_cfg["password"],
    processed_docs_dir=processed_dir,
)

prompt_handlers = AcademicPromptHandlers()

log("\n✅ Recursos inicializados!\n")


# ===== REGISTRAR HANDLERS =====
@mcp_server.list_tools()
async def list_tools():
    return tool_handlers.get_tool_definitions()


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any):
    return await tool_handlers.route_tool_call(name, arguments)


@mcp_server.list_resources()
async def list_resources():
    return resource_handlers.get_resource_definitions()


@mcp_server.read_resource()
async def read_resource(uri: str):
    return await resource_handlers.route_resource_call(uri)


@mcp_server.list_prompts()
async def list_prompts():
    return prompt_handlers.get_prompt_definitions()


@mcp_server.get_prompt()
async def get_prompt(name: str, arguments: Any):
    return await prompt_handlers.route_prompt_call(name, arguments)


# ===== SSE TRANSPORT =====
sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())
    return Response()


async def health(request):
    return JSONResponse({"status": "ok", "ready": True})


app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
        Route("/health", health),
    ],
)

if __name__ == "__main__":
    log("=" * 60)
    log(f"🚀 SERVIDOR LISTO EN: http://localhost:{PORT}/sse")
    log("=" * 60)
    log("\n📋 Configura Claude Desktop con:\n")
    log(
        """{
  "mcpServers": {
    "academic-cs": {
      "url": "http://localhost:8765/sse"
    }
  }
}"""
    )
    log("\n" + "=" * 60)
    log("Ctrl+C para detener\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

#!/usr/bin/env python
"""
Preload heavy modules BEFORE starting the MCP server.
This ensures the server can respond to initialize within 60 seconds.

MODO DE USO:
1. Ejecuta: python preload_and_serve.py --wait
2. Espera a que diga "LISTO - Presiona ENTER"
3. Presiona ENTER para iniciar el servidor
4. Abre Claude Desktop
"""
import sys
import os
import argparse

# Redirect stdout to stderr to protect MCP channel
original_stdout = sys.stdout
sys.stdout = sys.stderr

# Add path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

print(f"\n{'='*60}")
print("🎓 SERVIDOR MCP ACADÉMICO - PRECARGA")
print(f"{'='*60}\n")

# PRELOAD all heavy modules HERE
print("⏳ Precargando módulos pesados...\n")

print("  [1/6] Cargando numpy...")
import numpy as np

print("  ✅ numpy OK")

print("  [2/6] Cargando PyTorch...")
import torch

print(f"  ✅ PyTorch {torch.__version__} OK")

print("  [3/6] Cargando SentenceTransformers (esto tarda ~20s)...")
from sentence_transformers import SentenceTransformer

print("  ✅ SentenceTransformers OK")

print("  [4/6] Cargando Neo4j driver...")
from neo4j import GraphDatabase

print("  ✅ Neo4j driver OK")

print("  [5/6] Cargando hnswlib...")
import hnswlib

print("  ✅ hnswlib OK")

print("  [6/6] Cargando módulos del proyecto...")
from runtime_mcp.engines.hybrid_engine import HybridEngine
from runtime_mcp.engines.context_enricher import ContextEnricher
from runtime_mcp.handlers.tool_handlers import AcademicToolHandlers
from runtime_mcp.handlers.resource_handlers import AcademicResourceHandlers
from runtime_mcp.handlers.prompt_handlers import AcademicPromptHandlers

print("  ✅ Módulos del proyecto OK")

print(f"\n{'='*60}")
print("✅ TODOS LOS MÓDULOS PRECARGADOS!")
print(f"{'='*60}\n")

# Check for --wait flag
parser = argparse.ArgumentParser()
parser.add_argument("--wait", action="store_true", help="Esperar antes de iniciar servidor")
args, _ = parser.parse_known_args()

if args.wait:
    print("📋 INSTRUCCIONES:")
    print("   1. Mantén esta ventana abierta")
    print("   2. Presiona ENTER aquí para iniciar el servidor")
    print("   3. Luego abre/reinicia Claude Desktop\n")
    input(">>> Presiona ENTER para iniciar el servidor MCP... ")

print("\n🚀 Iniciando servidor MCP (los módulos ya están en memoria)...\n")

# Now start the actual MCP server (modules are already cached)
from runtime_mcp.academic_mcp_server import AcademicMCPServer
import asyncio

if __name__ == "__main__":
    try:
        server = AcademicMCPServer()
        asyncio.run(server.run())
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()

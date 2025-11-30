"""
SCRIPT PARA FORMATEAR CONTEXTO EXISTENTE PARA LLM
Toma un JSON de contexto y lo convierte en prompt listo para usar
"""

import json
import argparse
from pathlib import Path


def format_rag_prompt(context, query):
    """Formatea contexto RAG baseline para LLM"""

    prompt = f"""Eres un asistente académico experto en Ciencias de la Computación e Inteligencia Artificial.

{'='*80}
PREGUNTA DEL USUARIO:
{query}
{'='*80}

CONTEXTO RECUPERADO (Búsqueda Vectorial - RAG Baseline):

"""

    for item in context:
        prompt += f"""
────────────────────────────────────────────────────────────────────────────────
[Fragmento {item['rank']}]
📄 Documento: {item['document']}
📊 Relevancia (Similitud Coseno): {item['similarity']:.4f}

{item['text']}
"""

    prompt += f"""
{'='*80}

INSTRUCCIONES:
1. Responde la pregunta basándote ÚNICAMENTE en los fragmentos proporcionados arriba
2. Si la información no está disponible en el contexto, indica claramente: "La información no está disponible en el contexto proporcionado"
3. Cita los fragmentos relevantes usando la notación: [Fragmento X]
4. Sé académicamente riguroso y preciso
5. Estructura tu respuesta de forma clara y organizada

TU RESPUESTA:
"""

    return prompt


def format_hybrid_prompt(context, query):
    """Formatea contexto híbrido (vector + grafo) para LLM"""

    prompt = f"""Eres un asistente académico experto con acceso a un Knowledge Graph semántico de conceptos de Ciencias de la Computación (basado en CSO - Computer Science Ontology).

{'='*80}
PREGUNTA DEL USUARIO:
{query}
{'='*80}

CONTEXTO RECUPERADO (Sistema Híbrido: Búsqueda Vectorial + Knowledge Graph):

"""

    for item in context:
        prompt += f"""
────────────────────────────────────────────────────────────────────────────────
[Fragmento {item['rank']}]

📄 Documento: {item['document']}
📊 Score Híbrido: {item['hybrid_score']:.4f}
   ├─ Score Vectorial (Similitud): {item['vector_score']:.4f}
   └─ Score de Grafo (Semántico): {item['graph_score']:.4f}

"""

        # Conceptos CSO
        if item.get("concepts"):
            concepts_str = ", ".join(item["concepts"][:10])
            if len(item["concepts"]) > 10:
                concepts_str += f"... (+{len(item['concepts']) - 10} más)"

            prompt += f"🏷️  Conceptos CSO Identificados:\n   {concepts_str}\n\n"

        # Conceptos relacionados
        if item.get("related_concepts"):
            related_str = ", ".join(item["related_concepts"][:10])
            if len(item["related_concepts"]) > 10:
                related_str += f"... (+{len(item['related_concepts']) - 10} más)"

            prompt += f"🔗 Conceptos Relacionados (Knowledge Graph):\n   {related_str}\n\n"

        prompt += f"📝 TEXTO:\n{item['text']}\n"

    prompt += f"""
{'='*80}

INSTRUCCIONES:
1. Usa tanto el TEXTO como las relaciones del KNOWLEDGE GRAPH para responder
2. Los conceptos CSO y sus relaciones te dan contexto semántico adicional
3. Si mencionas conceptos técnicos, verifica que estén en el contexto o en los conceptos CSO
4. Cita fragmentos relevantes usando: [Fragmento X]
5. Si la información no está en el contexto, indícalo claramente
6. Aprovecha las relaciones conceptuales para dar respuestas más completas

NOTA TÉCNICA:
- Este contexto fue recuperado usando un sistema híbrido que combina:
  * 80% Búsqueda Vectorial (similitud semántica con sentence-transformers)
  * 20% Knowledge Graph (enriquecimiento conceptual con CSO)

TU RESPUESTA:
"""

    return prompt


def format_comparison_prompt(rag_context, hybrid_context, query):
    """Crea prompt comparando ambos enfoques lado a lado"""

    prompt = f"""ANÁLISIS COMPARATIVO: RAG BASELINE vs SISTEMA HÍBRIDO (Vector + Knowledge Graph)

{'='*100}
PREGUNTA: {query}
{'='*100}

═══════════════════════════════════════════════════════════════════════════════
CONTEXTO #1: RAG BASELINE (Solo Búsqueda Vectorial)
═══════════════════════════════════════════════════════════════════════════════

"""

    for item in rag_context:
        prompt += f"""
[RAG-{item['rank']}] {item['document']} (Similitud: {item['similarity']:.4f})
{item['text'][:300]}...

"""

    prompt += f"""
═══════════════════════════════════════════════════════════════════════════════
CONTEXTO #2: SISTEMA HÍBRIDO (Vector + Knowledge Graph CSO)
═══════════════════════════════════════════════════════════════════════════════

"""

    for item in hybrid_context:
        concepts = ", ".join(item.get("concepts", [])[:5])
        prompt += f"""
[HÍBRIDO-{item['rank']}] {item['document']} (Híbrido: {item['hybrid_score']:.4f}, Vector: {item['vector_score']:.4f}, Grafo: {item['graph_score']:.4f})
Conceptos: {concepts}
{item['text'][:300]}...

"""

    prompt += f"""
{'='*100}

TAREA:
Compara ambos contextos y responde:

1. ¿Qué diferencias observas en los fragmentos recuperados?
2. ¿El sistema híbrido recuperó documentos diferentes al RAG baseline?
3. ¿Los conceptos CSO del sistema híbrido agregan valor semántico?
4. Responde la pregunta del usuario usando el MEJOR contexto disponible
5. Indica cuál sistema proporcionó mejor información y por qué

TU ANÁLISIS:
"""

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Formatea contexto JSON para uso con LLM")

    parser.add_argument("--input", type=str, required=True, help="Archivo JSON con el contexto")

    parser.add_argument("--query", type=str, required=True, help="Pregunta original del usuario")

    parser.add_argument("--output", type=str, help="Archivo de salida (default: input_prompt.txt)")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "rag", "hybrid"],
        default="auto",
        help="Modo de formateo (auto detecta del JSON)",
    )

    parser.add_argument(
        "--compare",
        type=str,
        help="JSON adicional para comparar (ej: contexto RAG para comparar con híbrido)",
    )

    args = parser.parse_args()

    # Cargar contexto principal
    with open(args.input, "r", encoding="utf-8") as f:
        context = json.load(f)

    # Detectar tipo si es auto
    if args.mode == "auto":
        if isinstance(context, list) and len(context) > 0:
            if "hybrid_score" in context[0]:
                mode = "hybrid"
            else:
                mode = "rag"
        else:
            mode = "rag"
    else:
        mode = args.mode

    # Si hay comparación
    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as f:
            compare_context = json.load(f)

        # Determinar cuál es RAG y cuál híbrido
        if "hybrid_score" in context[0]:
            prompt = format_comparison_prompt(compare_context, context, args.query)
        else:
            prompt = format_comparison_prompt(context, compare_context, args.query)
    else:
        # Formatear según modo
        if mode == "hybrid":
            prompt = format_hybrid_prompt(context, args.query)
        else:
            prompt = format_rag_prompt(context, args.query)

    # Determinar archivo de salida
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_prompt.txt"

    # Guardar
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"✅ Prompt formateado guardado en: {output_path}")
    print(f"📏 Longitud: {len(prompt)} caracteres (~{len(prompt.split())} palabras)")
    print(f"🎯 Modo: {mode.upper()}")

    if args.compare:
        print("🔄 Modo comparación activado")

    print(f"\n💡 Puedes copiar este archivo y pegarlo directamente en Claude, GPT-4, u otro LLM\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print(
            """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     FORMATEADOR DE CONTEXTO PARA LLM                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Convierte JSON de contexto en prompts listos para usar con LLMs.

EJEMPLOS DE USO:

1. Formatear contexto RAG:
   python format_for_llm.py --input contexto_rag.json --query "¿Qué es aprendizaje supervisado?"

2. Formatear contexto híbrido:
   python format_for_llm.py --input contexto_hibrido.json --query "Explica algoritmos genéticos"

3. Comparar RAG vs Híbrido:
   python format_for_llm.py --input contexto_hibrido.json --query "..." --compare contexto_rag.json

4. Especificar salida:
   python format_for_llm.py --input contexto.json --query "..." --output mi_prompt.txt

SALIDA:
- Archivo .txt con prompt completo listo para copiar/pegar en LLM
- Incluye instrucciones y formato optimizado para Claude/GPT-4

Para más ayuda: python format_for_llm.py --help
"""
        )
        sys.exit(0)

    main()

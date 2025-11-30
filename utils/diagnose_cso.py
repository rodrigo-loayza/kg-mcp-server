# utils/diagnose_cso.py
"""
🔍 Diagnóstico de CSO - Explora la estructura del archivo
Útil para entender cómo está organizado CSO

Ejecuta:
    python utils/diagnose_cso.py CSO.3.5.ttl
"""

import sys
from pathlib import Path
from collections import Counter

try:
    from rdflib import Graph, Namespace
except ImportError:
    print("❌ rdflib no instalado")
    print("   Instala: pip install rdflib")
    sys.exit(1)


def diagnose_cso(ttl_path):
    """Diagnostica estructura de CSO"""

    print("\n" + "=" * 70)
    print("🔍 DIAGNÓSTICO DE CSO")
    print("=" * 70 + "\n")

    # Cargar grafo
    print(f"📥 Cargando {ttl_path}...")
    graph = Graph()
    graph.parse(ttl_path, format="turtle")
    print(f"   ✅ {len(graph)} triples cargadas\n")

    # Contar predicados (propiedades)
    print("📊 Top 20 Predicados (Propiedades):")
    print("-" * 70)

    predicates = Counter()
    for s, p, o in graph:
        predicates[str(p)] += 1

    for pred, count in predicates.most_common(20):
        # Acortar URIs largas
        if "#" in pred:
            pred_short = pred.split("#")[-1]
        elif "/" in pred:
            pred_short = pred.split("/")[-1]
        else:
            pred_short = pred

        print(f"   {count:6} × {pred_short:40} ({pred})")

    # Buscar namespaces
    print("\n📚 Namespaces Detectados:")
    print("-" * 70)

    namespaces = set()
    for s, p, o in graph:
        for uri in [str(s), str(p)]:
            if "#" in uri:
                ns = uri.split("#")[0] + "#"
                namespaces.add(ns)
            elif uri.count("/") > 2:
                ns = "/".join(uri.split("/")[:-1]) + "/"
                namespaces.add(ns)

    for ns in sorted(namespaces):
        print(f"   • {ns}")

    # Ejemplos de triples
    print("\n📝 Ejemplos de Triples:")
    print("-" * 70)

    cso_schema = Namespace("http://cso.kmi.open.ac.uk/schema/cso#")

    # Buscar ejemplo con prefLabel
    print("\n🏷️  Ejemplos con prefLabel:")
    count = 0
    for s, p, o in graph.triples((None, cso_schema.prefLabel, None)):
        print(f"   {s}")
        print(f"   → prefLabel: {o}")
        count += 1
        if count >= 3:
            break

    # Buscar ejemplo con superTopicOf
    print("\n🔗 Ejemplos con superTopicOf:")
    count = 0
    for s, p, o in graph.triples((None, cso_schema.superTopicOf, None)):
        print(f"   {s}")
        print(f"   → superTopicOf: {o}")
        count += 1
        if count >= 3:
            break

    # Buscar ejemplo con relatedEquivalent
    print("\n🔗 Ejemplos con relatedEquivalent:")
    count = 0
    for s, p, o in graph.triples((None, cso_schema.relatedEquivalent, None)):
        print(f"   {s}")
        print(f"   → relatedEquivalent: {o}")
        count += 1
        if count >= 3:
            break

    # Estadísticas por propiedad CSO
    print("\n📈 Estadísticas CSO:")
    print("-" * 70)

    props = {
        "prefLabel": cso_schema.prefLabel,
        "superTopicOf": cso_schema.superTopicOf,
        "relatedEquivalent": cso_schema.relatedEquivalent,
        "contributesTo": cso_schema.contributesTo,
    }

    for name, prop in props.items():
        count = sum(1 for _ in graph.triples((None, prop, None)))
        print(f"   {name:20} {count:6} ocurrencias")

    # Contar topics únicos
    print("\n🎯 Topics Únicos:")
    print("-" * 70)

    topics = set()
    for s, p, o in graph.triples((None, cso_schema.prefLabel, None)):
        topics.add(str(s))

    print(f"   Total: {len(topics)} topics")

    # Mostrar algunos ejemplos
    print("\n   Ejemplos de URIs de topics:")
    for topic_uri in list(topics)[:5]:
        print(f"      • {topic_uri}")

    print("\n" + "=" * 70)
    print("✅ DIAGNÓSTICO COMPLETADO")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Uso: python diagnose_cso.py <archivo.ttl>")
        print("\nEjemplo:")
        print("  python diagnose_cso.py CSO.3.5.ttl")
        sys.exit(1)

    ttl_path = sys.argv[1]

    if not Path(ttl_path).exists():
        print(f"❌ Archivo no encontrado: {ttl_path}")
        sys.exit(1)

    diagnose_cso(ttl_path)


if __name__ == "__main__":
    main()

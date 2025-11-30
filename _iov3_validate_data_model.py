#!/usr/bin/env python3
"""
IOV3: Validación del Modelo de Datos del Grafo de Conocimiento

Objetivo:
    Validar que el modelo de datos especificado cumple con:
    - ≥5 tipos de nodos diferentes
    - ≥5 tipos de relaciones diferentes
    - Ejemplos de instanciación con documentos académicos reales

Validación:
    - Cuenta tipos de nodos y relaciones en Neo4j
    - Extrae ejemplos representativos de cada tipo
    - Genera reporte para documentación de tesis

Salida:
    JSON con especificación completa del modelo de datos y ejemplos

Uso:
    python _iov3_validate_data_model.py --output results/iov3_data_model.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase


def validate_data_model(uri="bolt://localhost:7687", user="neo4j", password="password"):
    """Valida el modelo de datos del grafo"""

    print("=" * 70)
    print("📊 IOV3: VALIDACIÓN DEL MODELO DE DATOS")
    print("=" * 70)
    print()

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # 1. Contar tipos de nodos
        print("🔍 Tipos de Nodos...")
        node_types_query = """
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        ORDER BY count DESC
        """
        node_types = list(session.run(node_types_query))

        print(f"  Total tipos de nodos: {len(node_types)}\n")
        for record in node_types:
            print(f"  • {record['node_type']}: {record['count']} instancias")
        print()

        # 2. Contar tipos de relaciones
        print("🔗 Tipos de Relaciones...")
        rel_types_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        """
        rel_types = list(session.run(rel_types_query))

        print(f"  Total tipos de relaciones: {len(rel_types)}\n")
        for record in rel_types:
            print(f"  • {record['rel_type']}: {record['count']} instancias")
        print()

        # 3. Extraer ejemplos representativos
        examples = {}

        # Ejemplo: Document
        print("📄 Extrayendo ejemplos de nodos...")
        doc_example = session.run(
            """
        MATCH (d:Document)
        RETURN d.name as name, d.format as format, d.num_chunks as num_chunks
        LIMIT 3
        """
        ).data()
        examples["Document"] = doc_example

        # Ejemplo: Chunk
        chunk_example = session.run(
            """
        MATCH (c:Chunk)
        RETURN c.chunk_id as chunk_id, c.text as text, c.embedding_dim as embedding_dim
        LIMIT 2
        """
        ).data()
        # Truncar texto
        for ex in chunk_example:
            if ex["text"]:
                ex["text"] = ex["text"][:150] + "..." if len(ex["text"]) > 150 else ex["text"]
        examples["Chunk"] = chunk_example

        # Ejemplo: Topic
        topic_example = session.run(
            """
        MATCH (t:Topic)
        RETURN t.name as name
        LIMIT 3
        """
        ).data()
        examples["Topic"] = topic_example

        # Ejemplo: Concept
        concept_example = session.run(
            """
        MATCH (c:Concept)
        RETURN c.name as name, c.uri as uri
        LIMIT 3
        """
        ).data()
        examples["Concept"] = concept_example

        # Ejemplo: Algorithm
        algo_example = session.run(
            """
        MATCH (a:Algorithm)
        RETURN a.name as name
        LIMIT 3
        """
        ).data()
        examples["Algorithm"] = algo_example

        # Ejemplo: Course
        course_example = session.run(
            """
        MATCH (c:Course)
        RETURN c.code as code, c.name as name
        LIMIT 1
        """
        ).data()
        examples["Course"] = course_example

        # 4. Ejemplos de relaciones
        print("🔗 Extrayendo ejemplos de relaciones...")
        rel_examples = {}

        # CONTAINS
        contains_ex = session.run(
            """
        MATCH (d:Document)-[r:CONTAINS]->(c:Chunk)
        RETURN d.name as document, c.chunk_id as chunk
        LIMIT 2
        """
        ).data()
        rel_examples["CONTAINS"] = contains_ex

        # PREREQUISITE_OF
        prereq_ex = session.run(
            """
        MATCH (a1:Algorithm)-[r:PREREQUISITE_OF]->(a2:Algorithm)
        RETURN a1.name as prerequisite, a2.name as target
        LIMIT 2
        """
        ).data()
        rel_examples["PREREQUISITE_OF"] = prereq_ex

        # TEACHES
        teaches_ex = session.run(
            """
        MATCH (t:Topic)-[r:TEACHES]->(a:Algorithm)
        RETURN t.name as topic, a.name as algorithm
        LIMIT 2
        """
        ).data()
        rel_examples["TEACHES"] = teaches_ex

        # RELATED_TO
        related_ex = session.run(
            """
        MATCH (c:Concept)-[r:RELATED_TO]->(t:Topic)
        RETURN c.name as concept, t.name as topic
        LIMIT 2
        """
        ).data()
        rel_examples["RELATED_TO"] = related_ex

        print("  ✓ Ejemplos extraídos\n")

    driver.close()

    # 5. Validar IOV3
    num_node_types = len(node_types)
    num_rel_types = len(rel_types)

    iov3_node_types_passed = num_node_types >= 5
    iov3_rel_types_passed = num_rel_types >= 5
    iov3_has_examples = len(examples) > 0 and len(rel_examples) > 0

    print("=" * 70)
    print("📋 VALIDACIÓN IOV3")
    print("=" * 70)
    print()
    print(f"  • Tipos de nodos: {num_node_types} (requisito: ≥5)")
    print(f"    IOV3.1: {'✅ CUMPLIDO' if iov3_node_types_passed else '❌ NO CUMPLIDO'}")
    print()
    print(f"  • Tipos de relaciones: {num_rel_types} (requisito: ≥5)")
    print(f"    IOV3.2: {'✅ CUMPLIDO' if iov3_rel_types_passed else '❌ NO CUMPLIDO'}")
    print()
    print(f"  • Ejemplos de instanciación: {'Sí' if iov3_has_examples else 'No'}")
    print(f"    IOV3.3: {'✅ CUMPLIDO' if iov3_has_examples else '❌ NO CUMPLIDO'}")
    print()
    print("=" * 70)
    print()

    # 6. Generar reporte JSON
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "iov3_criteria": {
            "node_types": {
                "requirement": "≥5 tipos de nodos diferentes",
                "achieved": num_node_types,
                "passed": iov3_node_types_passed,
            },
            "relationship_types": {
                "requirement": "≥5 tipos de relaciones diferentes",
                "achieved": num_rel_types,
                "passed": iov3_rel_types_passed,
            },
            "has_examples": {
                "requirement": "Ejemplos de instanciación con documentos académicos reales",
                "achieved": "Sí" if iov3_has_examples else "No",
                "passed": iov3_has_examples,
            },
        },
        "data_model": {
            "node_types": [
                {"type": record["node_type"], "count": record["count"]} for record in node_types
            ],
            "relationship_types": [
                {"type": record["rel_type"], "count": record["count"]} for record in rel_types
            ],
        },
        "examples": {
            "nodes": examples,
            "relationships": rel_examples,
        },
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Validar IOV3: Modelo de datos del grafo")
    parser.add_argument(
        "--output",
        default="results/iov3_data_model.json",
        help="Archivo de salida JSON",
    )
    parser.add_argument("--uri", default="bolt://localhost:7687", help="URI de Neo4j")
    parser.add_argument("--user", default="neo4j", help="Usuario de Neo4j")
    parser.add_argument("--password", default="password", help="Contraseña de Neo4j")

    args = parser.parse_args()

    # Validar modelo
    report = validate_data_model(args.uri, args.user, args.password)

    # Guardar reporte
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"💾 Reporte guardado en: {output_path}")
    print()


if __name__ == "__main__":
    main()

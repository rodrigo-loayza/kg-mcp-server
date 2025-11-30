# kg_builder.py
"""
🗄️ Knowledge Graph Builder - Constructor del Grafo en Neo4j
Consolida instantiate_kg.py + populate_neo4j.py

Responsabilidades:
- Cargar datos desde JSON
- Merge de múltiples fuentes
- Crear constraints en Neo4j
- Insertar nodos y relaciones
- Verificar integridad del grafo

Autor: Rodrigo Cárdenas
"""

import json
from pathlib import Path
from neo4j import GraphDatabase
from typing import List, Dict, Optional
from models.academic_kg_model import NodeType, RelationType, create_cypher_schema


class KGBuilder:
    """Constructor del Knowledge Graph en Neo4j"""

    def __init__(self, uri: str, user: str, password: str):
        """
        Inicializa builder con conexión a Neo4j

        Args:
            uri: URI de Neo4j (ej: bolt://localhost:7687)
            user: Usuario de Neo4j
            password: Contraseña de Neo4j
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.stats = {
            "total_nodes": 0,
            "total_relations": 0,
            "nodes_by_type": {},
            "relations_by_type": {},
        }

    def close(self):
        """Cierra conexión a Neo4j"""
        self.driver.close()

    def clear_database(self):
        """Limpia toda la base de datos Neo4j"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ Base de datos limpiada")

    def create_constraints(self):
        """Crea constraints de unicidad para todos los tipos de nodos"""
        print("\n🔒 Creando constraints...")

        with self.driver.session() as session:
            for constraint in create_cypher_schema():
                try:
                    session.run(constraint)
                    print(f"   ✅ Constraint creado")
                except Exception as e:
                    # Constraint ya existe
                    print(f"   ⚠️  Constraint ya existe")

    def load_from_json(self, json_path: Path) -> Dict:
        """
        Carga datos desde archivo JSON

        Args:
            json_path: Ruta al archivo JSON

        Returns:
            Dict con nodes y relations
        """
        print(f"\n📂 Cargando: {json_path.name}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"   Nodos: {len(data.get('nodes', []))}")
        print(f"   Relaciones: {len(data.get('relations', []))}")

        return data

    def merge_datasets(self, datasets: List[Dict]) -> Dict:
        """
        Combina múltiples datasets eliminando duplicados

        Args:
            datasets: Lista de dicts con nodes y relations

        Returns:
            Dict merged con nodes y relations únicos
        """
        print("\n🔀 Mergeando datasets...")

        merged_nodes = {}
        merged_relations = set()

        # Merge nodos (por ID único)
        for dataset in datasets:
            for node in dataset.get("nodes", []):
                node_id = node["id"]
                if node_id not in merged_nodes:
                    merged_nodes[node_id] = node
                else:
                    # Actualizar propiedades si existen nuevas
                    merged_nodes[node_id].update(node)

        # Merge relaciones (evitar duplicados)
        for dataset in datasets:
            for rel in dataset.get("relations", []):
                # Crear tupla única para relación
                rel_tuple = (rel["from"], rel["to"], rel["type"])
                # Usar JSON para guardar relación completa
                merged_relations.add(json.dumps(rel, sort_keys=True))

        merged_data = {
            "nodes": list(merged_nodes.values()),
            "relations": [json.loads(r) for r in merged_relations],
        }

        print(f"   ✅ Nodos merged: {len(merged_data['nodes'])}")
        print(f"   ✅ Relaciones merged: {len(merged_data['relations'])}")

        return merged_data

    def instantiate_nodes(self, nodes: List[Dict]):
        """
        Crea nodos en Neo4j usando MERGE

        Args:
            nodes: Lista de nodos a crear
        """
        print("\n📝 Insertando nodos...")

        node_counts = {}

        with self.driver.session() as session:
            for node in nodes:
                node_type = node.get("type", "Unknown")

                # Validar tipo de nodo (solo advertencia)
                valid_types = [nt.value for nt in NodeType]
                if node_type not in valid_types:
                    print(
                        f"   ⚠️  Tipo de nodo no estándar: {node_type} (insertando de todos modos)"
                    )

                # Aplanar properties (Neo4j no acepta dicts anidados)
                props = {}
                for key, value in node.items():
                    if key == "type":
                        continue
                    elif key == "properties" and isinstance(value, dict):
                        # Aplanar dict de properties
                        props.update(value)
                    else:
                        props[key] = value

                # Asegurar que id está en props
                if "id" not in props:
                    props["id"] = node.get("id")

                # Construir query con MERGE (evita duplicados)
                prop_assignments = ", ".join([f"{key}: ${key}" for key in props.keys()])

                query = f"""
                MERGE (n:{node_type} {{id: $id}})
                ON CREATE SET n += {{{prop_assignments}}}
                ON MATCH SET n += {{{prop_assignments}}}
                """

                try:
                    session.run(query, **props)
                    node_counts[node_type] = node_counts.get(node_type, 0) + 1
                except Exception as e:
                    print(f"   ❌ Error creando nodo {node.get('id')}: {e}")

        # Actualizar stats
        self.stats["nodes_by_type"] = node_counts
        self.stats["total_nodes"] = sum(node_counts.values())

        print("\n📊 Nodos creados por tipo:")
        for node_type, count in sorted(node_counts.items()):
            print(f"   {node_type:15} {count:3} nodos")

    def instantiate_relations(self, relations: List[Dict]):
        """
        Crea relaciones en Neo4j

        Args:
            relations: Lista de relaciones a crear
        """
        print("\n🔗 Creando relaciones...")

        rel_counts = {}

        with self.driver.session() as session:
            for rel in relations:
                rel_type = rel.get("type", "UNKNOWN")

                # Extraer propiedades
                props = {k: v for k, v in rel.items() if k not in ["from", "to", "type"]}

                # Construir SET clause para propiedades
                if props:
                    prop_assignments = ", ".join([f"r.{key} = ${key}" for key in props.keys()])
                    set_clause = f"SET {prop_assignments}"
                else:
                    set_clause = ""

                query = f"""
                MATCH (a {{id: $from_id}})
                MATCH (b {{id: $to_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                {set_clause}
                """

                try:
                    session.run(query, from_id=rel["from"], to_id=rel["to"], **props)
                    rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
                except Exception as e:
                    # Silenciar errores de nodos no encontrados (puede pasar en merge)
                    pass

        # Actualizar stats
        self.stats["relations_by_type"] = rel_counts
        self.stats["total_relations"] = sum(rel_counts.values())

        print("\n📊 Relaciones creadas por tipo:")
        for rel_type, count in sorted(rel_counts.items()):
            print(f"   {rel_type:20} {count:3} relaciones")

    def verify_instantiation(self):
        """Verifica la instanciación del grafo"""
        print("\n" + "=" * 70)
        print("🔍 VERIFICACIÓN DEL GRAFO")
        print("=" * 70)

        with self.driver.session() as session:
            # Contar nodos por tipo
            result = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] as type, count(n) as count
                ORDER BY count DESC
                """
            )

            print("\n📊 Nodos por tipo:")
            for record in result:
                print(f"   {record['type']:15} {record['count']:3} nodos")

            # Contar relaciones por tipo
            result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
                """
            )

            print("\n🔗 Relaciones por tipo:")
            for record in result:
                print(f"   {record['rel_type']:20} {record['count']:3} relaciones")

            # Estadísticas generales
            result = session.run("MATCH (n) RETURN count(n) as total_nodes")
            total_nodes = result.single()["total_nodes"]

            result = session.run("MATCH ()-[r]->() RETURN count(r) as total_rels")
            total_rels = result.single()["total_rels"]

            print("\n" + "=" * 70)
            print(f"📈 TOTAL: {total_nodes} nodos, {total_rels} relaciones")
            print("=" * 70)

    def build_from_multiple_sources(self, sources: List[Path]):
        """
        Pipeline completo: carga múltiples fuentes y construye el KG

        Args:
            sources: Lista de rutas a archivos JSON
        """
        print("\n" + "=" * 70)
        print("🗂️ CONSTRUCCIÓN DE KNOWLEDGE GRAPH ACADÉMICO")
        print("=" * 70)

        # 1. Cargar datasets
        datasets = []
        for source in sources:
            if source.exists():
                data = self.load_from_json(source)
                datasets.append(data)
            else:
                print(f"   ⚠️  Archivo no encontrado: {source}")

        if not datasets:
            print("❌ No se cargaron datasets")
            return

        # 2. Merge datasets
        merged_data = self.merge_datasets(datasets)

        # 3. Limpiar base de datos
        print("\n🗑️  Limpiando base de datos...")
        self.clear_database()

        # 4. Crear constraints
        self.create_constraints()

        # 5. Insertar nodos
        self.instantiate_nodes(merged_data["nodes"])

        # 6. Crear relaciones
        self.instantiate_relations(merged_data["relations"])

        # 7. Verificar
        self.verify_instantiation()

        print("\n" + "=" * 70)
        print("✅ KNOWLEDGE GRAPH CONSTRUIDO EXITOSAMENTE")
        print("=" * 70)
        print("\n🌐 Abre Neo4j Browser:")
        print("   http://localhost:7474")
        print("\n💡 Consultas de ejemplo:")
        print("   MATCH (c:Course)-[r:TEACHES]->(t:Topic) RETURN c, r, t LIMIT 25")
        print("   MATCH (d:Document)-[r:CONTAINS]->(c:Concept) RETURN d, r, c LIMIT 25")
        print("   MATCH (n) RETURN n LIMIT 100")


def main():
    """Script standalone para poblar Neo4j desde JSON"""
    import sys

    if len(sys.argv) < 2:
        print("Uso: python kg_builder.py <json_file1> [json_file2] ...")
        print("\nEjemplo:")
        print("  python kg_builder.py kg_from_syllabi.json kg_from_documents.json")
        sys.exit(1)

    # Archivos de entrada
    sources = [Path(arg) for arg in sys.argv[1:]]

    # Configuración Neo4j
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"

    # Crear builder
    builder = KGBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # Construir grafo
        builder.build_from_multiple_sources(sources)
    finally:
        builder.close()


if __name__ == "__main__":
    main()

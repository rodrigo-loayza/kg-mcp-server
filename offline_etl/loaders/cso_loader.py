# offline_etl/loaders/cso_loader.py
"""
🎓 CSO Loader - Computer Science Ontology Integration
Carga y procesa CSO (Computer Science Ontology) para enriquecer el Knowledge Graph

Funcionalidades:
- Carga CSO.ttl (formato RDF/Turtle)
- Mapeo a CS2023 Knowledge Areas
- Búsqueda de topics por nombre (fuzzy matching)
- Extracción de jerarquías (broader/narrower)
- Generación de prerequisitos automáticos

Autor: Rodrigo Cárdenas
Basado en trabajo previo de integración CSO
"""

from rdflib import Graph, Namespace, RDF, RDFS, SKOS
import json
from pathlib import Path
from typing import List, Dict, Set, Optional
from collections import defaultdict
from difflib import get_close_matches


class CSOLoader:
    """
    Cargador de Computer Science Ontology (CSO)
    Integrado con el modelo de KG Académico
    """

    def __init__(self, ttl_path: str = "CSO.3.5.ttl"):
        """
        Args:
            ttl_path: Ruta al archivo CSO en formato Turtle
        """
        self.ttl_path = Path(ttl_path)
        self.graph = Graph()

        # Namespaces RDF
        self.cso_ns = Namespace("http://cso.kmi.open.ac.uk/schema/cso#")
        self.skos_ns = SKOS

        # Datos extraídos
        self.topics = {}  # {uri: {id, label, broader, narrower, related}}
        self.topics_by_label = {}  # {label_lower: uri} para búsqueda rápida

        # Mapeo CSO → CS2023 Knowledge Areas
        # IDs de CSO son como: "machine_learning", "artificial_intelligence"
        # (extraídos de URIs como http://cso.kmi.open.ac.uk/topics/machine_learning)
        self.CSO_TO_CS2023 = {
            # AI & ML
            "artificial_intelligence": "AI - Artificial Intelligence",
            "machine_learning": "AI - Machine Learning",
            "deep_learning": "AI - Deep Learning",
            "neural_network": "AI - Neural Networks",
            "robotics": "AI - Robotics",
            "computer_vision": "AI - Computer Vision",
            "natural_language_processing": "AI - Natural Language Processing",
            "speech_recognition": "AI - Speech Recognition",
            "pattern_recognition": "AI - Pattern Recognition",
            "data_mining": "AI - Data Mining",
            # Algorithms & Theory
            "algorithm": "AL - Algorithmic Foundations",
            "data_structure": "AL - Data Structures",
            "complexity": "AL - Computational Complexity",
            "graph_theory": "AL - Graph Theory",
            "optimization": "AL - Optimization",
            "computational_complexity": "AL - Computational Complexity",
            "approximation_algorithm": "AL - Approximation Algorithms",
            # Software Engineering
            "software_engineering": "SE - Software Engineering",
            "software_development": "SE - Software Development",
            "software_testing": "SE - Software Testing",
            "design_pattern": "SE - Design Patterns",
            "agile_software_development": "SE - Agile Development",
            "software_architecture": "SE - Software Architecture",
            # Systems
            "operating_system": "OS - Operating Systems",
            "distributed_system": "OS - Distributed Systems",
            "parallel_computing": "OS - Parallel Computing",
            "concurrent_computing": "OS - Concurrent Computing",
            "cloud_computing": "OS - Cloud Computing",
            # Networks & Security
            "computer_network": "NC - Networking and Communication",
            "network_protocol": "NC - Network Protocols",
            "computer_security": "SEC - Security",
            "cryptography": "SEC - Cryptography",
            "information_security": "SEC - Information Security",
            "network_security": "SEC - Network Security",
            # Data & Databases
            "database": "DM - Data Management",
            "relational_database": "DM - Relational Databases",
            "information_retrieval": "DM - Information Retrieval",
            "big_data": "DM - Big Data",
            "database_management_system": "DM - Database Management Systems",
            # HCI
            "human_computer_interaction": "HCI - Human-Computer Interaction",
            "user_interface": "HCI - User Interface Design",
            "usability": "HCI - Usability",
            # Programming
            "programming": "SDF - Software Development Fundamentals",
            "programming_language": "SDF - Programming Languages",
            "object_oriented_programming": "SDF - Object-Oriented Programming",
            "functional_programming": "SDF - Functional Programming",
            # Hardware & Architecture
            "computer_architecture": "AR - Architecture and Organization",
            "computer_hardware": "AR - Computer Hardware",
            "parallel_architecture": "AR - Parallel Architecture",
            # Web & Internet
            "world_wide_web": "NC - Web Technologies",
            "web_service": "NC - Web Services",
            "semantic_web": "NC - Semantic Web",
        }

        self.loaded = False

    def load(self, verbose: bool = True) -> bool:
        """
        Carga y parsea el archivo CSO

        Args:
            verbose: Mostrar progreso

        Returns:
            True si carga exitosa
        """
        if not self.ttl_path.exists():
            print(f"❌ Archivo no encontrado: {self.ttl_path}")
            print(f"   Descarga CSO de: https://cso.kmi.open.ac.uk/download")
            return False

        if verbose:
            print(f"📥 Cargando CSO desde: {self.ttl_path}")

        try:
            # Parsear TTL
            self.graph.parse(self.ttl_path, format="turtle")

            if verbose:
                print(f"   ✅ {len(self.graph)} triples RDF cargadas")

            # Extraer datos
            self._extract_topics(verbose)
            self._extract_relations(verbose)

            self.loaded = True
            return True

        except Exception as e:
            print(f"❌ Error cargando CSO: {e}")
            return False

    def _extract_topics(self, verbose: bool = True):
        """Extrae todos los topics con sus labels"""
        if verbose:
            print(f"🔍 Extrayendo topics...")

        count = 0

        # CSO usa rdfs:label (NO cso:prefLabel)
        # Buscar todos los topics que tienen rdfs:label
        for s, p, o in self.graph.triples((None, RDFS.label, None)):
            topic_uri = str(s)
            label = str(o)

            # Solo procesar topics de CSO (no otros recursos)
            if "cso.kmi.open.ac.uk/topics/" not in topic_uri:
                continue

            # El ID es la última parte de la URI
            if "/" in topic_uri:
                topic_id = topic_uri.split("/")[-1]
            else:
                topic_id = topic_uri.split("#")[-1]

            self.topics[topic_uri] = {
                "id": topic_id,
                "label": label,
                "uri": topic_uri,
                "broader": [],
                "narrower": [],
                "related": [],
                "contributes": [],  # contributesTo relations
            }

            # Índice por label (lowercase para búsqueda)
            self.topics_by_label[label.lower()] = topic_uri
            count += 1

        if verbose:
            print(f"   ✅ {count} topics extraídos")

    def _extract_relations(self, verbose: bool = True):
        """Extrae relaciones CSO (superTopicOf, relatedEquivalent, contributesTo)"""
        if verbose:
            print(f"🔗 Extrayendo relaciones...")

        from rdflib import Namespace

        cso_schema = Namespace("http://cso.kmi.open.ac.uk/schema/cso#")

        broader_count = 0
        narrower_count = 0
        related_count = 0
        contributes_count = 0

        # CSO usa superTopicOf (no SKOS broader/narrower)
        # Si A superTopicOf B, entonces A es broader que B (más general)
        for s, p, o in self.graph.triples((None, cso_schema.superTopicOf, None)):
            parent_uri = str(s)  # A (más general)
            child_uri = str(o)  # B (más específico)

            if parent_uri in self.topics and child_uri in self.topics:
                # Parent es broader (más general) del child
                self.topics[child_uri]["broader"].append(parent_uri)
                broader_count += 1

                # Child es narrower (más específico) del parent
                self.topics[parent_uri]["narrower"].append(child_uri)
                narrower_count += 1

        # Relaciones related (relatedEquivalent en CSO)
        for s, p, o in self.graph.triples((None, cso_schema.relatedEquivalent, None)):
            s_uri = str(s)
            o_uri = str(o)
            if s_uri in self.topics and o_uri in self.topics:
                self.topics[s_uri]["related"].append(o_uri)
                related_count += 1

        # Relaciones contributesTo (más común: 48,980 ocurrencias!)
        # A contributesTo B significa que A contribuye/ayuda a resolver B
        for s, p, o in self.graph.triples((None, cso_schema.contributesTo, None)):
            from_uri = str(s)
            to_uri = str(o)
            if from_uri in self.topics and to_uri in self.topics:
                # Guardar en lista separada para procesamiento especial
                if "contributes" not in self.topics[from_uri]:
                    self.topics[from_uri]["contributes"] = []
                self.topics[from_uri]["contributes"].append(to_uri)
                contributes_count += 1

        if verbose:
            print(
                f"   ✅ {broader_count} broader, {narrower_count} narrower, {related_count} related, {contributes_count} contributes"
            )

    def find_topic(self, query: str, threshold: float = 0.6) -> Dict:
        """
        Busca un topic por nombre (fuzzy matching)

        Args:
            query: Nombre del topic a buscar
            threshold: Umbral de similitud (0-1)

        Returns:
            Dict con formato: {"found": bool, "uri": str, "label": str, "similarity": float}
        """
        if not self.loaded:
            return {"found": False}

        query_lower = query.lower()

        # Búsqueda exacta
        if query_lower in self.topics_by_label:
            uri = self.topics_by_label[query_lower]
            topic_data = self.topics[uri]
            return {
                "found": True,
                "uri": uri,
                "label": topic_data.get("label", query),
                "id": topic_data.get("id", ""),
                "similarity": 1.0,
            }

        # Fuzzy matching
        matches = get_close_matches(query_lower, self.topics_by_label.keys(), n=1, cutoff=threshold)

        if matches:
            matched_label = matches[0]
            uri = self.topics_by_label[matched_label]
            topic_data = self.topics[uri]

            # Calcular similitud real
            from difflib import SequenceMatcher

            similarity = SequenceMatcher(None, query_lower, matched_label).ratio()

            return {
                "found": True,
                "uri": uri,
                "label": topic_data.get("label", matched_label),
                "id": topic_data.get("id", ""),
                "similarity": similarity,
            }

        return {"found": False}

    def get_prerequisites(self, topic_uri: str, max_depth: int = 2) -> List[str]:
        """
        Obtiene prerequisitos de un topic (conceptos broader)

        Args:
            topic_uri: URI del topic
            max_depth: Profundidad máxima de búsqueda

        Returns:
            Lista de URIs de prerequisitos
        """
        if topic_uri not in self.topics:
            return []

        prerequisites = set()
        current_level = {topic_uri}

        for depth in range(max_depth):
            next_level = set()
            for uri in current_level:
                if uri in self.topics:
                    broader = self.topics[uri]["broader"]
                    for b_uri in broader:
                        if b_uri not in prerequisites:
                            prerequisites.add(b_uri)
                            next_level.add(b_uri)

            current_level = next_level
            if not current_level:
                break

        return list(prerequisites)

    def get_subtopics(self, topic_uri: str, max_depth: int = 1) -> List[str]:
        """
        Obtiene subtopics de un topic (conceptos narrower)

        Args:
            topic_uri: URI del topic
            max_depth: Profundidad máxima

        Returns:
            Lista de URIs de subtopics
        """
        if topic_uri not in self.topics:
            return []

        subtopics = set()
        current_level = {topic_uri}

        for depth in range(max_depth):
            next_level = set()
            for uri in current_level:
                if uri in self.topics:
                    narrower = self.topics[uri]["narrower"]
                    for n_uri in narrower:
                        if n_uri not in subtopics:
                            subtopics.add(n_uri)
                            next_level.add(n_uri)

            current_level = next_level
            if not current_level:
                break

        return list(subtopics)

    def get_related_topics(self, topic_uri: str) -> List[str]:
        """
        Obtiene topics relacionados

        Args:
            topic_uri: URI del topic

        Returns:
            Lista de URIs relacionados
        """
        if topic_uri not in self.topics:
            return []

        return self.topics[topic_uri]["related"]

    def enrich_topic(self, topic_name: str) -> Dict:
        """
        Enriquece un topic del sílabo con info de CSO

        Args:
            topic_name: Nombre del topic (ej: "Machine Learning")

        Returns:
            Dict con info enriquecida
        """
        enrichment = {
            "found_in_cso": False,
            "cso_uri": None,
            "cso_id": None,
            "cs2023_area": None,
            "prerequisites": [],
            "subtopics": [],
            "related": [],
        }

        # Buscar en CSO
        cso_topic = self.find_topic(topic_name)

        if not cso_topic:
            return enrichment

        enrichment["found_in_cso"] = True
        enrichment["cso_uri"] = cso_topic["uri"]
        enrichment["cso_id"] = cso_topic["id"]

        # Mapear a CS2023
        topic_id = cso_topic["id"]
        if topic_id in self.CSO_TO_CS2023:
            enrichment["cs2023_area"] = self.CSO_TO_CS2023[topic_id]

        # Obtener relaciones
        uri = cso_topic["uri"]
        enrichment["prerequisites"] = self.get_prerequisites(uri, max_depth=2)
        enrichment["subtopics"] = self.get_subtopics(uri, max_depth=1)
        enrichment["related"] = self.get_related_topics(uri)

        return enrichment

    def extract_cs2023_aligned_topics(self, depth: int = 1) -> Dict:
        """
        Extrae topics alineados con CS2023 Knowledge Areas

        Args:
            depth: Profundidad de subtree (1=conservador, 2=completo)

        Returns:
            Dict con nodes y relations para Neo4j
        """
        print("\n" + "=" * 70)
        print("🎓 EXTRACCIÓN CSO ALINEADA CON CS2023")
        print(f"   Profundidad: {depth}")
        print("=" * 70 + "\n")

        # Buscar topics raíz
        root_topics = {}
        for cso_id, cs2023_area in self.CSO_TO_CS2023.items():
            # Buscar URI del topic
            for uri, data in self.topics.items():
                if data["id"] == cso_id:
                    root_topics[cso_id] = {
                        "uri": uri,
                        "cs2023_area": cs2023_area,
                        "label": data["label"],
                    }
                    break

        print(f"📚 Topics raíz encontrados: {len(root_topics)}\n")
        for cso_id, data in sorted(root_topics.items()):
            print(f"   ✓ {cso_id:40} → {data['cs2023_area']}")

        # Extraer subtrees
        all_topic_uris = set()
        coverage = defaultdict(int)

        for cso_id, data in root_topics.items():
            # Agregar root
            uri = data["uri"]
            all_topic_uris.add(uri)
            coverage[data["cs2023_area"]] += 1

            # Agregar subtopics
            subtopics = self.get_subtopics(uri, max_depth=depth)
            all_topic_uris.update(subtopics)
            coverage[data["cs2023_area"]] += len(subtopics)

        print(f"\n📊 Cobertura por Knowledge Area:")
        for ka, count in sorted(coverage.items()):
            print(f"   {ka}: {count} topics")

        print(f"\n🎯 TOTAL: {len(all_topic_uris)} topics extraídos")

        # Convertir a nodos y relaciones
        nodes, relations = self._topics_to_kg(all_topic_uris)

        return {
            "nodes": nodes,
            "relations": relations,
            "metadata": {
                "source": "Computer Science Ontology v3.5",
                "alignment": "CS2023 Knowledge Areas",
                "depth": depth,
                "total_topics": len(all_topic_uris),
            },
        }

    def _topics_to_kg(self, topic_uris: Set[str]) -> tuple:
        """
        Convierte topics CSO a nodos y relaciones del KG
        Versión mejorada con clasificación de tipos y relaciones inteligentes
        """
        nodes = []
        relations = []
        uri_to_data = {}  # {uri: (node_id, node_type, label)}

        # Crear nodos con clasificación de tipo
        for uri in topic_uris:
            if uri not in self.topics:
                continue

            topic_data = self.topics[uri]
            label = topic_data["label"]

            # Clasificar tipo de nodo
            node_type = self._classify_topic(label)
            node_id = f"{node_type.lower()}_{topic_data['id']}"

            node = {
                "type": node_type,
                "id": node_id,
                "name": label,
                "source": "cso",
                "cso_uri": uri,
            }

            nodes.append(node)
            uri_to_data[uri] = (node_id, node_type, label)

        # Crear relaciones inteligentes
        for uri in topic_uris:
            if uri not in self.topics:
                continue

            topic_data = self.topics[uri]
            from_id, from_type, from_label = uri_to_data[uri]

            # 1. Relaciones broader → REQUIRES o EXTENDS
            for broader_uri in topic_data["broader"]:
                if broader_uri in uri_to_data:
                    to_id, to_type, to_label = uri_to_data[broader_uri]

                    # Si son del mismo tipo → EXTENDS (especialización)
                    if from_type == to_type and from_type in ["Algorithm", "DataStructure"]:
                        rel_type = "EXTENDS"
                    else:
                        # Diferente tipo → REQUIRES (prerequisito)
                        rel_type = "REQUIRES"

                    relations.append(
                        {
                            "from": from_id,
                            "to": to_id,
                            "type": rel_type,
                            "source": "cso_hierarchy",
                            "strength": 0.8,
                        }
                    )

            # 2. Relaciones related → RELATED_TO
            for related_uri in topic_data["related"]:
                if related_uri in uri_to_data:
                    to_id, to_type, to_label = uri_to_data[related_uri]

                    relations.append(
                        {
                            "from": from_id,
                            "to": to_id,
                            "type": "RELATED_TO",
                            "source": "cso",
                            "strength": 0.6,
                        }
                    )

            # 3. Relaciones contributesTo → SOLVES o EXEMPLIFIES
            if "contributes" in topic_data:
                for contributes_uri in topic_data["contributes"]:
                    if contributes_uri in uri_to_data:
                        to_id, to_type, to_label = uri_to_data[contributes_uri]

                        # Algoritmo/DataStructure contribuye a Problema → SOLVES
                        if from_type in ["Algorithm", "DataStructure"] and to_type == "Problem":
                            relations.append(
                                {
                                    "from": from_id,
                                    "to": to_id,
                                    "type": "SOLVES",
                                    "source": "cso_contributes",
                                    "efficiency": 0.85,
                                }
                            )

                        # Algoritmo/DataStructure contribuye a Concepto → EXEMPLIFIES
                        elif from_type in ["Algorithm", "DataStructure"] and to_type == "Concept":
                            relations.append(
                                {
                                    "from": from_id,
                                    "to": to_id,
                                    "type": "EXEMPLIFIES",
                                    "source": "cso_contributes",
                                    "clarity_score": 0.8,
                                }
                            )

                        # Otro caso → CONTRIBUTES_TO (genérico)
                        else:
                            relations.append(
                                {
                                    "from": from_id,
                                    "to": to_id,
                                    "type": "CONTRIBUTES_TO",
                                    "source": "cso_contributes",
                                }
                            )

        # Estadísticas de clasificación
        node_types = defaultdict(int)
        for node in nodes:
            node_types[node["type"]] += 1

        rel_types = defaultdict(int)
        for rel in relations:
            rel_types[rel["type"]] += 1

        print(f"\n📊 Distribución de nodos:")
        for ntype, count in sorted(node_types.items()):
            print(f"   {ntype:20} {count:3} nodos")

        print(f"\n📈 Distribución de relaciones:")
        for rtype, count in sorted(rel_types.items()):
            print(f"   {rtype:20} {count:3} relaciones")

        return nodes, relations

    def _classify_topic(self, label: str) -> str:
        """
        Clasifica un topic CSO en tipo de nodo del modelo académico

        Returns:
            Uno de: Algorithm, DataStructure, Problem, Concept
        """
        label_lower = label.lower()

        # Palabras clave para clasificación
        algorithm_keywords = [
            "algorithm",
            "sort",
            "search",
            "traversal",
            "optimization",
            "sorting",
            "searching",
            "procedure",
            "method",
        ]

        structure_keywords = [
            "tree",
            "list",
            "array",
            "stack",
            "queue",
            "heap",
            "graph",
            "hash",
            "structure",
            "table",
            "index",
            "linked",
            "binary",
        ]

        problem_keywords = [
            "problem",
            "shortest path",
            "matching",
            "scheduling",
            "knapsack",
            "coloring",
            "partitioning",
            "assignment",
        ]

        # Clasificar
        if any(kw in label_lower for kw in algorithm_keywords):
            return "Algorithm"
        elif any(kw in label_lower for kw in structure_keywords):
            return "DataStructure"
        elif any(kw in label_lower for kw in problem_keywords):
            return "Problem"
        else:
            return "Concept"


# ============================================
# SCRIPT STANDALONE (opcional)
# ============================================


def main():
    """Script standalone para extraer CSO y exportar JSON"""
    import argparse

    parser = argparse.ArgumentParser(description="Extrae topics de CSO alineados con CS2023")

    parser.add_argument("--ttl", type=str, default="CSO.3.5.ttl", help="Ruta al archivo CSO.ttl")

    parser.add_argument(
        "--depth", type=int, default=1, help="Profundidad de subtree (1=conservador, 2=completo)"
    )

    parser.add_argument(
        "--output", type=str, default="kg_from_cso.json", help="Archivo JSON de salida"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("CSO → ACADEMIC KG EXTRACTOR")
    print("=" * 70 + "\n")

    # Cargar CSO
    loader = CSOLoader(args.ttl)

    if not loader.load():
        print("\n❌ No se pudo cargar CSO")
        print("   Descarga de: https://cso.kmi.open.ac.uk/download")
        return

    # Extraer topics alineados
    result = loader.extract_cs2023_aligned_topics(depth=args.depth)

    # Exportar JSON
    print(f"\n📦 Exportando a {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Exportado: {len(result['nodes'])} nodos, {len(result['relations'])} relaciones")

    print("\n" + "=" * 70)
    print("✅ COMPLETADO")
    print("=" * 70)
    print("\nPróximo paso:")
    print(f"  python offline_etl/main_etl.py --use-cso --cso-file {args.output}")


if __name__ == "__main__":
    main()

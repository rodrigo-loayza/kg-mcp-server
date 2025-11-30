# concept_linker.py
"""
Concept Linker: Conecta concepts extraídos de documentos con el grafo existente
Genera relaciones RELATED_TO entre Document Concepts y Course Topics/Concepts
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ConceptLinker:
    """
    Conecta concepts extraídos de documentos con nodos existentes del grafo
    Genera relaciones RELATED_TO basadas en similitud semántica
    """

    def __init__(self, embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"):
        """
        Usa modelo MULTILINGÜE para manejar español-inglés
        - Reconoce "velocidad" ≈ "velocity"
        - Soporta contenido mixto español/inglés
        """
        self.model = SentenceTransformer(embedding_model)
        self.similarity_threshold = 0.70  # Threshold para RELATED_TO
        self.merge_threshold = 0.90  # Threshold para merge (mismo concepto)

    def link_concepts_to_topics(self, concepts_data: Dict, topics_data: Dict) -> Dict:
        """
        Conecta concepts de documentos con topics de sílabos

        Args:
            concepts_data: Output de document_processor (kg_from_documents.json)
            topics_data: Output de syllabus_extractor (kg_from_syllabi.json)

        Returns:
            Dict con nuevas relaciones RELATED_TO
        """
        print("\n" + "=" * 70)
        print("🔗 CONCEPT LINKER: Conectando Concepts con Topics")
        print("=" * 70)

        # Extraer concepts y topics
        concepts = [n for n in concepts_data["nodes"] if n.get("type") == "Concept"]
        topics = [n for n in topics_data["nodes"] if n.get("type") == "Topic"]

        print(f"\n📊 Datos de entrada:")
        print(f"   Concepts (documentos): {len(concepts)}")
        print(f"   Topics (sílabos):      {len(topics)}")

        if not concepts or not topics:
            print("⚠️  No hay concepts o topics para conectar")
            return {"relations": [], "merged_nodes": []}

        # Generar embeddings
        concept_embeddings = self._generate_embeddings(concepts, "name")
        topic_embeddings = self._generate_embeddings(topics, "name")

        # Calcular similitudes
        similarity_matrix = cosine_similarity(concept_embeddings, topic_embeddings)

        # Generar relaciones y detectar merges
        new_relations = []
        merged_nodes = []
        merge_mappings = {}  # concept_id -> topic_id

        for i, concept in enumerate(concepts):
            concept_id = concept["id"]
            concept_name = concept["name"]

            # Encontrar topic más similar
            similarities = similarity_matrix[i]
            max_sim_idx = np.argmax(similarities)
            max_similarity = similarities[max_sim_idx]

            if max_similarity >= self.merge_threshold:
                # Son el mismo concepto → merge
                topic = topics[max_sim_idx]
                topic_id = topic["id"]

                print(
                    f"   🔀 MERGE: '{concept_name}' → '{topic['name']}' (sim: {max_similarity:.3f})"
                )

                merged_nodes.append(
                    {
                        "concept_id": concept_id,
                        "topic_id": topic_id,
                        "similarity": float(max_similarity),
                        "action": "merge",
                    }
                )

                merge_mappings[concept_id] = topic_id

            elif max_similarity >= self.similarity_threshold:
                # Relacionados pero diferentes → RELATED_TO
                topic = topics[max_sim_idx]
                topic_id = topic["id"]

                print(
                    f"   🔗 LINK: '{concept_name}' ↔ '{topic['name']}' (sim: {max_similarity:.3f})"
                )

                new_relations.append(
                    {
                        "from": concept_id,
                        "to": topic_id,
                        "type": "RELATED_TO",
                        "similarity_score": float(max_similarity),
                        "relation_type": "cross_source_similarity",
                    }
                )

        print(f"\n✅ Resultados:")
        print(f"   Relaciones RELATED_TO: {len(new_relations)}")
        print(f"   Nodos para merge:      {len(merged_nodes)}")

        return {
            "relations": new_relations,
            "merged_nodes": merged_nodes,
            "merge_mappings": merge_mappings,
        }

    def apply_merges(self, concepts_data: Dict, merge_info: Dict) -> Dict:
        """
        Aplica merges: reemplaza concept_ids con topic_ids en relaciones
        """
        if not merge_info.get("merged_nodes"):
            return concepts_data

        print("\n🔀 Aplicando merges...")

        merge_mappings = merge_info["merge_mappings"]

        # Filtrar nodos: remover concepts que se mergearon
        filtered_nodes = [n for n in concepts_data["nodes"] if n.get("id") not in merge_mappings]

        # Actualizar relaciones: reemplazar concept_ids con topic_ids
        updated_relations = []
        for rel in concepts_data["relations"]:
            from_id = rel["from"]
            to_id = rel["to"]

            # Reemplazar si fue mergeado
            if from_id in merge_mappings:
                from_id = merge_mappings[from_id]
            if to_id in merge_mappings:
                to_id = merge_mappings[to_id]

            # Evitar auto-relaciones
            if from_id == to_id:
                continue

            updated_relations.append({**rel, "from": from_id, "to": to_id})

        removed_count = len(concepts_data["nodes"]) - len(filtered_nodes)
        print(f"   ✅ Nodos removidos (merged): {removed_count}")
        print(f"   ✅ Relaciones actualizadas: {len(updated_relations)}")

        return {
            "nodes": filtered_nodes,
            "relations": updated_relations,
            "metadata": concepts_data.get("metadata", {}),
        }

    def _generate_embeddings(self, nodes: List[Dict], name_key: str) -> np.ndarray:
        """Genera embeddings para una lista de nodos"""
        texts = [n.get(name_key, "") for n in nodes]
        return self.model.encode(texts, show_progress_bar=False)

    def link_and_merge_pipeline(self, concepts_file: Path, topics_file: Path, output_file: Path):
        """
        Pipeline completo: carga datos, conecta concepts, aplica merges, exporta
        """
        print("\n" + "=" * 70)
        print("🏗️  PIPELINE DE CONCEPT LINKING")
        print("=" * 70)

        # 1. Cargar datos
        print("\n📂 Cargando datos...")
        with open(concepts_file, "r", encoding="utf-8") as f:
            concepts_data = json.load(f)
        with open(topics_file, "r", encoding="utf-8") as f:
            topics_data = json.load(f)

        # 2. Conectar concepts con topics
        link_results = self.link_concepts_to_topics(concepts_data, topics_data)

        # 3. Agregar nuevas relaciones
        concepts_data["relations"].extend(link_results["relations"])

        # 4. Aplicar merges
        concepts_data = self.apply_merges(concepts_data, link_results)

        # 5. Actualizar metadata
        concepts_data["metadata"]["linked"] = True
        concepts_data["metadata"]["new_relations"] = len(link_results["relations"])
        concepts_data["metadata"]["merged_nodes"] = len(link_results["merged_nodes"])

        # 6. Exportar
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(concepts_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Datos enlazados exportados: {output_file}")
        print(f"\n📊 Resumen final:")
        print(f"   Nodos totales:       {len(concepts_data['nodes'])}")
        print(f"   Relaciones totales:  {len(concepts_data['relations'])}")
        print(f"   Nuevas relaciones:   {link_results['relations']}")

        return concepts_data


class ConceptLinkerWithCSO:
    """
    Concept Linker mejorado que usa CSO para enriquecimiento semántico

    Ventajas sobre ConceptLinker básico:
    - Extrae prerequisites automáticos desde CSO
    - Genera relaciones tipadas (PREREQUISITE_OF, EXTENDS, SOLVES)
    - Clasifica nodos por tipo (Algorithm, DataStructure, Problem, Concept)
    - Agrega nodos intermedios necesarios
    """

    def __init__(self, cso_loader, similarity_threshold: float = 0.70):
        """
        Args:
            cso_loader: Instancia de CSOLoader ya cargada
            similarity_threshold: Umbral para relaciones RELATED_TO
        """
        self.cso = cso_loader
        self.similarity_threshold = similarity_threshold

    def link_concepts(self, kg_data: Dict) -> Dict:
        """
        Enlaza conceptos usando CSO como fuente de conocimiento

        Args:
            kg_data: Grafo actual con nodos y relaciones

        Returns:
            kg_data enriquecido con nodos y relaciones de CSO
        """
        print("\n" + "=" * 70)
        print("🔗 CONCEPT LINKER CON CSO")
        print("=" * 70)

        # Debug: Ver qué nodos tenemos
        all_node_types = {}
        for node in kg_data["nodes"]:
            ntype = node.get("type", "Unknown")
            all_node_types[ntype] = all_node_types.get(ntype, 0) + 1

        print(f"\n📊 Nodos disponibles:")
        for ntype, count in sorted(all_node_types.items()):
            print(f"   {ntype:20} {count:3} nodos")

        # Extraer conceptos existentes (incluir Topic que viene del sílabo)
        concepts = [n for n in kg_data["nodes"] if n.get("type") in ["Concept", "Topic"]]

        print(f"\n🔍 Conceptos a enriquecer: {len(concepts)}")

        if len(concepts) == 0:
            print("⚠️  No hay conceptos para enriquecer")
            print("   Los nodos deben tener type='Concept' o type='Topic'")
            return kg_data

        new_nodes = []
        new_relations = []
        enriched_count = 0

        for concept in concepts:
            concept_name = concept.get("name", "")
            concept_id = concept.get("id")

            if not concept_name:
                print(f"⚠️  Concepto sin nombre, saltando...")
                continue

            print(f"\n🔍 Procesando '{concept_name}'...")

            try:
                # CRÍTICO: Si el topic ya tiene cso_uri (del discovery en step_1), usar ese
                cso_uri = concept.get("cso_uri")

                if cso_uri:
                    print(f"   ✓ Usando URI del discovery: {concept.get('cso_label', cso_uri)}")
                    cso_match = {
                        "found": True,
                        "uri": cso_uri,
                        "label": concept.get("cso_label", concept_name),
                        "similarity": concept.get("cso_similarity", 1.0),
                    }
                else:
                    # Si no tiene URI, buscar en CSO
                    print(f"   🔍 Buscando en CSO...")
                    cso_match = self.cso.find_topic(concept_name, threshold=0.6)

                # VALIDACIÓN ROBUSTA
                if cso_match is None:
                    print(f"   ⚠️  Error en búsqueda CSO (retornó None)")
                    continue

                if not isinstance(cso_match, dict):
                    print(f"   ⚠️  Respuesta CSO inválida (tipo: {type(cso_match)})")
                    continue

                if not cso_match.get("found", False):
                    print(f"   ⚠️  No encontrado en CSO")
                    continue

                # Verificar campos requeridos
                if "uri" not in cso_match or "label" not in cso_match:
                    print(f"   ⚠️  Respuesta CSO incompleta")
                    continue

                print(
                    f"   ✅ Match: {cso_match['label']} (sim: {cso_match.get('similarity', 0):.2f})"
                )
                enriched_count += 1

                cso_uri = cso_match["uri"]

                # 1. Extraer prerequisites (broader)
                prerequisites = self.cso.get_prerequisites(cso_uri, max_depth=2)
                print(f"   📚 Prerequisites: {len(prerequisites)}")

                for prereq_uri in prerequisites:
                    if prereq_uri not in self.cso.topics:
                        continue

                    prereq_data = self.cso.topics[prereq_uri]
                    prereq_type = self._classify_type(prereq_data["label"])
                    prereq_id = f"{prereq_type.lower()}_{prereq_data['id']}"

                    # Crear nodo si no existe
                    prereq_node = {
                        "type": prereq_type,
                        "id": prereq_id,
                        "name": prereq_data["label"],
                        "source": "cso",
                        "cso_uri": prereq_uri,
                    }

                    if not self._node_exists(prereq_node, new_nodes + kg_data["nodes"]):
                        new_nodes.append(prereq_node)

                    # Crear relación PREREQUISITE_OF
                    new_relations.append(
                        {
                            "from": concept_id,
                            "to": prereq_id,
                            "type": "PREREQUISITE_OF",
                            "source": "cso_hierarchy",
                            "strength": 0.85,
                        }
                    )

                # 2. Extraer subtopics (narrower)
                subtopics = self.cso.get_subtopics(cso_uri, max_depth=1)
                print(f"   🌳 Subtopics: {len(subtopics)}")

                for subtopic_uri in subtopics:
                    if subtopic_uri not in self.cso.topics:
                        continue

                    subtopic_data = self.cso.topics[subtopic_uri]
                    subtopic_type = self._classify_type(subtopic_data["label"])
                    subtopic_id = f"{subtopic_type.lower()}_{subtopic_data['id']}"

                    # Crear nodo si no existe
                    subtopic_node = {
                        "type": subtopic_type,
                        "id": subtopic_id,
                        "name": subtopic_data["label"],
                        "source": "cso",
                        "cso_uri": subtopic_uri,
                    }

                    if not self._node_exists(subtopic_node, new_nodes + kg_data["nodes"]):
                        new_nodes.append(subtopic_node)

                    # Crear relación EXTENDS (subtopic extiende concept)
                    new_relations.append(
                        {
                            "from": subtopic_id,
                            "to": concept_id,
                            "type": "EXTENDS",
                            "source": "cso_hierarchy",
                            "strength": 0.80,
                        }
                    )

                # 3. Extraer related topics
                related = self.cso.get_related_topics(cso_uri)
                print(f"   🔗 Related: {len(related)}")

                for related_uri in related[:5]:  # Limitar a 5 para no explotar el grafo
                    if related_uri not in self.cso.topics:
                        continue

                    related_data = self.cso.topics[related_uri]
                    related_type = self._classify_type(related_data["label"])
                    related_id = f"{related_type.lower()}_{related_data['id']}"

                    # Crear nodo si no existe
                    related_node = {
                        "type": related_type,
                        "id": related_id,
                        "name": related_data["label"],
                        "source": "cso",
                        "cso_uri": related_uri,
                    }

                    if not self._node_exists(related_node, new_nodes + kg_data["nodes"]):
                        new_nodes.append(related_node)

                    # Crear relación RELATED_TO
                    new_relations.append(
                        {
                            "from": concept_id,
                            "to": related_id,
                            "type": "RELATED_TO",
                            "source": "cso",
                            "strength": 0.60,
                        }
                    )

                # 4. Procesar contributesTo (si existe)
                if "contributes" in self.cso.topics[cso_uri]:
                    contributes_to = self.cso.topics[cso_uri]["contributes"]
                    print(f"   💡 ContributesTo: {len(contributes_to)}")

                    for contrib_uri in contributes_to[:3]:  # Limitar a 3
                        if contrib_uri not in self.cso.topics:
                            continue

                        contrib_data = self.cso.topics[contrib_uri]
                        contrib_type = self._classify_type(contrib_data["label"])
                        contrib_id = f"{contrib_type.lower()}_{contrib_data['id']}"

                        # Crear nodo si no existe
                        contrib_node = {
                            "type": contrib_type,
                            "id": contrib_id,
                            "name": contrib_data["label"],
                            "source": "cso",
                            "cso_uri": contrib_uri,
                        }

                        if not self._node_exists(contrib_node, new_nodes + kg_data["nodes"]):
                            new_nodes.append(contrib_node)

                        # Decidir tipo de relación según tipos de nodo
                        from_type = self._classify_type(concept_name)
                        to_type = contrib_type

                        if from_type in ["Algorithm", "DataStructure"] and to_type == "Problem":
                            rel_type = "SOLVES"
                            extra_props = {"efficiency": 0.85}
                        elif from_type in ["Algorithm", "DataStructure"] and to_type == "Concept":
                            rel_type = "EXEMPLIFIES"
                            extra_props = {"clarity_score": 0.80}
                        else:
                            rel_type = "CONTRIBUTES_TO"
                            extra_props = {}

                        new_relations.append(
                            {
                                "from": concept_id,
                                "to": contrib_id,
                                "type": rel_type,
                                "source": "cso_contributes",
                                **extra_props,
                            }
                        )

            except Exception as e:
                print(f"   ❌ Error procesando concepto '{concept_name}': {e}")
                import traceback

                traceback.print_exc()
                continue

        # Agregar nodos y relaciones al KG
        kg_data["nodes"].extend(new_nodes)
        kg_data["relations"].extend(new_relations)

        # Estadísticas
        print("\n" + "=" * 70)
        print("✅ ENRIQUECIMIENTO COMPLETADO")
        print("=" * 70)
        print(f"\n📊 Estadísticas:")
        print(f"   Conceptos enriquecidos:  {enriched_count}/{len(concepts)}")
        print(f"   Nodos agregados:         {len(new_nodes)}")
        print(f"   Relaciones agregadas:    {len(new_relations)}")

        # Distribución por tipo de nodo
        node_types = defaultdict(int)
        for node in new_nodes:
            node_types[node["type"]] += 1

        if node_types:
            print(f"\n📦 Distribución de nodos agregados:")
            for ntype, count in sorted(node_types.items()):
                print(f"   {ntype:20} {count:3} nodos")

        # Distribución por tipo de relación
        rel_types = defaultdict(int)
        for rel in new_relations:
            rel_types[rel["type"]] += 1

        if rel_types:
            print(f"\n🔗 Distribución de relaciones agregadas:")
            for rtype, count in sorted(rel_types.items()):
                print(f"   {rtype:20} {count:3} relaciones")

        return kg_data

    def _classify_type(self, label: str) -> str:
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
            "technique",
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

    def _node_exists(self, node: Dict, node_list: List[Dict]) -> bool:
        """Verifica si un nodo ya existe en la lista"""
        node_id = node["id"]
        return any(n["id"] == node_id for n in node_list)


def main():
    """Script standalone para concept linking"""
    import sys

    if len(sys.argv) < 3:
        print("Uso: python concept_linker.py <kg_from_documents.json> <kg_from_syllabi.json>")
        sys.exit(1)

    concepts_file = Path(sys.argv[1])
    topics_file = Path(sys.argv[2])
    output_file = Path("kg_from_documents_linked.json")

    linker = ConceptLinker()
    linker.link_and_merge_pipeline(concepts_file, topics_file, output_file)

    print("\n🎯 Siguiente paso:")
    print(f"   python instantiate_kg_v2.py")
    print(f"   (usará automáticamente el archivo linked)")


if __name__ == "__main__":
    main()

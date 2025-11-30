"""
Document-Topic Linker
Relaciona documentos con topics basándose en:
1. Estructura de carpetas (Semana 1, 2, 3...)
2. Nombres de archivos
3. Keywords del contenido
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class DocumentTopicLinker:
    """
    Relaciona documentos con topics usando estructura de carpetas
    y análisis de contenido
    """

    def __init__(self, cso_discoverer, syllabus_topics: List[Dict]):
        """
        Args:
            cso_discoverer: Instancia de CSOTopicDiscoverer
            syllabus_topics: Topics extraídos del sílabo con mapeo CSO
        """
        self.discoverer = cso_discoverer
        self.syllabus_topics = syllabus_topics

        # Crear índice de topics por capítulo/semana
        self.topics_by_chapter = self._index_topics_by_chapter()

    def link_documents_to_topics(self, documents_data: Dict, input_dir: Path) -> Dict:
        """
        Crea relaciones entre documentos y topics

        Args:
            documents_data: Datos de documentos (del document_extractor)
            input_dir: Directorio raíz con estructura de carpetas

        Returns:
            documents_data enriquecido con relaciones
        """
        print("\n" + "=" * 70)
        print("🔗 DOCUMENT-TOPIC LINKER")
        print("=" * 70)

        new_relations = []
        documents = [n for n in documents_data["nodes"] if n["type"] == "Document"]

        print(f"\n📊 Documentos a procesar: {len(documents)}")
        print(f"📚 Topics del sílabo: {len(self.syllabus_topics)}")

        for doc in documents:
            # Manejar estructura con/sin properties
            props = doc.get("properties", doc)  # Fallback si no hay 'properties'

            # Obtener path (puede ser 'path' o 'relative_path')
            path_str = props.get("path", props.get("relative_path", ""))
            doc_path = Path(path_str) if path_str else None

            doc_id = doc["id"]
            doc_name = props.get("name", doc.get("name", "unknown"))

            print(f"\n📄 Procesando: {doc_name}")

            # 1. Obtener carpeta padre
            folder_info = self._get_folder_hierarchy(doc_path, input_dir)

            if folder_info:
                print(f"   📁 Carpeta: {folder_info['immediate_parent']}")
                if folder_info["week"]:
                    print(f"   📅 Semana: {folder_info['week']}")

            # 2. Descubrir topic desde nombre de archivo
            filename_topic = self.discoverer.discover_from_document_name(doc_name)

            if filename_topic:
                print(f"   🏷️  Topic (nombre): {filename_topic['cso_label']}")

            # 3. Buscar topic del sílabo correspondiente
            matched_topics = self._match_to_syllabus_topics(doc_name, folder_info, filename_topic)

            # 4. Crear relaciones
            for topic_match in matched_topics:
                syllabus_topic_id = topic_match["syllabus_topic_id"]
                match_reason = topic_match["reason"]
                confidence = topic_match["confidence"]

                print(
                    f"   ✅ Matched: {topic_match['topic_name']} (conf: {confidence:.2f}, razón: {match_reason})"
                )

                # Relación: Document → Topic (DISCUSSES)
                new_relations.append(
                    {
                        "from": doc_id,
                        "to": syllabus_topic_id,
                        "type": "DISCUSSES",
                        "confidence": confidence,
                        "match_reason": match_reason,
                        "source": "document_topic_linker",
                    }
                )

                # Si hay carpeta padre, crear relación de agrupación
                if folder_info and folder_info["week"]:
                    # Relación: Topic → Semana (IMPLIED_BY_WEEK)
                    week_id = f"week_{folder_info['week']}"

                    # Agregar nodo Week si no existe
                    week_node = {
                        "id": week_id,
                        "type": "Week",
                        "number": folder_info["week"],
                        "name": f"Semana {folder_info['week']}",
                    }

                    if not any(n["id"] == week_id for n in documents_data["nodes"]):
                        documents_data["nodes"].append(week_node)

                    new_relations.append(
                        {
                            "from": doc_id,
                            "to": week_id,
                            "type": "BELONGS_TO_WEEK",
                            "source": "folder_structure",
                        }
                    )

        # Agregar nuevas relaciones
        documents_data["relations"].extend(new_relations)

        print(f"\n✅ Relaciones creadas: {len(new_relations)}")

        # Estadísticas
        rel_types = defaultdict(int)
        for rel in new_relations:
            rel_types[rel["type"]] += 1

        print("\n📊 Distribución de relaciones:")
        for rtype, count in rel_types.items():
            print(f"   {rtype:25} {count:3} relaciones")

        return documents_data

    def _get_folder_hierarchy(self, doc_path: Path, input_dir: Path) -> Optional[Dict]:
        """
        Extrae información de jerarquía de carpetas

        Args:
            doc_path: Ruta completa del documento
            input_dir: Directorio raíz

        Returns:
            Dict con info de carpeta o None
        """
        try:
            # Obtener ruta relativa
            rel_path = doc_path.relative_to(input_dir)

            # Carpetas en el path
            parts = rel_path.parts[:-1]  # Excluir nombre de archivo

            if not parts:
                return None

            # Carpeta padre inmediata
            immediate_parent = parts[-1] if parts else ""

            # Buscar número de semana
            week = None
            for part in parts:
                week_match = re.search(r"[Ss]emana\s*(\d+)", part)
                if week_match:
                    week = int(week_match.group(1))
                    break

            return {
                "immediate_parent": immediate_parent,
                "all_parents": list(parts),
                "week": week,
                "depth": len(parts),
            }

        except ValueError:
            return None

    def _match_to_syllabus_topics(
        self, doc_name: str, folder_info: Optional[Dict], filename_topic: Optional[Dict]
    ) -> List[Dict]:
        """
        Busca topics del sílabo que coincidan con el documento

        Args:
            doc_name: Nombre del documento
            folder_info: Info de carpeta
            filename_topic: Topic descubierto desde nombre

        Returns:
            Lista de topics matched con confidence
        """
        matched = []

        # Estrategia 1: Match directo por nombre de archivo
        if filename_topic:
            cso_label = filename_topic["cso_label"]

            for topic in self.syllabus_topics:
                if topic.get("cso_label", "").lower() == cso_label.lower():
                    matched.append(
                        {
                            "syllabus_topic_id": topic["syllabus_topic_id"],
                            "topic_name": topic["syllabus_topic_name"],
                            "confidence": 0.9,
                            "reason": "filename_cso_match",
                        }
                    )

        # Estrategia 2: Match por semana/capítulo
        if folder_info and folder_info["week"]:
            week_num = folder_info["week"]

            # Mapeo aproximado: Semana → Capítulo
            # (puedes ajustar según tu curso)
            chapter_map = {
                1: 1,  # Semana 1 → Capítulo 1
                2: 2,  # Semana 2 → Capítulo 2
                3: 2,  # Semana 3 → Capítulo 2
                4: 3,  # Semana 4 → Capítulo 3
                5: 3,
                6: 4,
                7: 4,
                8: 4,
                9: 5,
                10: 5,
            }

            expected_chapter = chapter_map.get(week_num)

            if expected_chapter:
                for topic in self.syllabus_topics:
                    chapter_str = topic.get("chapter", "")
                    chapter_match = re.search(r"Capítulo\s*(\d+)", chapter_str)

                    if chapter_match and int(chapter_match.group(1)) == expected_chapter:
                        # Evitar duplicados
                        if not any(
                            m["syllabus_topic_id"] == topic["syllabus_topic_id"] for m in matched
                        ):
                            matched.append(
                                {
                                    "syllabus_topic_id": topic["syllabus_topic_id"],
                                    "topic_name": topic["syllabus_topic_name"],
                                    "confidence": 0.7,
                                    "reason": "week_chapter_mapping",
                                }
                            )

        # Estrategia 3: Match por keywords en nombre
        doc_name_lower = doc_name.lower()

        for topic in self.syllabus_topics:
            topic_name_lower = topic["syllabus_topic_name"].lower()
            topic_keywords = topic.get("keywords", [])

            # Buscar keywords del topic en nombre del doc
            keyword_matches = 0
            for keyword in topic_keywords:
                if keyword.lower() in doc_name_lower:
                    keyword_matches += 1

            if keyword_matches >= 1:
                # Evitar duplicados
                if not any(m["syllabus_topic_id"] == topic["syllabus_topic_id"] for m in matched):
                    confidence = 0.6 + (0.1 * keyword_matches)
                    matched.append(
                        {
                            "syllabus_topic_id": topic["syllabus_topic_id"],
                            "topic_name": topic["syllabus_topic_name"],
                            "confidence": min(confidence, 0.85),
                            "reason": f"keyword_match_{keyword_matches}",
                        }
                    )

        # Ordenar por confidence
        matched.sort(key=lambda x: x["confidence"], reverse=True)

        # Retornar solo top 2 matches
        return matched[:2]

    def _index_topics_by_chapter(self) -> Dict[int, List[Dict]]:
        """Indexa topics por número de capítulo"""
        index = defaultdict(list)

        for topic in self.syllabus_topics:
            chapter_str = topic.get("chapter", "")
            chapter_match = re.search(r"Capítulo\s*(\d+)", chapter_str)

            if chapter_match:
                chapter_num = int(chapter_match.group(1))
                index[chapter_num].append(topic)

        return index


def main():
    """Test standalone"""
    from cso_topic_discoverer import CSOTopicDiscoverer
    from loaders.cso_loader import CSOLoader

    # Cargar CSO
    cso_file = "CSO.3.5.ttl"
    cso_loader = CSOLoader(cso_file)

    if not cso_loader.load():
        print("❌ Error cargando CSO")
        return

    # Crear discoverer
    discoverer = CSOTopicDiscoverer(cso_loader)

    # Topics del sílabo (ejemplo)
    syllabus_topics = [
        {
            "syllabus_topic_id": "topic_inf265_ch1",
            "syllabus_topic_name": "ALGORITMOS Y HEURÍSTICAS DE BÚSQUEDA",
            "cso_uri": "https://cso.kmi.open.ac.uk/topics/search",
            "cso_label": "search",
            "chapter": "Capítulo 2",
            "keywords": ["algoritmos", "búsqueda", "heurísticas"],
        },
        {
            "syllabus_topic_id": "topic_inf265_ch2",
            "syllabus_topic_name": "APRENDIZAJE DE MAQUINA SUPERVISADO",
            "cso_uri": "https://cso.kmi.open.ac.uk/topics/machine-learning",
            "cso_label": "machine learning",
            "chapter": "Capítulo 4",
            "keywords": ["aprendizaje", "supervisado", "clasificación"],
        },
    ]

    # Crear linker
    linker = DocumentTopicLinker(discoverer, syllabus_topics)

    # Documents data (ejemplo)
    documents_data = {
        "nodes": [
            {
                "id": "doc_busqueda",
                "type": "Document",
                "name": "BusquedaSinInformacion.pdf",
                "path": "/data/raw/INF265/Semana 2/BusquedaSinInformacion.pdf",
            },
            {
                "id": "doc_ml",
                "type": "Document",
                "name": "LecturaML.pdf",
                "path": "/data/raw/INF265/Semana 6/LecturaML.pdf",
            },
        ],
        "relations": [],
    }

    input_dir = Path("/data/raw/INF265")

    # Link
    enriched = linker.link_documents_to_topics(documents_data, input_dir)

    print(f"\n✅ Resultado:")
    print(f"   Nodos: {len(enriched['nodes'])}")
    print(f"   Relaciones: {len(enriched['relations'])}")


if __name__ == "__main__":
    main()

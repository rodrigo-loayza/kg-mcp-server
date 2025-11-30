"""
CSO Topic Discoverer
Descubre y mapea topics desde CSO con soporte multilingüe (español-inglés)
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import Counter, defaultdict
import json


class CSOTopicDiscoverer:
    """
    Descubre topics relevantes desde CSO usando:
    1. Topics del sílabo (español) → CSO (inglés)
    2. Nombres de archivos → CSO
    3. Keywords de contenido → CSO
    """

    def __init__(self, cso_loader):
        """
        Args:
            cso_loader: Instancia de CSOLoader ya cargada
        """
        self.cso = cso_loader

        # Diccionario español → inglés para topics comunes de CS
        self.es_to_en = {
            # IA
            "inteligencia artificial": "artificial intelligence",
            "agentes inteligentes": "intelligent agents",
            "agente": "agent",
            "agentes": "agents",
            # Búsqueda
            "búsqueda": "search",
            "busqueda": "search",
            "algoritmos de búsqueda": "search algorithms",
            "algoritmos de busqueda": "search algorithms",
            "heurística": "heuristic",
            "heurísticas": "heuristics",
            "heuristica": "heuristic",
            "heuristicas": "heuristics",
            "metaheurística": "metaheuristic",
            "metaheurísticas": "metaheuristics",
            "metaheuristica": "metaheuristic",
            "metaheuristicas": "metaheuristics",
            "bioinspiradas": "bio-inspired",
            # Algoritmos
            "algoritmos": "algorithms",
            "algoritmo": "algorithm",
            "algoritmos genéticos": "genetic algorithms",
            "algoritmo genético": "genetic algorithm",
            "enjambre de partículas": "particle swarm optimization",
            "colonia de hormigas": "ant colony optimization",
            "recocido simulado": "simulated annealing",
            "escalada": "hill climbing",
            # ML
            "aprendizaje": "learning",
            "aprendizaje de máquina": "machine learning",
            "aprendizaje de maquina": "machine learning",
            "aprendizaje automático": "machine learning",
            "aprendizaje supervisado": "supervised learning",
            "supervisado": "supervised learning",
            "aprendizaje no supervisado": "unsupervised learning",
            "no supervisado": "unsupervised learning",
            "clasificación": "classification",
            "clasificacion": "classification",
            "regresión": "regression",
            "regresion": "regression",
            "agrupamiento": "clustering",
            "fundamentos": "fundamentals",
            # Redes Neuronales
            "redes neuronales": "neural networks",
            "red neuronal": "neural network",
            "aprendizaje profundo": "deep learning",
            "perceptrón": "perceptron",
            "perceptron": "perceptron",
            # Compiladores
            "compiladores": "compilers",
            "compilador": "compiler",
            "teoría de compiladores": "compiler theory",
            "teoria de compiladores": "compiler theory",
            "análisis léxico": "lexical analysis",
            "análisis sintáctico": "syntactic analysis",
            "analisis lexico": "lexical analysis",
            "analisis sintactico": "syntactic analysis",
            "introducción": "introduction",
            "introduccion": "introduction",
            # Estructuras
            "árbol": "tree",
            "arbol": "tree",
            "grafo": "graph",
            "lista": "list",
            "pila": "stack",
            "cola": "queue",
        }

    def discover_from_syllabus_topics(self, topics: List[Dict]) -> List[Dict]:
        """
        Descubre topics CSO desde topics del sílabo

        Args:
            topics: Lista de topics extraídos del sílabo

        Returns:
            Lista de topics descubiertos con info de CSO
        """
        discovered = []

        print("\n🔍 Descubriendo topics CSO desde sílabo...")

        for topic in topics:
            topic_name = topic.get("name", "")
            keywords = topic.get("keywords", [])

            print(f"\n   📌 Topic: {topic_name}")

            # 1. Intentar buscar directamente (nombre completo)
            match = self._find_in_cso_multilingual(topic_name, threshold=0.6)

            if match.get("found", False):
                print(
                    f"      ✅ Encontrado (completo): {match.get('label', '')} (sim: {match.get('similarity', 0):.2f})"
                )

                discovered.append(
                    {
                        "syllabus_topic_id": topic["id"],
                        "syllabus_topic_name": topic_name,
                        "cso_uri": match.get("uri", ""),
                        "cso_label": match.get("label", ""),
                        "similarity": match.get("similarity", 0.0),
                        "source": "syllabus",
                        "chapter": topic.get("chapter", ""),
                    }
                )
                continue

            # 2. Intentar buscar por palabras individuales del nombre
            print(f"      ⚠️  No encontrado completo, buscando por palabras...")

            # Extraer palabras significativas del nombre (sin stopwords)
            stopwords = {"de", "la", "el", "en", "y", "a", "los", "del", "para", "con", "una", "un"}
            words = [
                w.lower() for w in topic_name.split() if w.lower() not in stopwords and len(w) > 3
            ]

            best_match = None
            best_similarity = 0.0

            for word in words:
                word_match = self._find_in_cso_multilingual(word, threshold=0.5)

                if word_match.get("found", False):
                    sim = word_match.get("similarity", 0)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = word_match
                        best_word = word

            if best_match:
                print(
                    f"      ✅ Match por palabra '{best_word}': {best_match.get('label', '')} (sim: {best_similarity:.2f})"
                )

                discovered.append(
                    {
                        "syllabus_topic_id": topic["id"],
                        "syllabus_topic_name": topic_name,
                        "cso_uri": best_match.get("uri", ""),
                        "cso_label": best_match.get("label", ""),
                        "similarity": best_similarity,
                        "source": "syllabus_word",
                        "chapter": topic.get("chapter", ""),
                        "matched_word": best_word,
                    }
                )
                continue

            # 3. Intentar con keywords extraídos del contenido
            if keywords:
                print(f"      ⚠️  Probando con keywords del contenido...")

                for keyword in keywords[:5]:  # Top 5 keywords
                    kw_match = self._find_in_cso_multilingual(keyword, threshold=0.5)

                    if kw_match.get("found", False) and kw_match.get("similarity", 0) > 0.5:
                        print(
                            f"      ✅ Match con keyword '{keyword}': {kw_match.get('label', '')}"
                        )

                        discovered.append(
                            {
                                "syllabus_topic_id": topic["id"],
                                "syllabus_topic_name": topic_name,
                                "cso_uri": kw_match.get("uri", ""),
                                "cso_label": kw_match.get("label", ""),
                                "similarity": kw_match.get("similarity", 0.0),
                                "source": "syllabus_keyword",
                                "chapter": topic.get("chapter", ""),
                                "matched_keyword": keyword,
                            }
                        )
                        break

            # Si no encontró nada
            if not any(d["syllabus_topic_id"] == topic["id"] for d in discovered):
                print(f"      ⚠️  No se encontró match en CSO")

                discovered.append(
                    {
                        "syllabus_topic_id": topic["id"],
                        "syllabus_topic_name": topic_name,
                        "cso_uri": kw_match.get("uri", ""),
                        "cso_label": kw_match.get("label", ""),
                        "similarity": kw_match.get("similarity", 0.0),
                        "source": "syllabus_keyword",
                        "chapter": topic.get("chapter", ""),
                        "matched_keyword": keyword,
                    }
                )
                break

        print(f"\n   ✅ {len(discovered)} topics mapeados a CSO")

        return discovered

    def discover_from_document_name(self, filename: str, threshold: float = 0.5) -> Optional[Dict]:
        """
        Descubre topic CSO desde nombre de archivo

        Args:
            filename: Nombre del archivo (ej: "Agentes (Parte1).pdf")
            threshold: Umbral de similitud

        Returns:
            Topic descubierto o None
        """
        # Limpiar nombre
        clean_name = self._clean_filename(filename)

        if not clean_name or len(clean_name) < 3:
            return None

        # Buscar en CSO
        match = self._find_in_cso_multilingual(clean_name, threshold=threshold)

        if match.get("found", False):
            return {
                "filename": filename,
                "cso_uri": match.get("uri", ""),
                "cso_label": match.get("label", ""),
                "similarity": match.get("similarity", 0.0),
                "source": "filename",
            }

        return None

    def discover_from_folder_name(self, folder_name: str) -> Optional[Dict]:
        """
        Descubre topic desde nombre de carpeta

        Args:
            folder_name: Nombre de carpeta (ej: "Semana 1", "LABORATORIO 0")

        Returns:
            Información de carpeta
        """
        # Detectar número de semana
        week_match = re.search(r"[Ss]emana\s*(\d+)", folder_name)
        lab_match = re.search(r"[Ll]aboratorio\s*(\d+)", folder_name)

        folder_info = {
            "folder_name": folder_name,
            "week": int(week_match.group(1)) if week_match else None,
            "lab": int(lab_match.group(1)) if lab_match else None,
            "type": "lecture" if week_match else ("lab" if lab_match else "other"),
        }

        return folder_info

    def discover_from_content_keywords(self, text: str, top_n: int = 10) -> List[Dict]:
        """
        Descubre topics desde keywords del contenido

        Args:
            text: Texto del documento
            top_n: Top N keywords a extraer

        Returns:
            Lista de topics descubiertos
        """
        # Limitar texto para performance (primeros 5000 chars)
        text_sample = text[:5000]

        # Extraer keywords
        keywords = self._extract_keywords(text_sample, top_n=top_n)

        # Buscar cada keyword en CSO
        discovered = []

        for keyword, freq in keywords:
            match = self._find_in_cso_multilingual(keyword, threshold=0.6)

            if match.get("found", False):
                discovered.append(
                    {
                        "keyword": keyword,
                        "frequency": freq,
                        "cso_uri": match.get("uri", ""),
                        "cso_label": match.get("label", ""),
                        "similarity": match.get("similarity", 0.0),
                        "source": "content_keyword",
                    }
                )

        return discovered

    def _find_in_cso_multilingual(self, text: str, threshold: float = 0.6) -> Dict:
        """
        Busca en CSO con soporte multilingüe

        Strategy:
        1. Intentar directamente en inglés/español
        2. Traducir español → inglés
        3. Fuzzy matching
        """
        text_lower = text.lower().strip()

        # 1. Traducir si está en español
        text_en = self.es_to_en.get(text_lower, text_lower)

        # 2. Buscar en CSO
        match = self.cso.find_topic(text_en, threshold=threshold)

        # Validar que match no sea None
        if match is None:
            return {"found": False, "label": "", "uri": "", "similarity": 0.0}

        # 3. Si no encuentra, probar con el original
        if not match.get("found", False) and text_en != text_lower:
            match2 = self.cso.find_topic(text_lower, threshold=threshold)
            if match2 is not None:
                match = match2

        return match

    def _clean_filename(self, filename: str) -> str:
        """
        Limpia nombre de archivo para extraer keywords

        Ejemplos:
        - "Agentes (Parte1).pdf" → "Agentes"
        - "BusquedaSinInformacion.pdf" → "Busqueda Sin Informacion"
        - "lab1.ipynb" → "" (no descriptivo)
        """
        # Remover extensión
        name = Path(filename).stem

        # Remover sufijos comunes
        name = re.sub(r"\s*\(Parte\s*\d+\)", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s*-\s*\d+", "", name)
        name = re.sub(r"^\s*[Ll]ab\s*\d*", "", name)  # lab1, Lab01, etc.
        name = re.sub(r"^\s*[Tt]utorial", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^\s*[Ee]jemplo", "", name, flags=re.IGNORECASE)

        # Separar CamelCase
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

        # Limpiar
        name = name.strip()

        # Si es muy corto o genérico, retornar vacío
        generic = ["lab", "tutorial", "ejemplo", "sesion", "clase", "notebook"]
        if len(name) < 3 or name.lower() in generic:
            return ""

        return name

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[tuple]:
        """
        Extrae top N keywords de un texto

        Returns:
            Lista de (keyword, frequency)
        """
        # Stopwords español
        stopwords = {
            "de",
            "la",
            "el",
            "en",
            "y",
            "a",
            "los",
            "del",
            "se",
            "las",
            "por",
            "un",
            "para",
            "con",
            "no",
            "una",
            "su",
            "al",
            "lo",
            "como",
            "más",
            "o",
            "pero",
            "sus",
            "le",
            "ya",
            "fue",
            "este",
            "ha",
            "sí",
            "porque",
            "esta",
            "son",
            "entre",
            "está",
            "cuando",
            "muy",
            "sin",
            "sobre",
            "ser",
            "tiene",
            "también",
            "me",
            "hasta",
            "hay",
            "donde",
            "han",
            "quien",
            "están",
            "estado",
            "desde",
            "todo",
            "nos",
            "durante",
            "estados",
            "todos",
            "uno",
            "les",
            "ni",
            "contra",
            "otros",
            "fueron",
            "ese",
            "eso",
            "había",
            "ante",
            "ellos",
            "e",
            "esto",
            "mí",
            "antes",
            "algunos",
            "qué",
            "unos",
            "yo",
            "otro",
            "otras",
            "otra",
            "él",
            "tanto",
            "esa",
            "estos",
            "mucho",
            "quienes",
            "nada",
            "muchos",
            "cual",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "am",
            "an",
            "as",
            "at",
            "by",
            "for",
            "from",
            "in",
            "into",
            "of",
            "on",
            "or",
            "out",
            "to",
            "up",
            "with",
            "and",
            "not",
            "but",
            "if",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "some",
            "such",
            "than",
            "too",
            "very",
            "one",
            "two",
        }

        # Tokenizar
        words = re.findall(r"\b[a-záéíóúñ]{3,}\b", text.lower())

        # Filtrar stopwords
        keywords = [w for w in words if w not in stopwords]

        # Contar frecuencias
        word_freq = Counter(keywords)

        # Top N
        return word_freq.most_common(top_n)


def main():
    """Test standalone"""
    from loaders.cso_loader import CSOLoader

    # Cargar CSO
    cso_file = "CSO.3.5.ttl"
    cso_loader = CSOLoader(cso_file)

    if not cso_loader.load():
        print("❌ Error cargando CSO")
        return

    # Crear discoverer
    discoverer = CSOTopicDiscoverer(cso_loader)

    # Test 1: Topics del sílabo
    print("\n" + "=" * 70)
    print("TEST 1: Topics desde sílabo")
    print("=" * 70)

    syllabus_topics = [
        {
            "id": "topic_1",
            "name": "ALGORITMOS Y HEURÍSTICAS DE BÚSQUEDA",
            "keywords": ["algoritmos", "búsqueda", "heurísticas"],
        },
        {
            "id": "topic_2",
            "name": "METAHEURÍSTICAS BIOINSPIRADAS",
            "keywords": ["metaheurísticas", "algoritmos", "genéticos"],
        },
        {
            "id": "topic_3",
            "name": "APRENDIZAJE DE MAQUINA SUPERVISADO",
            "keywords": ["aprendizaje", "supervisado", "clasificación"],
        },
    ]

    discovered = discoverer.discover_from_syllabus_topics(syllabus_topics)

    print(f"\n✅ Descubiertos: {len(discovered)} topics")
    for d in discovered:
        print(f"   {d['syllabus_topic_name']} → {d['cso_label']}")

    # Test 2: Nombres de archivo
    print("\n" + "=" * 70)
    print("TEST 2: Topics desde nombres de archivo")
    print("=" * 70)

    filenames = [
        "Agentes (Parte1).pdf",
        "BusquedaSinInformacion.pdf",
        "LecturaML.pdf",
        "Sesion10 (Preprocesamiento).pdf",
    ]

    for filename in filenames:
        discovered = discoverer.discover_from_document_name(filename)
        if discovered:
            print(f"   {filename} → {discovered['cso_label']}")
        else:
            print(f"   {filename} → No encontrado")


if __name__ == "__main__":
    main()

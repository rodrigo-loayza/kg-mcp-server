"""
Syllabus Extractor V2 - Mejorado para INF265
Extrae topics desde "PROGRAMA ANALÍTICO" con regex mejorado
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
from dataclasses import dataclass, asdict


@dataclass
class CourseInfo:
    """Información extraída de un sílabo"""

    code: str
    name: str
    credits: float
    semester: str
    level: str
    area: str
    topics: List[Dict]


class SyllabusExtractor:
    """
    Extrae información estructurada desde sílabos PDF
    Mejorado para manejar diferentes formatos
    """

    def __init__(self):
        self.area_map = {
            "INF265": "artificial_intelligence",
            "INF263": "algorithms",
        }

    def extract_from_pdf(self, pdf_path: Path) -> CourseInfo:
        """Extrae información completa de un sílabo PDF"""
        print(f"\n📄 Procesando sílabo: {pdf_path.name}")

        # Leer PDF
        text = self._read_pdf(pdf_path)

        # Extraer información básica
        code = self._extract_course_code(text)
        name = self._extract_course_name(text)
        credits = self._extract_credits(text)
        semester = self._extract_semester(text)
        level = "intermedio"
        area = self.area_map.get(code, "computer_science")

        # Extraer topics (MEJORADO)
        topics = self._extract_topics_from_chapters(text, code)

        print(f"   ✅ Curso: {code} - {name}")
        print(f"   ✅ Topics extraídos: {len(topics)}")

        return CourseInfo(
            code=code,
            name=name,
            credits=credits,
            semester=semester,
            level=level,
            area=area,
            topics=topics,
        )

    def _read_pdf(self, pdf_path: Path) -> str:
        """Lee texto de PDF"""
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_course_code(self, text: str) -> str:
        """Extrae código del curso"""
        match = re.search(r"CLAVE\s+([A-Z0-9]+)", text)
        if match:
            return match.group(1)

        match = re.search(r"\b(INF\d+|1INF\d+)\b", text)
        return match.group(1) if match else "UNKNOWN"

    def _extract_course_name(self, text: str) -> str:
        """Extrae nombre del curso"""
        match = re.search(r"CURSO\s+([A-ZÁÉÍÓÚÑ\s]+)(?:\n|CLAVE)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Curso Desconocido"

    def _extract_credits(self, text: str) -> float:
        """Extrae créditos"""
        match = re.search(r"CRÉDITOS\s+(\d+)", text)
        return float(match.group(1)) if match else 3.0

    def _extract_semester(self, text: str) -> str:
        """Extrae semestre"""
        match = re.search(r"(\d{4}-\d)", text)
        return match.group(1) if match else "2023-2"

    def _extract_topics_from_chapters(self, text: str, course_code: str) -> List[Dict]:
        """
        MEJORADO: Extrae topics desde CAPÍTULOS del PROGRAMA ANALÍTICO

        Busca patrones como:
        - CAPÍTULO 1 TEORÍA DE COMPILADORES - INTRODUCCIÓN (3 horas)
        - CAPÍTULO 2 ALGORITMOS Y HEURÍSTICAS DE BÚSQUEDA (9 horas)
        """
        topics = []

        # DEBUG: Mostrar snippet del texto
        print(f"\n   🔍 Buscando 'PROGRAMA ANALÍTICO' en {len(text)} caracteres...")

        # Buscar snippet alrededor de "PROGRAMA"
        programa_idx = text.find("PROGRAMA")
        if programa_idx > 0:
            snippet = text[max(0, programa_idx - 50) : programa_idx + 100]
            print(f"   📝 Contexto encontrado: ...{snippet}...")

        # Buscar sección "VI. PROGRAMA ANALÍTICO" (flexible con espacios)
        # Soporta: "VI. PROGRAMA", "VI.PROGRAMA", "VI . PROGRAMA"
        program_match = re.search(
            r"VI\.\s*PROGRAMA\s+ANALÍTICO(.+?)(?:VII\.|VIII\.|$)", text, re.DOTALL | re.IGNORECASE
        )

        if not program_match:
            print("   ⚠️  No se encontró 'VI. PROGRAMA ANALÍTICO' con el patrón principal")
            # Intentar variante sin acento
            program_match = re.search(
                r"VI\.\s*PROGRAMA\s+ANALITICO(.+?)(?:VII\.|VIII\.|$)",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if program_match:
                print("   ✓ Encontrado con 'ANALITICO' (sin acento)")

        if not program_match:
            print("   ⚠️  No se encontró 'PROGRAMA ANALÍTICO'")
            return self._extract_topics_fallback(text, course_code)

        program_text = program_match.group(1)
        print(f"   ✓ Sección encontrada ({len(program_text)} caracteres)")

        # DEBUG: Mostrar primeros 200 chars
        print(f"   📄 Inicio de sección: {program_text[:200]}...")

        # Extraer capítulos con regex mejorado
        # Patrón: CAPÍTULO N TÍTULO (X horas)
        chapters = re.findall(
            r"CAPÍTULO\s+(\d+)\s+([A-ZÁÉÍÓÚÑ\s\-\(\)]+?)\s*\((\d+)\s+horas?\)",
            program_text,
            re.IGNORECASE,
        )

        print(f"\n   📚 Capítulos encontrados: {len(chapters)}")

        for chapter_num, title, hours in chapters:
            # Limpiar título
            title = title.strip()
            title = re.sub(r"\s+", " ", title)

            # Extraer descripción del capítulo (siguiente párrafo)
            desc_pattern = (
                rf"CAPÍTULO\s+{chapter_num}\s+{re.escape(title)}.*?\n(.+?)(?:CAPÍTULO|\n\n|$)"
            )
            desc_match = re.search(desc_pattern, program_text, re.DOTALL | re.IGNORECASE)

            description = ""
            keywords = []
            if desc_match:
                description = desc_match.group(1).strip()
                # Extraer keywords importantes del descripción
                keywords = self._extract_keywords_from_text(description)

            topic_id = self._generate_topic_id(title, course_code, chapter_num)

            topic = {
                "id": topic_id,
                "type": "Topic",
                "name": title,
                "chapter": f"Capítulo {chapter_num}",
                "hours": int(hours),
                "description": description[:500] if description else "",  # Primeros 500 chars
                "keywords": keywords,
                "level": "intermedio",
                "area": self.area_map.get(course_code, "computer_science"),
            }

            topics.append(topic)
            print(f"      ✓ Capítulo {chapter_num}: {title}")

        return topics

    def _extract_topics_fallback(self, text: str, course_code: str) -> List[Dict]:
        """
        Fallback: Si no encuentra PROGRAMA ANALÍTICO, busca en SUMILLA
        """
        print("   💡 Intentando extraer desde SUMILLA...")

        # Buscar sección "IV. SUMILLA" (flexible con espacios)
        sumilla_match = re.search(
            r"IV\.\s*SUMILLA(.+?)(?:V\.|VI\.|$)", text, re.DOTALL | re.IGNORECASE
        )

        if not sumilla_match:
            print("   ⚠️  Tampoco se encontró SUMILLA")
            return []

        sumilla_text = sumilla_match.group(1)

        # Extraer frases separadas por puntos o comas
        topics_raw = re.split(r"[,;]\s*", sumilla_text)

        topics = []
        for idx, topic_text in enumerate(topics_raw[:8], 1):  # Máximo 8 topics
            topic_text = topic_text.strip()

            if len(topic_text) < 10 or len(topic_text) > 100:
                continue

            topic_id = self._generate_topic_id(topic_text, course_code, idx)

            topics.append(
                {
                    "id": topic_id,
                    "type": "Topic",
                    "name": topic_text,
                    "chapter": f"Topic {idx}",
                    "keywords": self._extract_keywords_from_text(topic_text),
                    "level": "intermedio",
                    "area": self.area_map.get(course_code, "computer_science"),
                }
            )

        return topics

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extrae keywords importantes de un texto"""
        # Stopwords en español
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
            "o",
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
            "sea",
            "poco",
            "ella",
            "estar",
            "haber",
            "estas",
            "estaba",
            "estamos",
            "algunas",
            "algo",
            "nosotros",
        }

        # Tokenizar
        words = re.findall(r"\b[a-záéíóúñ]{3,}\b", text.lower())

        # Filtrar stopwords
        keywords = [w for w in words if w not in stopwords]

        # Contar frecuencias
        from collections import Counter

        word_freq = Counter(keywords)

        # Top 5 más frecuentes
        top_keywords = [word for word, _ in word_freq.most_common(5)]

        return top_keywords

    def _generate_topic_id(self, title: str, course_code: str, chapter_num: int) -> str:
        """Genera ID único para topic"""
        # Limpiar título para ID
        title_clean = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower())
        title_clean = title_clean[:30]  # Máximo 30 chars

        return f"topic_{course_code.lower()}_ch{chapter_num}_{title_clean}"

    def to_kg_format(self, course_info: CourseInfo) -> Dict:
        """Convierte CourseInfo a formato KG"""
        nodes = []
        relations = []

        # Nodo Course
        course_node = {
            "id": f"course_{course_info.code.lower()}",
            "type": "Course",
            "name": course_info.name,
            "code": course_info.code,
            "credits": course_info.credits,
            "semester": course_info.semester,
            "level": course_info.level,
            "area": course_info.area,
        }
        nodes.append(course_node)

        # Nodos Topic
        for topic in course_info.topics:
            nodes.append(topic)

            # Relación TEACHES: Course → Topic
            relations.append(
                {"from": course_node["id"], "to": topic["id"], "type": "TEACHES", "weight": 1.0}
            )

        return {
            "nodes": nodes,
            "relations": relations,
            "metadata": {
                "source": "syllabus_v2",
                "course_code": course_info.code,
                "topics_count": len(course_info.topics),
            },
        }


def main():
    """Test standalone"""
    import sys

    if len(sys.argv) < 2:
        print("Uso: python syllabus_extractor_v2.py <silabo.pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌ No se encontró: {pdf_path}")
        sys.exit(1)

    extractor = SyllabusExtractor()
    course_info = extractor.extract_from_pdf(pdf_path)

    # Convertir a KG
    kg_data = extractor.to_kg_format(course_info)

    # Guardar
    output_file = Path("kg_from_syllabi.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(kg_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Guardado: {output_file}")
    print(f"   Nodos: {len(kg_data['nodes'])}")
    print(f"   Relaciones: {len(kg_data['relations'])}")


if __name__ == "__main__":
    main()

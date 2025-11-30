# academic_kg_model.py
"""
Modelo Completo de Knowledge Graph Académico
9 Nodos + 10 Relaciones (Versión Final con CSO para Tesis)

Nodos:
- Base (6): Course, Topic, Concept, Document, Chunk, Reference
- CSO (3): Algorithm, DataStructure, Problem

Relaciones:
- Base (5): TEACHES, CONTAINS, REQUIRES, RELATED_TO, CITES
- CSO (5): PREREQUISITE_OF, EXTENDS, SOLVES, EXEMPLIFIES, CONTRIBUTES_TO
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class NodeType(Enum):
    """9 tipos de nodos optimizados para construcción + CSO"""

    # Tipos base (desde sílabos y documentos)
    COURSE = "Course"  # Cursos (desde sílabos)
    TOPIC = "Topic"  # Temas principales (desde sílabos + CSO)
    CONCEPT = "Concept"  # Conceptos técnicos (desde documentos)
    DOCUMENT = "Document"  # PDFs, notebooks
    CHUNK = "Chunk"  # Fragmentos de documentos (para HNSW)
    REFERENCE = "Reference"  # Bibliografía

    # Tipos CSO (enriquecimiento semántico)
    ALGORITHM = "Algorithm"  # Algoritmos (desde CSO)
    DATASTRUCTURE = "DataStructure"  # Estructuras de datos (desde CSO)
    PROBLEM = "Problem"  # Problemas (desde CSO)


class RelationType(Enum):
    """10 tipos de relaciones (base + CSO)"""

    # Relaciones base
    TEACHES = "TEACHES"  # Course → Topic
    CONTAINS = "CONTAINS"  # Document → Concept / Document → Chunk
    REQUIRES = "REQUIRES"  # Course → Course (prerequisitos)
    RELATED_TO = "RELATED_TO"  # Concept ↔ Concept (similitud)
    CITES = "CITES"  # Document → Reference

    # Relaciones CSO (enriquecimiento semántico)
    PREREQUISITE_OF = "PREREQUISITE_OF"  # Topic → Concept (broader)
    EXTENDS = "EXTENDS"  # Concept → Topic (narrower)
    SOLVES = "SOLVES"  # Algorithm → Problem
    EXEMPLIFIES = "EXEMPLIFIES"  # Algorithm/DataStructure → Concept
    CONTRIBUTES_TO = "CONTRIBUTES_TO"  # Genérico CSO


class AcademicLevel(Enum):
    BASIC = "básico"
    INTERMEDIATE = "intermedio"
    ADVANCED = "avanzado"


@dataclass
class NodeProperties:
    """Propiedades base para nodos"""

    id: str
    name: str
    type: NodeType
    level: Optional[str] = None
    area: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class RelationProperties:
    """Propiedades base para relaciones"""

    from_id: str
    to_id: str
    type: RelationType
    weight: Optional[float] = None
    metadata: Optional[Dict] = None


# Mapeo de tipos de nodos a etiquetas Neo4j
NODE_LABELS = {
    NodeType.COURSE: "Course",
    NodeType.TOPIC: "Topic",
    NodeType.CONCEPT: "Concept",
    NodeType.DOCUMENT: "Document",
    NodeType.CHUNK: "Chunk",
    NodeType.REFERENCE: "Reference",
    NodeType.ALGORITHM: "Algorithm",
    NodeType.DATASTRUCTURE: "DataStructure",
    NodeType.PROBLEM: "Problem",
}

# Mapeo de tipos de relaciones a etiquetas Neo4j
RELATION_LABELS = {
    RelationType.TEACHES: "TEACHES",
    RelationType.CONTAINS: "CONTAINS",
    RelationType.REQUIRES: "REQUIRES",
    RelationType.RELATED_TO: "RELATED_TO",
    RelationType.CITES: "CITES",
    RelationType.PREREQUISITE_OF: "PREREQUISITE_OF",
    RelationType.EXTENDS: "EXTENDS",
    RelationType.SOLVES: "SOLVES",
    RelationType.EXEMPLIFIES: "EXEMPLIFIES",
    RelationType.CONTRIBUTES_TO: "CONTRIBUTES_TO",
}


def create_cypher_schema() -> List[str]:
    """
    Genera queries Cypher para crear constraints de unicidad

    Returns:
        Lista de queries Cypher para constraints
    """
    constraints = []

    # Constraint para cada tipo de nodo
    for node_type in NodeType:
        label = node_type.value
        constraint = f"""
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label})
        REQUIRE n.id IS UNIQUE
        """
        constraints.append(constraint)

    return constraints

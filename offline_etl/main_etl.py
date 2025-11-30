# main_etl.py
"""
🗂️ ETL Pipeline Maestro - Construcción Completa del Knowledge Graph Académico
FASE OFFLINE: Ejecutar una vez o cuando se actualizan documentos

Pipeline: Extract → Transform → Load
- Extract: Sílabos + Documentos
- Transform: Linking + Embeddings
- Load: Neo4j + HNSW

Autor: Rodrigo Cárdenas
Universidad: PUCP
Curso: INF265 - Tesis
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Componentes del pipeline
from extractors.syllabus_extractor import SyllabusExtractor
from extractors.document_extractor import MultiFormatExtractor
from extractors.concept_linker import ConceptLinker
from extractors.cso_topic_discoverer import CSOTopicDiscoverer
from extractors.document_topic_linker import DocumentTopicLinker
from builders.kg_builder import KGBuilder
from builders.hnsw_builder import HNSWBuilder
from models.academic_kg_model import NodeType, RelationType


class ETLPipeline:
    """
    Pipeline ETL Maestro para construcción del Knowledge Graph Académico

    Orquesta todo el proceso de construcción de índices:
    1. Extract: Extrae información de sílabos y documentos
    2. Transform: Procesa, enlaza y genera embeddings
    3. Load: Construye Neo4j KG + HNSW index
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        course_code: str,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "12345678",
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        max_documents: Optional[int] = None,
        use_cso: bool = False,
        cso_file: Optional[str] = None,
    ):
        """
        Inicializa el pipeline ETL

        Args:
            input_dir: Directorio con PDFs de entrada
            output_dir: Directorio para chunks procesados
            course_code: Código del curso (ej: INF265)
            neo4j_uri: URI de Neo4j
            neo4j_user: Usuario Neo4j
            neo4j_password: Password Neo4j
            embedding_model: Modelo de embeddings (multilingüe)
            max_documents: Límite de documentos a procesar (None = todos)
            use_cso: Si True, usa CSO para enriquecimiento semántico
            cso_file: Ruta al archivo CSO.ttl
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.course_code = course_code
        self.max_documents = max_documents

        # Configuración Neo4j
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Configuración embeddings
        self.embedding_model = embedding_model

        # Configuración CSO
        self.use_cso = use_cso
        self.cso_file = cso_file
        self.cso_loader = None

        # Archivos de salida
        self.kg_syllabi_file = Path("kg_from_syllabi.json")
        self.kg_docs_file = Path("kg_from_documents.json")
        self.kg_linked_file = Path("kg_from_documents_linked.json")
        self.hnsw_index_file = Path("data/indices/hnsw_index.bin")

        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Path("data/indices").mkdir(parents=True, exist_ok=True)

        # Inicializar componentes
        self.syllabus_extractor = SyllabusExtractor()
        self.document_extractor = MultiFormatExtractor()
        self.concept_linker = ConceptLinker(embedding_model=embedding_model)
        self.kg_builder = KGBuilder(neo4j_uri, neo4j_user, neo4j_password)
        self.hnsw_builder = HNSWBuilder(embedding_model=embedding_model)

        # Variables para CSO discovery
        self.cso_discoverer = None
        self.discovered_topics = []

        # Cargar CSO si se especificó
        if self.use_cso and self.cso_file:
            self._load_cso()

        self.stats = {
            "start_time": None,
            "end_time": None,
            "syllabi_processed": 0,
            "documents_processed": 0,
            "nodes_created": 0,
            "relations_created": 0,
            "chunks_indexed": 0,
        }

    def _load_cso(self):
        """
        Carga CSO para enriquecimiento semántico
        """
        print("\n" + "=" * 70)
        print("🌐 CARGANDO CSO (Computer Science Ontology)")
        print("=" * 70)

        try:
            from loaders.cso_loader import CSOLoader

            print(f"📥 Archivo CSO: {self.cso_file}")

            self.cso_loader = CSOLoader(self.cso_file)

            if self.cso_loader.load(verbose=True):
                print("✅ CSO cargado exitosamente")
                print(f"   Topics disponibles: {len(self.cso_loader.topics):,}")
                self.use_cso = True
            else:
                print("❌ Error cargando CSO")
                print("   Continuando sin CSO...")
                self.use_cso = False
                self.cso_loader = None

        except Exception as e:
            print(f"❌ Error al cargar CSO: {e}")
            print("   Continuando sin CSO...")
            self.use_cso = False
            self.cso_loader = None

    def step_1_extract_from_syllabi(self) -> bool:
        """
        PASO 1: Extraer información desde sílabos

        NUEVO: Usa SyllabusExtractorV2 y CSO Topic Discovery

        Output: kg_from_syllabi.json
        Nodos: Course, Topic (con mapeo CSO si use_cso=True)
        """
        print("\n" + "=" * 70)
        print("📚 PASO 1: EXTRACCIÓN DESDE SÍLABO")
        print("=" * 70)

        try:
            # Buscar PDF de sílabo (recursivamente)
            syllabus_pdfs = list(self.input_dir.rglob("*silabo*.pdf")) + list(
                self.input_dir.rglob("*syllabus*.pdf")
            )

            if not syllabus_pdfs:
                print(f"⚠️  No se encontró sílabo en {self.input_dir}")
                print("   Continuando sin sílabo (opcional)...")
                return True

            # Usar el primer sílabo encontrado
            syllabus_pdf = syllabus_pdfs[0]

            if len(syllabus_pdfs) > 1:
                print(f"⚠️  Múltiples sílabos encontrados ({len(syllabus_pdfs)})")
                print(f"   Usando: {syllabus_pdf.name}")
            else:
                print(f"📄 Sílabo encontrado: {syllabus_pdf.name}")

            # Extraer con V2
            extractor = SyllabusExtractor()
            course_info = extractor.extract_from_pdf(syllabus_pdf)

            # Convertir a formato KG
            kg_data = extractor.to_kg_format(course_info)

            # CSO Discovery (si está habilitado)
            if self.use_cso and self.cso_loader:
                print("\n🌐 Descubriendo topics en CSO...")

                try:
                    self.cso_discoverer = CSOTopicDiscoverer(self.cso_loader)

                    # Descubrir desde topics del sílabo
                    self.discovered_topics = self.cso_discoverer.discover_from_syllabus_topics(
                        course_info.topics
                    )

                    # Actualizar topics con info CSO
                    for node in kg_data["nodes"]:
                        if node["type"] == "Topic":
                            # Buscar en discovered
                            for disc in self.discovered_topics:
                                if disc["syllabus_topic_id"] == node["id"]:
                                    node["cso_uri"] = disc["cso_uri"]
                                    node["cso_label"] = disc["cso_label"]
                                    node["cso_similarity"] = disc["similarity"]
                                    break

                    print(f"   ✅ {len(self.discovered_topics)} topics mapeados a CSO")

                except Exception as e:
                    print(f"   ⚠️  Error en CSO discovery: {e}")
                    print("   Continuando sin mapeo CSO...")

            # Guardar
            with open(self.kg_syllabi_file, "w", encoding="utf-8") as f:
                json.dump(kg_data, f, ensure_ascii=False, indent=2)

            self.stats["syllabi_processed"] = 1

            topics_count = len([n for n in kg_data["nodes"] if n["type"] == "Topic"])

            print(f"\n✅ Sílabo procesado: {self.kg_syllabi_file}")
            print(f"   Curso: {course_info.code} - {course_info.name}")
            print(f"   Topics: {topics_count}")
            print(f"   Total nodos: {len(kg_data['nodes'])}")
            print(f"   Relaciones: {len(kg_data['relations'])}")

            return True

        except Exception as e:
            print(f"❌ Error procesando sílabo: {e}")
            import traceback

            traceback.print_exc()
            print("⚠️  Continuando sin sílabo...")
            return True  # No crítico
            print(f"   Relaciones: {len(relations)}")

            return True

        except Exception as e:
            print(f"❌ Error en extracción de sílabo: {e}")
            import traceback

            traceback.print_exc()
            return False

    def step_2_extract_from_documents(self) -> bool:
        """
        PASO 2: Procesar documentos de clase

        Output:
        - processed_docs/*_chunks.json
        - processed_docs/*_embeddings.npy
        - kg_from_documents.json

        Nodos: Document, Concept
        """
        print("\n" + "=" * 70)
        print("📄 PASO 2: PROCESAMIENTO DE DOCUMENTOS")
        print("=" * 70)

        try:
            # Buscar documentos RECURSIVAMENTE (en subdirectorios también)
            document_files = []
            extensions = ["*.pdf", "*.ipynb", "*.docx", "*.pptx", "*.html", "*.md"]

            print(f"🔍 Buscando documentos recursivamente en: {self.input_dir}")

            for ext in extensions:
                # rglob busca recursivamente en todos los subdirectorios
                found = list(self.input_dir.rglob(ext))
                document_files.extend(found)
                if found:
                    print(f"   Encontrados {len(found)} archivos {ext}")

            # Excluir sílabos
            document_files = [
                f
                for f in document_files
                if "silabo" not in f.name.lower() and "syllabus" not in f.name.lower()
            ]

            if not document_files:
                print(f"❌ No se encontraron documentos en {self.input_dir}")
                return False

            print(f"\n📚 Total encontrados: {len(document_files)} documentos")

            # LIMITAR a max_documents si está configurado (para pruebas)
            max_docs = getattr(self, "max_documents", None)
            if max_docs and max_docs > 0:
                document_files = document_files[:max_docs]
                print(f"⚠️  MODO PRUEBA: Limitado a {len(document_files)} documentos")

            print(f"\n📝 Procesando {len(document_files)} documentos...")

            # Procesar cada documento
            all_nodes = []
            all_relations = []

            for doc_file in document_files:
                # Mostrar ruta relativa en el log
                try:
                    rel_path = doc_file.relative_to(self.input_dir)
                    print(f"\n📖 Procesando: {rel_path}")
                except ValueError:
                    print(f"\n📖 Procesando: {doc_file.name}")

                try:
                    # Procesar documento con base_dir para rutas relativas
                    doc_data = self.document_extractor.process_document(
                        doc_file,
                        self.course_code,
                        self.output_dir,
                        base_dir=self.input_dir,  # ⭐ Pasar base_dir
                    )

                    # Agregar a colección
                    all_nodes.extend(doc_data["nodes"])
                    all_relations.extend(doc_data["relations"])

                    print(
                        f"   ✅ {len(doc_data['nodes'])} nodos, {len(doc_data['relations'])} relaciones"
                    )

                except Exception as e:
                    print(f"   ⚠️  Error procesando {doc_file.name}: {e}")
                    continue

            # Exportar KG desde documentos
            export_data = {
                "nodes": all_nodes,
                "relations": all_relations,
                "metadata": {
                    "source": "Documentos PAIDEIA",
                    "course": self.course_code,
                    "total_documents": len(document_files),
                    "total_nodes": len(all_nodes),
                    "total_relations": len(all_relations),
                },
            }

            with open(self.kg_docs_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.stats["documents_processed"] = len(document_files)

            print(f"\n✅ Documentos procesados: {self.kg_docs_file}")
            print(f"   Nodos: {len(all_nodes)}")
            print(f"   Relaciones: {len(all_relations)}")

            return True

        except Exception as e:
            print(f"❌ Error en procesamiento de documentos: {e}")
            import traceback

            traceback.print_exc()
            return False

    def step_2_5_link_documents_to_topics(self) -> bool:
        """
        PASO 2.5: Relacionar documentos con topics

        Usa:
        - Estructura de carpetas (Semana 1, 2, 3...)
        - Nombres de archivos
        - Keywords de contenido

        Output: Actualiza kg_from_documents.json con relaciones DISCUSSES
        """
        if not self.use_cso or not self.cso_discoverer:
            print("\n⚠️  Saltando document-topic linking (CSO no habilitado)")
            return True

        print("\n" + "=" * 70)
        print("🔗 PASO 2.5: DOCUMENT-TOPIC LINKING")
        print("=" * 70)

        try:
            # Verificar archivos
            if not self.kg_docs_file.exists():
                print("⚠️  No hay documentos procesados, saltando...")
                return True

            if not self.discovered_topics:
                print("⚠️  No hay topics del sílabo con mapeo CSO, saltando...")
                return True

            # Cargar datos de documentos
            with open(self.kg_docs_file, "r", encoding="utf-8") as f:
                docs_data = json.load(f)

            # Crear linker
            linker = DocumentTopicLinker(self.cso_discoverer, self.discovered_topics)

            # Link documentos con topics
            enriched_data = linker.link_documents_to_topics(docs_data, self.input_dir)

            # Guardar
            with open(self.kg_docs_file, "w", encoding="utf-8") as f:
                json.dump(enriched_data, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Document-topic linking completado")
            print(f"   Archivo actualizado: {self.kg_docs_file}")

            return True

        except Exception as e:
            print(f"❌ Error en document-topic linking: {e}")
            import traceback

            traceback.print_exc()
            print("⚠️  Continuando sin linking...")
            return True  # No crítico

    def step_3_link_concepts(self) -> bool:
        """
        PASO 3: Conectar concepts de documentos con topics de sílabos

        NUEVO: Si use_cso=True, usa CSO para enriquecimiento semántico

        Output: kg_from_documents_linked.json

        Genera relaciones RELATED_TO y merges de concepts
        """
        print("\n" + "=" * 70)
        print("🔗 PASO 3: CONCEPT LINKING")
        print("=" * 70)

        if not self.kg_syllabi_file.exists() or not self.kg_docs_file.exists():
            print("⚠️  Faltan archivos para linking, saltando paso...")
            return True  # No es crítico

        try:
            # Cargar datos
            with open(self.kg_docs_file, "r", encoding="utf-8") as f:
                docs_data = json.load(f)
                if not isinstance(docs_data, dict) or "nodes" not in docs_data:
                    print("⚠️  Formato de kg_from_documents.json inválido, saltando linking...")
                    return True

            with open(self.kg_syllabi_file, "r", encoding="utf-8") as f:
                syllabi_data = json.load(f)
                if not isinstance(syllabi_data, dict) or "nodes" not in syllabi_data:
                    print("⚠️  Formato de kg_from_syllabi.json inválido, saltando linking...")
                    return True

            # Combinar datos
            combined_data = {
                "nodes": syllabi_data.get("nodes", []) + docs_data.get("nodes", []),
                "relations": syllabi_data.get("relations", []) + docs_data.get("relations", []),
                "metadata": {
                    "sources": ["syllabi", "documents"],
                    "timestamp": datetime.now().isoformat(),
                },
            }

            # Aplicar linking según configuración
            if self.use_cso and self.cso_loader:
                print("\n🌐 Usando CSO para enriquecimiento semántico...")

                from extractors.concept_linker import ConceptLinkerWithCSO

                cso_linker = ConceptLinkerWithCSO(self.cso_loader)
                enriched_data = cso_linker.link_concepts(combined_data)

                print(f"\n✅ Enriquecido con CSO")

            else:
                print("\n🔗 Usando linking básico (embeddings)...")

                linked_data = self.concept_linker.link_and_merge_pipeline(
                    concepts_file=self.kg_docs_file,
                    topics_file=self.kg_syllabi_file,
                    output_file=self.kg_linked_file,
                )

                enriched_data = linked_data

                print(f"✅ Linking básico completado")

            # Guardar resultado
            with open(self.kg_linked_file, "w", encoding="utf-8") as f:
                json.dump(enriched_data, f, indent=2, ensure_ascii=False)

            print(f"\n📁 Guardado: {self.kg_linked_file}")
            print(f"   Nodos totales: {len(enriched_data['nodes'])}")
            print(f"   Relaciones totales: {len(enriched_data['relations'])}")

            # Actualizar referencia
            self.kg_docs_file = self.kg_linked_file

            return True

        except Exception as e:
            print(f"❌ Error en concept linking: {e}")
            import traceback

            traceback.print_exc()
            print("⚠️  Continuando sin linking...")
            return True  # No es crítico

    def step_4_build_neo4j_graph(self) -> bool:
        """
        PASO 4: Construir grafo en Neo4j

        Merge de sílabos + documentos → Neo4j
        """
        print("\n" + "=" * 70)
        print("🗄️ PASO 4: CONSTRUCCIÓN DEL GRAFO EN NEO4J")
        print("=" * 70)

        try:
            # Fuentes disponibles
            sources = []
            if self.kg_syllabi_file.exists():
                sources.append(self.kg_syllabi_file)
            if self.kg_docs_file.exists():
                sources.append(self.kg_docs_file)

            if not sources:
                print("❌ No hay fuentes de datos disponibles")
                return False

            # Construir grafo
            self.kg_builder.build_from_multiple_sources(sources)

            self.stats["nodes_created"] = self.kg_builder.stats["total_nodes"]
            self.stats["relations_created"] = self.kg_builder.stats["total_relations"]

            print(f"✅ Grafo construido en Neo4j")

            return True

        except Exception as e:
            print(f"❌ Error construyendo grafo: {e}")
            import traceback

            traceback.print_exc()
            return False

    def step_5_build_hnsw_index(self) -> bool:
        """
        PASO 5: Construir índice HNSW

        Output: data/indices/hnsw_index.bin

        IMPORTANTE: Requiere que TODOS los chunks tengan embeddings
        """
        print("\n" + "=" * 70)
        print("🔍 PASO 5: CONSTRUCCIÓN DE ÍNDICE HNSW")
        print("=" * 70)

        if not self.output_dir.exists():
            print(f"❌ Directorio procesado no encontrado: {self.output_dir}")
            return False

        try:
            # ⭐ VERIFICAR que TODOS los chunks tengan embeddings
            chunk_files = list(self.output_dir.glob("*_chunks.json"))

            if not chunk_files:
                print(f"❌ No se encontraron chunks en {self.output_dir}")
                return False

            print(f"\n📂 Verificando embeddings para {len(chunk_files)} documentos...")

            missing_embeddings = []
            existing_embeddings = []

            for chunk_file in chunk_files:
                embedding_file = chunk_file.parent / chunk_file.name.replace(
                    "_chunks.json", "_embeddings.npy"
                )
                doc_name = chunk_file.stem.replace("_chunks", "")

                if not embedding_file.exists():
                    missing_embeddings.append(doc_name)
                else:
                    existing_embeddings.append(doc_name)

            # Mostrar resumen
            print(f"   ✅ Con embeddings: {len(existing_embeddings)}/{len(chunk_files)}")
            print(f"   ❌ Sin embeddings:  {len(missing_embeddings)}/{len(chunk_files)}")

            # Si faltan embeddings, NO construir índice parcial
            if missing_embeddings:
                print(f"\n❌ FALTAN EMBEDDINGS PARA {len(missing_embeddings)} DOCUMENTOS:")
                for doc in missing_embeddings[:10]:  # Mostrar primeros 10
                    print(f"   ❌ {doc}")
                if len(missing_embeddings) > 10:
                    print(f"   ... y {len(missing_embeddings) - 10} más")

                print(f"\n⚠️  NO se construirá índice HNSW incompleto")
                print(f"   (índice parcial causaría inconsistencias en búsqueda híbrida)")
                print(f"\n💡 SOLUCIÓN:")
                print(f"   1. Genera los embeddings faltantes:")
                print(
                    f"      python generate_embeddings.py --input {self.output_dir} --batch-size 4"
                )
                print(f"   ")
                print(f"   2. Vuelve a ejecutar SOLO el paso 5:")
                print(
                    f"      python offline_etl/builders/hnsw_builder.py --input {self.output_dir}"
                )
                print(f"\n⚠️  Pipeline continúa (Neo4j está completo), pero HNSW queda vacío")

                self.stats["chunks_indexed"] = 0
                return True  # No crítico para continuar

            # ✅ Todos los embeddings presentes, construir índice
            print(f"\n✅ Todos los documentos tienen embeddings")

            # Construir índice
            self.hnsw_builder.build_from_directory(self.output_dir)

            # Verificar que se indexaron chunks
            if self.hnsw_builder.chunk_counter == 0:
                print(f"❌ No se indexaron chunks (error inesperado)")
                return False

            # Guardar índice
            self.hnsw_builder.save_index(self.hnsw_index_file)

            self.stats["chunks_indexed"] = self.hnsw_builder.chunk_counter

            print(f"\n✅ Índice HNSW completo y guardado:")
            print(f"   Archivo: {self.hnsw_index_file}")
            print(f"   Chunks:  {self.hnsw_builder.chunk_counter}")
            print(f"   Docs:    {len(chunk_files)}")

            return True

        except Exception as e:
            print(f"❌ Error construyendo índice HNSW: {e}")
            import traceback

            traceback.print_exc()
            return True  # No crítico

    def run(self) -> bool:
        """
        Ejecuta el pipeline ETL completo

        Returns:
            bool: True si todo fue exitoso
        """
        self.stats["start_time"] = datetime.now()

        print("\n" + "=" * 70)
        print("🚀 ETL PIPELINE - CONSTRUCCIÓN DEL KNOWLEDGE GRAPH ACADÉMICO")
        print("=" * 70)
        print(f"\n📅 Inicio: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📂 Configuración:")
        print(f"   Input:  {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Curso:  {self.course_code}")
        print(f"   Modelo: {self.embedding_model}")

        # Pipeline steps
        steps = [
            ("Extracción de Sílabos", self.step_1_extract_from_syllabi),
            ("Procesamiento de Documentos", self.step_2_extract_from_documents),
            ("Document-Topic Linking", self.step_2_5_link_documents_to_topics),  # NUEVO
            ("Concept Linking", self.step_3_link_concepts),
            ("Construcción del Grafo Neo4j", self.step_4_build_neo4j_graph),
            ("Construcción del Índice HNSW", self.step_5_build_hnsw_index),
        ]

        success_count = 0
        for i, (step_name, step_func) in enumerate(steps, 1):
            print(f"\n{'=' * 70}")
            print(f"PASO {i}/{len(steps)}: {step_name}")
            print(f"{'=' * 70}")

            success = step_func()
            if success:
                success_count += 1
                print(f"\n✅ {step_name} completado")
            else:
                print(f"\n❌ {step_name} falló")
                # Algunos pasos son opcionales
                if step_name not in ["Concept Linking"]:
                    print("❌ Pipeline detenido por error crítico")
                    return False

        self.stats["end_time"] = datetime.now()
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        # Resumen final
        print("\n" + "=" * 70)
        print("📊 RESUMEN DE CONSTRUCCIÓN")
        print("=" * 70)
        print(f"\n⏱️  Duración: {duration:.1f} segundos")
        print(f"✅ Pasos exitosos: {success_count}/{len(steps)}")
        print(f"\n📈 Estadísticas:")
        print(f"   Sílabos procesados:    {self.stats['syllabi_processed']}")
        print(f"   Documentos procesados: {self.stats['documents_processed']}")
        print(f"   Nodos creados:         {self.stats['nodes_created']}")
        print(f"   Relaciones creadas:    {self.stats['relations_created']}")
        print(f"   Chunks indexados:      {self.stats['chunks_indexed']}")

        # Instrucciones finales según el estado
        if self.stats["chunks_indexed"] == 0:
            print(f"\n📝 PASOS SIGUIENTES:")
            print(f"   Los embeddings se generan en un paso separado:")
            print(f"   ")
            print(f"   1. Generar embeddings:")
            print(
                f"      python utils/generate_embeddings.py --input {self.output_dir} --batch-size 4"
            )
            print(f"   ")
            print(f"   2. Reconstruir solo el índice HNSW (no re-ejecutar todo el pipeline):")
            print(f"      python offline_etl/builders/hnsw_builder.py --input {self.output_dir}")
            print(f"   ")
        elif self.stats["chunks_indexed"] < self.stats["documents_processed"] * 10:  # Estimación
            print(f"\n⚠️  ATENCIÓN: Índice HNSW parcial ({self.stats['chunks_indexed']} chunks)")
            print(f"   Algunos documentos no tienen embeddings.")
            print(f"   ")
            print(f"   Para completar:")
            print(
                f"   1. python utils/generate_embeddings.py --input {self.output_dir} --batch-size 4"
            )
            print(f"   2. python offline_etl/builders/hnsw_builder.py --input {self.output_dir}")
            print(f"   ")
        else:
            print(f"\n✅ Sistema completo - índice HNSW con todos los chunks")

        if success_count == len(steps):
            print("\n✅ ¡PIPELINE ETL COMPLETADO!")
            print("\n🌐 Siguientes pasos:")
            print("   1. Verificar Neo4j: http://localhost:7474")
            print("   2. Iniciar servidor MCP: python runtime_mcp/academic_mcp_server.py")

            print("   1. Abre Neo4j Browser: http://localhost:7474")
            print("   2. Inicia el servidor MCP: python runtime_mcp/academic_mcp_server.py")
            return True
        else:
            print("\n⚠️  Construcción parcial. Revisa los errores arriba.")
            return False

    def cleanup(self):
        """Limpia recursos"""
        self.kg_builder.close()


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description="ETL Pipeline para Construcción del Knowledge Graph Académico"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/user-data/uploads"),
        help="Directorio con PDFs de entrada",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directorio para chunks procesados",
    )

    parser.add_argument(
        "--course-code",
        type=str,
        default="INF265",
        help="Código del curso (ej: INF265)",
    )

    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="URI de Neo4j",
    )

    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Usuario de Neo4j",
    )

    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="12345678",
        help="Contraseña de Neo4j",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
        help="Modelo de embeddings (multilingüe)",
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Límite de documentos a procesar (para pruebas). Default: todos",
    )

    parser.add_argument(
        "--use-cso",
        action="store_true",
        help="Usar CSO (Computer Science Ontology) para enriquecimiento semántico",
    )

    parser.add_argument(
        "--cso-file",
        type=str,
        default="CSO.3.5.ttl",
        help="Ruta al archivo CSO.ttl (default: CSO.3.5.ttl)",
    )

    args = parser.parse_args()

    # Crear pipeline
    pipeline = ETLPipeline(
        input_dir=args.input,
        output_dir=args.output,
        course_code=args.course_code,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        embedding_model=args.embedding_model,
        max_documents=args.max_documents,
        use_cso=args.use_cso,
        cso_file=args.cso_file if args.use_cso else None,
    )

    try:
        # Ejecutar pipeline
        success = pipeline.run()
        sys.exit(0 if success else 1)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()

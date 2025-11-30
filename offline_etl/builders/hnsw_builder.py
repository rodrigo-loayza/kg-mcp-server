# hnsw_builder.py
"""
🔍 HNSW Builder - Constructor del Índice Vectorial
Extrae funcionalidad de indexación de etl_pipeline.py

Responsabilidades:
- Cargar chunks procesados
- Construir índice HNSW
- Guardar/cargar índice
- Búsqueda vectorial eficiente

Autor: Rodrigo Cárdenas
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import hnswlib
from sentence_transformers import SentenceTransformer


class HNSWBuilder:
    """Constructor del índice HNSW para búsqueda vectorial"""

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        dimension: int = 768,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        """
        Inicializa builder HNSW

        Args:
            embedding_model: Modelo de embeddings (multilingüe)
            dimension: Dimensión de embeddings (768 para multilingüe, 384 para simples)
            M: Parámetro M de HNSW (mayor = más conexiones, más memoria)
            ef_construction: Parámetro de construcción (mayor = mejor calidad, más lento)
            ef_search: Parámetro de búsqueda (mayor = mejor recall, más lento)
        """
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Inicializar índice HNSW
        self.index = hnswlib.Index(space="cosine", dim=dimension)
        self.index_initialized = False

        # Modelo de embeddings
        print(f"🔧 Cargando modelo de embeddings: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        print(f"   ✅ Modelo cargado (dimensión: {dimension})")

        # Mapeos
        self.id_to_chunk = {}
        self.chunk_counter = 0

    def initialize_index(self, max_elements: int = 10000):
        """
        Inicializa el índice HNSW con capacidad máxima

        Args:
            max_elements: Número máximo de elementos a indexar
        """
        print(f"\n🔧 Inicializando índice HNSW...")
        print(f"   Max elements: {max_elements}")
        print(f"   M: {self.M}")
        print(f"   ef_construction: {self.ef_construction}")

        self.index.init_index(
            max_elements=max_elements, ef_construction=self.ef_construction, M=self.M
        )

        self.index.set_ef(self.ef_search)
        self.index_initialized = True

        print(f"   ✅ Índice inicializado")

    def add_chunks_from_file(self, chunk_file: Path):
        """
        Agrega chunks desde un archivo JSON con embeddings

        Args:
            chunk_file: Ruta al archivo *_chunks.json
        """
        # Cargar chunks
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Cargar embeddings
        embedding_file = chunk_file.parent / chunk_file.name.replace(
            "_chunks.json", "_embeddings.npy"
        )

        if not embedding_file.exists():
            print(f"   ⚠️  Embeddings no encontrados: {embedding_file.name}")
            return 0

        embeddings = np.load(embedding_file)

        # Agregar al índice
        added = 0
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = self.chunk_counter
            self.id_to_chunk[chunk_id] = chunk

            # Agregar al índice HNSW
            self.index.add_items(embedding.reshape(1, -1), np.array([chunk_id]))

            self.chunk_counter += 1
            added += 1

        return added

    def build_from_directory(self, processed_dir: Path):
        """
        Construye índice HNSW desde directorio de chunks procesados

        Args:
            processed_dir: Directorio con *_chunks.json y *_embeddings.npy
        """
        print(f"\n📂 Construyendo índice desde: {processed_dir}")

        # Buscar archivos de chunks
        chunk_files = list(processed_dir.glob("*_chunks.json"))

        if not chunk_files:
            print(f"❌ No se encontraron chunks en {processed_dir}")
            return

        print(f"📄 Encontrados {len(chunk_files)} archivos de chunks")

        # Contar total de chunks para inicializar índice
        total_chunks = 0
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                    total_chunks += len(chunks)
            except UnicodeDecodeError as e:
                print(f"   ⚠️  Error de encoding en {chunk_file.name}: {e}")
                print(f"      Intentando con latin-1...")
                try:
                    with open(chunk_file, "r", encoding="latin-1") as f:
                        chunks = json.load(f)
                        total_chunks += len(chunks)
                except Exception as e2:
                    print(f"      ❌ No se pudo leer {chunk_file.name}: {e2}")
                    continue
            except Exception as e:
                print(f"   ⚠️  Error leyendo {chunk_file.name}: {e}")
                continue

        # Inicializar índice con capacidad suficiente
        self.initialize_index(max_elements=max(total_chunks, 10000))

        # Agregar chunks al índice
        print(f"\n📥 Agregando chunks al índice...")

        for chunk_file in chunk_files:
            print(f"   📖 Procesando: {chunk_file.name}")
            added = self.add_chunks_from_file(chunk_file)
            print(f"      ✅ {added} chunks agregados")

        print(f"\n✅ Índice HNSW construido: {self.chunk_counter} chunks totales")

    def save_index(self, filepath: Path):
        """
        Guarda el índice HNSW en disco

        Args:
            filepath: Ruta donde guardar el índice (ej: hnsw_index.bin)
        """
        if not self.index_initialized:
            print("❌ Índice no inicializado, no se puede guardar")
            return

        # Guardar índice HNSW
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(filepath))

        # Guardar mapeos
        mapping_file = filepath.parent / f"{filepath.stem}_mappings.json"
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id_to_chunk": {str(k): v for k, v in self.id_to_chunk.items()},
                    "chunk_counter": self.chunk_counter,
                    "dimension": self.dimension,
                    "M": self.M,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"💾 Índice HNSW guardado:")
        print(f"   Índice:   {filepath}")
        print(f"   Mappings: {mapping_file}")

    def load_index(self, filepath: Path):
        """
        Carga un índice HNSW desde disco

        Args:
            filepath: Ruta al archivo del índice
        """
        # Cargar índice HNSW
        self.index.load_index(str(filepath))
        self.index_initialized = True

        # Cargar mapeos
        mapping_file = filepath.parent / f"{filepath.stem}_mappings.json"

        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.id_to_chunk = {int(k): v for k, v in data["id_to_chunk"].items()}
            self.chunk_counter = data["chunk_counter"]

            # Restaurar parámetros
            self.dimension = data.get("dimension", self.dimension)
            self.M = data.get("M", self.M)
            self.ef_construction = data.get("ef_construction", self.ef_construction)
            self.ef_search = data.get("ef_search", self.ef_search)

        # Configurar ef de búsqueda
        self.index.set_ef(self.ef_search)

        print(f"✅ Índice HNSW cargado:")
        print(f"   Chunks: {self.chunk_counter}")
        print(f"   Dimensión: {self.dimension}")
        print(f"   ef_search: {self.ef_search}")

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Busca chunks similares a la query

        Args:
            query: Texto de consulta
            k: Número de resultados a retornar

        Returns:
            Lista de dicts con chunk_id, content, score, metadata
        """
        if not self.index_initialized:
            print("❌ Índice no inicializado")
            return []

        # Generar embedding de query
        query_embedding = self.model.encode([query])[0]

        # Buscar en HNSW
        labels, distances = self.index.knn_query(query_embedding, k=k)

        # Construir resultados
        results = []
        for label, distance in zip(labels[0], distances[0]):
            chunk = self.id_to_chunk[label]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "score": float(1 - distance),  # Convertir distancia a similitud
                    "doc_id": chunk["doc_id"],
                    "metadata": chunk.get("metadata", {}),
                }
            )

        return results


def main():
    """
    Script standalone para (re)construir índice HNSW

    Uso típico después de generar embeddings:
        python offline_etl/builders/hnsw_builder.py --input data/processed
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Construye índice HNSW desde chunks con embeddings"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed"),
        help="Directorio con *_chunks.json y *_embeddings.npy",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/indices/hnsw_index.bin"),
        help="Archivo de salida para índice HNSW",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
        help="Modelo de embeddings (debe coincidir con el usado para generar .npy)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🔍 CONSTRUCCIÓN DE ÍNDICE HNSW")
    print("=" * 70)
    print(f"\n📂 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    print(f"🤖 Modelo: {args.model}\n")

    # Crear builder
    builder = HNSWBuilder(embedding_model=args.model)

    # Construir índice
    print("🔧 Construyendo índice desde embeddings...")
    builder.build_from_directory(args.input)

    if builder.chunk_counter == 0:
        print("\n❌ No se encontraron embeddings")
        print("   Primero genera embeddings con:")
        print("   python utils/generate_embeddings.py --input data/processed --batch-size 4")
        sys.exit(1)

    # Guardar índice
    print(f"\n💾 Guardando índice...")
    builder.save_index(args.output)

    print("\n" + "=" * 70)
    print("✅ ÍNDICE HNSW CONSTRUIDO EXITOSAMENTE")
    print("=" * 70)
    print(f"\n📊 Estadísticas:")
    print(f"   Chunks indexados: {builder.chunk_counter}")
    print(f"   Dimensión: {builder.dimension}")
    print(f"   Archivo: {args.output}")
    print(f"\n🎉 ¡Listo para búsquedas híbridas!")


if __name__ == "__main__":
    main()

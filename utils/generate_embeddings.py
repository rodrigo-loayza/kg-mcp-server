#!/usr/bin/env python3
# utils/generate_embeddings.py
"""
🔢 Generador de Embeddings - Proceso Separado
Script standalone para generar embeddings de chunks procesados

USO:
    python utils/generate_embeddings.py --input data/processed --batch-size 4

Para Windows con poca memoria:
    python utils/generate_embeddings.py --input data/processed --batch-size 4 --model-small

Autor: Rodrigo Cárdenas
"""

import json
import argparse
import gc
import sys
from pathlib import Path
from typing import List

# Agregar directorio padre al path para imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers no instalado")
    print("   Ejecuta: pip install sentence-transformers")
    exit(1)


class EmbeddingGenerator:
    """Generador de embeddings con manejo robusto de memoria"""

    def __init__(
        self, model_name: str = "paraphrase-multilingual-mpnet-base-v2", batch_size: int = 16
    ):
        """
        Args:
            model_name: Nombre del modelo de Sentence Transformers
            batch_size: Tamaño de batch para procesamiento
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def load_model(self):
        """Carga el modelo de embeddings"""
        if self.model is None:
            print(f"🔄 Cargando modelo: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"   ✅ Modelo cargado")

    def generate_for_file(self, chunk_file: Path) -> bool:
        """
        Genera embeddings para un archivo de chunks

        Args:
            chunk_file: Ruta al archivo *_chunks.json

        Returns:
            True si exitoso, False si falla
        """
        # Verificar si ya existen embeddings
        embedding_file = chunk_file.parent / chunk_file.name.replace(
            "_chunks.json", "_embeddings.npy"
        )

        if embedding_file.exists():
            print(f"   ⏭️  Ya existe: {embedding_file.name}")
            return True

        try:
            # Cargar chunks
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if not chunks:
                print(f"   ⚠️  Archivo vacío: {chunk_file.name}")
                return False

            print(f"   📖 Procesando: {chunk_file.name} ({len(chunks)} chunks)")

            # Extraer textos
            texts = [chunk.get("content", "") for chunk in chunks]
            texts = [t for t in texts if t]  # Filtrar vacíos

            if not texts:
                print(f"   ⚠️  No hay contenido en chunks")
                return False

            # Asegurar que el modelo esté cargado
            self.load_model()

            # Generar embeddings en batches
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # Mostrar progreso
                progress = min(i + self.batch_size, len(texts))
                print(f"      🔢 Batch {i//self.batch_size + 1}: {progress}/{len(texts)}", end="\r")

                try:
                    batch_embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=min(8, len(batch)),  # Batch interno pequeño
                    )
                    all_embeddings.append(batch_embeddings)

                    # Liberar memoria explícitamente cada N batches
                    if (i // self.batch_size) % 10 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"\n      ❌ Error en batch {i//self.batch_size + 1}: {e}")
                    return False

            print()  # Nueva línea después del progreso

            # Concatenar todos los batches
            embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

            # Guardar embeddings
            np.save(embedding_file, embeddings)
            print(f"   ✅ Guardado: {embedding_file.name} ({embeddings.shape})")

            # Liberar memoria
            del embeddings
            del all_embeddings
            gc.collect()

            return True

        except MemoryError as e:
            print(f"   ❌ Error de memoria: {e}")
            print(f"      💡 Intenta con --batch-size 4 o --model-small")
            return False

        except OSError as e:
            if "1455" in str(e):
                print(f"   ❌ Archivo de paginación insuficiente (Windows)")
                print(f"      💡 Soluciones:")
                print(f"         1. Cierra otras aplicaciones")
                print(f"         2. Aumenta archivo de paginación de Windows")
                print(f"         3. Usa --batch-size 4")
                print(f"         4. Usa --model-small")
            else:
                print(f"   ❌ Error del sistema: {e}")
            return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def generate_for_directory(self, processed_dir: Path) -> dict:
        """
        Genera embeddings para todos los chunks en un directorio

        Args:
            processed_dir: Directorio con *_chunks.json

        Returns:
            Dict con estadísticas
        """
        print(f"\n📂 Buscando chunks en: {processed_dir}")

        chunk_files = list(processed_dir.glob("*_chunks.json"))

        if not chunk_files:
            print(f"❌ No se encontraron chunks en {processed_dir}")
            return {"total": 0, "success": 0, "failed": 0}

        print(f"📄 Encontrados {len(chunk_files)} archivos de chunks\n")

        stats = {"total": len(chunk_files), "success": 0, "failed": 0, "skipped": 0}

        for chunk_file in chunk_files:
            success = self.generate_for_file(chunk_file)

            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1

        return stats

    def cleanup(self):
        """Libera memoria del modelo"""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            print("\n🧹 Modelo descargado de memoria")


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(description="Generador de Embeddings para Chunks Procesados")

    parser.add_argument(
        "--input", type=Path, default=Path("data/processed"), help="Directorio con *_chunks.json"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamaño de batch (4-32, usa 4-8 para poca memoria)",
    )

    parser.add_argument(
        "--model-small",
        action="store_true",
        help="Usar modelo pequeño (all-MiniLM-L6-v2, 384 dims)",
    )

    args = parser.parse_args()

    # Seleccionar modelo
    if args.model_small:
        model_name = "all-MiniLM-L6-v2"  # Más pequeño, más rápido
        print("⚠️  Usando modelo pequeño (384 dimensiones)")
    else:
        model_name = "paraphrase-multilingual-mpnet-base-v2"  # Mejor calidad

    print("=" * 70)
    print("🔢 GENERADOR DE EMBEDDINGS")
    print("=" * 70)
    print(f"\n📂 Input: {args.input}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"🤖 Modelo: {model_name}\n")

    # Crear generador
    generator = EmbeddingGenerator(model_name=model_name, batch_size=args.batch_size)

    try:
        # Generar embeddings
        stats = generator.generate_for_directory(args.input)

        # Resumen
        print("\n" + "=" * 70)
        print("📊 RESUMEN")
        print("=" * 70)
        print(f"\n✅ Exitosos: {stats['success']}/{stats['total']}")
        print(f"❌ Fallidos:  {stats['failed']}/{stats['total']}")

        if stats["failed"] > 0:
            print("\n💡 Si hay errores de memoria:")
            print("   1. Ejecuta con --batch-size 4")
            print("   2. Usa --model-small")
            print("   3. Cierra otras aplicaciones")
            print("   4. Procesa menos documentos a la vez")

        if stats["success"] > 0:
            print("\n🎉 ¡Embeddings generados exitosamente!")

    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()

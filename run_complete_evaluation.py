#!/usr/bin/env python3
"""
Script Maestro de Evaluación Completa

Ejecuta todas las pruebas necesarias para validar los IOVs
de la tesis y genera un reporte completo.

Uso:
    python run_complete_evaluation.py
"""

import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime


class MasterEvaluator:
    """Orquestador de evaluación completa"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": [],
            "iov_compliance": {},
            "errors": []
        }
    
    def run_command(self, cmd: list, description: str) -> bool:
        """Ejecuta un comando y registra resultado"""
        
        print(f"\n{'='*70}")
        print(f"▶️ {description}")
        print(f"{'='*70}\n")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(result.stdout)
            
            self.results["tests_run"].append({
                "description": description,
                "command": " ".join(cmd),
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Error ejecutando: {description}")
            print(f"   {e.stderr}")
            
            self.results["tests_run"].append({
                "description": description,
                "command": " ".join(cmd),
                "status": "error",
                "error": e.stderr,
                "timestamp": datetime.now().isoformat()
            })
            
            self.results["errors"].append({
                "test": description,
                "error": str(e)
            })
            
            return False
    
    def check_prerequisites(self) -> bool:
        """Verifica que todos los prerequisitos estén listos"""
        
        print("\n" + "="*70)
        print("🔍 VERIFICANDO PREREQUISITOS")
        print("="*70 + "\n")
        
        all_ok = True
        
        # 1. Verificar Neo4j
        print("1. Verificando Neo4j...")
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "12345678")
            )
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    print("   ✅ Neo4j está corriendo y accesible")
                else:
                    print("   ❌ Neo4j responde pero hay un problema")
                    all_ok = False
            driver.close()
        except Exception as e:
            print(f"   ❌ Neo4j no accesible: {e}")
            all_ok = False
        
        # 2. Verificar índice HNSW
        print("\n2. Verificando índice HNSW...")
        index_path = Path("data/indices/hnsw_index.bin")
        if index_path.exists():
            print(f"   ✅ Índice encontrado: {index_path}")
        else:
            print(f"   ❌ Índice no encontrado: {index_path}")
            all_ok = False
        
        # 3. Verificar chunks procesados
        print("\n3. Verificando chunks procesados...")
        chunks_dir = Path("data/processed/chunks")
        if chunks_dir.exists():
            chunk_files = list(chunks_dir.glob("*_chunks.json"))
            print(f"   ✅ Directorio encontrado: {len(chunk_files)} documentos")
        else:
            print(f"   ❌ Directorio no encontrado: {chunks_dir}")
            all_ok = False
        
        # 4. Verificar queries de prueba
        print("\n4. Verificando queries de prueba...")
        queries_file = Path("test_queries.json")
        if queries_file.exists():
            with open(queries_file, "r") as f:
                data = json.load(f)
                queries = data.get("queries", [])
                print(f"   ✅ Archivo encontrado: {len(queries)} queries")
        else:
            print(f"   ❌ Archivo no encontrado: {queries_file}")
            all_ok = False
        
        # 5. Verificar dependencias Python
        print("\n5. Verificando dependencias Python...")
        required_packages = [
            "neo4j",
            "sentence_transformers",
            "hnswlib",
            "numpy"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} no instalado")
                all_ok = False
        
        print()
        if all_ok:
            print("✅ Todos los prerequisitos están listos\n")
        else:
            print("❌ Algunos prerequisitos faltan. Revisa los errores arriba.\n")
        
        return all_ok
    
    def run_evaluation(self):
        """Ejecuta evaluación completa"""
        
        print("\n" + "="*70)
        print("🎯 INICIANDO EVALUACIÓN COMPLETA DEL SISTEMA")
        print("="*70)
        print(f"Timestamp: {self.results['timestamp']}")
        print()
        
        # 1. Verificar prerequisitos
        if not self.check_prerequisites():
            print("❌ No se pueden ejecutar las pruebas sin los prerequisitos")
            return False
        
        input("⏸️ Presiona Enter para continuar con las pruebas...")
        
        # 2. Test del motor directamente
        success = self.run_command(
            ["python", "test_engines_direct.py"],
            "TEST 1: Motor de búsqueda directo (sin MCP)"
        )
        
        if not success:
            print("⚠️ Continuando a pesar del error...")
        
        input("\n⏸️ Presiona Enter para continuar...")
        
        # 3. Visualización del grafo
        success = self.run_command(
            ["python", "visualize_knowledge_graph.py"],
            "TEST 2: Visualización del grafo de conocimiento"
        )
        
        if not success:
            print("⚠️ Continuando a pesar del error...")
        
        input("\n⏸️ Presiona Enter para continuar...")
        
        # 4. Evaluación con métricas
        success = self.run_command(
            ["python", "evaluate_complete_system.py"],
            "TEST 3: Evaluación con métricas (nDCG, Recall, Precision)"
        )
        
        if not success:
            print("⚠️ Continuando a pesar del error...")
        
        input("\n⏸️ Presiona Enter para continuar...")
        
        # 5. Test del servidor MCP local
        success = self.run_command(
            ["python", "test_mcp_server_local.py"],
            "TEST 4: Servidor MCP (prueba local)"
        )
        
        if not success:
            print("⚠️ Continuando a pesar del error...")
        
        # 6. Analizar resultados
        self.analyze_results()
        
        # 7. Generar reporte
        self.generate_report()
        
        return True
    
    def analyze_results(self):
        """Analiza resultados de evaluación"""
        
        print("\n" + "="*70)
        print("📊 ANALIZANDO RESULTADOS")
        print("="*70 + "\n")
        
        # Cargar resultados de evaluación si existen
        eval_results_file = Path("evaluation_results.json")
        
        if eval_results_file.exists():
            with open(eval_results_file, "r") as f:
                eval_results = json.load(f)
            
            # Analizar cumplimiento de IOVs
            proposed = eval_results.get("proposed", {})
            iov_compliance = eval_results.get("iov_compliance", {})
            
            self.results["iov_compliance"] = {
                "OE2_RE2.3_IOV1_ndcg": {
                    "description": "nDCG@10 ≥0.75",
                    "target": 0.75,
                    "achieved": proposed.get("mean_ndcg@10", 0),
                    "compliant": iov_compliance.get("IOV1_ndcg_ge_075", False)
                },
                "OE2_RE2.3_IOV2_recall_improvement": {
                    "description": "Mejora ≥25% en Recall@10 vs baseline",
                    "target": 25,
                    "achieved": eval_results.get("comparison", {}).get("recall_improvement_percent", 0),
                    "compliant": iov_compliance.get("IOV2_recall_improvement_ge_25pct", False)
                }
            }
            
            print("📈 CUMPLIMIENTO DE IOVs:\n")
            
            for iov_id, iov_data in self.results["iov_compliance"].items():
                status = "✅" if iov_data["compliant"] else "❌"
                print(f"{status} {iov_id}")
                print(f"   {iov_data['description']}")
                print(f"   Meta: {iov_data['target']}")
                print(f"   Alcanzado: {iov_data['achieved']}")
                print()
        
        else:
            print("⚠️ No se encontraron resultados de evaluación")
            print(f"   Archivo esperado: {eval_results_file}")
    
    def generate_report(self):
        """Genera reporte final"""
        
        print("\n" + "="*70)
        print("📝 GENERANDO REPORTE FINAL")
        print("="*70 + "\n")
        
        report_file = Path("EVALUATION_REPORT.json")
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Reporte guardado en: {report_file}\n")
        
        # Imprimir resumen
        print("📊 RESUMEN DE EVALUACIÓN:")
        print(f"   Tests ejecutados: {len(self.results['tests_run'])}")
        print(f"   Tests exitosos: {sum(1 for t in self.results['tests_run'] if t['status'] == 'success')}")
        print(f"   Tests con error: {sum(1 for t in self.results['tests_run'] if t['status'] == 'error')}")
        print(f"   IOVs evaluados: {len(self.results['iov_compliance'])}")
        print()
        
        if self.results["iov_compliance"]:
            compliant_iovs = sum(1 for iov in self.results['iov_compliance'].values() if iov['compliant'])
            total_iovs = len(self.results['iov_compliance'])
            print(f"   IOVs cumplidos: {compliant_iovs}/{total_iovs}")
        
        if self.results["errors"]:
            print(f"\n⚠️ Errores encontrados: {len(self.results['errors'])}")
            for error in self.results["errors"]:
                print(f"   - {error['test']}")


def main():
    """Función principal"""
    
    evaluator = MasterEvaluator()
    
    try:
        evaluator.run_evaluation()
        
        print("\n" + "="*70)
        print("✅ EVALUACIÓN COMPLETA FINALIZADA")
        print("="*70)
        print()
        print("📂 Archivos generados:")
        print("   - EVALUATION_REPORT.json (reporte completo)")
        print("   - evaluation_results.json (métricas detalladas)")
        print("   - graph_visualizations/ (visualizaciones del grafo)")
        print()
        print("🎯 Próximos pasos:")
        print("   1. Revisa EVALUATION_REPORT.json para ver el cumplimiento de IOVs")
        print("   2. Usa las visualizaciones del grafo para tu tesis")
        print("   3. Incluye las métricas en tus secciones de resultados")
        print()
    
    except KeyboardInterrupt:
        print("\n\n🛑 Evaluación interrumpida por usuario")
    
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

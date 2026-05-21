from flask import Flask, jsonify, request
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic.semantic_service import SemanticService

app = Flask(__name__)

DB_PATH = os.environ.get("KANJI_DB_PATH", "data/processed/kanji_dict.db")
service = None

try:
    service = SemanticService(DB_PATH)
    print(f"SemanticService inicializado: {DB_PATH}")
except Exception as e:
    print(f"Aviso: {e}")
    print("Execute: python src/semantic/build_dictionary.py")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "kanji-n2-n5-semantic-api",
        "db_loaded": service is not None,
    })


@app.route("/kanji/<char>", methods=["GET"])
def get_kanji(char):
    if not service:
        return jsonify({"error": "Serviço não inicializado. Execute build_dictionary.py."}), 503
    result = service.get_kanji_info(char)
    return jsonify(result), 200 if result["found"] else 404


@app.route("/batch", methods=["POST"])
def batch_lookup():
    """Consulta múltiplos kanjis de uma vez (máx. 50)."""
    if not service:
        return jsonify({"error": "Serviço não inicializado."}), 503
    data = request.get_json()
    chars = data.get("kanjis", []) if data else []
    if not chars:
        return jsonify({"error": "Campo 'kanjis' é obrigatório."}), 400
    results = [service.get_kanji_info(c) for c in chars[:50]]
    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Iniciando API Semântica N2-N5 na porta {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

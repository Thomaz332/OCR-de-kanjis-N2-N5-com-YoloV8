import sqlite3
import json
import requests
import os


class SemanticService:
    """Consulta informações semânticas de kanjis a partir do banco Kanjidic2 local."""

    def __init__(self, db_path="data/processed/kanji_dict.db"):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Banco de dados não encontrado: {db_path}\n"
                "Execute: python src/semantic/build_dictionary.py"
            )

    def get_kanji_info(self, kanji_char):
        """
        Retorna um dicionário com significados, leituras e nível JLPT.
        Prioriza banco local; fallback para kanjiapi.dev.
        """
        result = {
            "kanji": kanji_char,
            "found": False,
            "source": None,
            "jisho_url": f"https://jisho.org/search/{kanji_char}%23kanji",
        }

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM kanji WHERE literal = ?", (kanji_char,))
        row = c.fetchone()
        conn.close()

        if row:
            result["found"] = True
            result["source"] = "local_kanjidic2"
            result["data"] = {
                "grade":        row[1],
                "jlpt":         row[2],
                "freq":         row[3],
                "stroke_count": row[4],
                "meanings":     json.loads(row[5]),
                "readings_on":  json.loads(row[6]),
                "readings_kun": json.loads(row[7]),
                "nanori":       json.loads(row[8]),
            }
            return result

        try:
            resp = requests.get(
                f"https://kanjiapi.dev/v1/kanji/{kanji_char}", timeout=3
            )
            if resp.status_code == 200:
                data = resp.json()
                result["found"] = True
                result["source"] = "online_kanjiapi"
                result["data"] = {
                    "grade":        data.get("grade"),
                    "jlpt":         data.get("jlpt"),
                    "freq":         None,
                    "stroke_count": data.get("stroke_count"),
                    "meanings":     data.get("meanings", []),
                    "readings_on":  data.get("on_readings", []),
                    "readings_kun": data.get("kun_readings", []),
                    "nanori":       data.get("nanori", []),
                }
        except Exception:
            pass

        return result


if __name__ == "__main__":
    service = SemanticService()
    for char in ["明", "日", "学", "xyz"]:
        print(f"\n--- {char} ---")
        print(json.dumps(service.get_kanji_info(char), indent=2, ensure_ascii=False))

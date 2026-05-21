#!/usr/bin/env python3
"""
Constrói o banco SQLite de kanjis a partir do Kanjidic2 (padrão ouro open-source).
Baixa automaticamente o XML comprimido do EDRDG e indexa todos os caracteres.

Esquema:
  kanji(literal PK, grade, jlpt, freq, stroke_count, meanings, readings_on, readings_kun, nanori)
"""
import xml.etree.ElementTree as ET
import gzip
import json
import sqlite3
import os
import requests
from tqdm import tqdm

KANJIDIC_URL = "http://www.edrdg.org/kanjidic/kanjidic2.xml.gz"


def download_kanjidic(output_path):
    print(f"Baixando Kanjidic2 de {KANJIDIC_URL}...")
    response = requests.get(KANJIDIC_URL, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download concluído.")


def parse_and_index(xml_path, db_path):
    print(f"Parseando {xml_path} e indexando em {db_path}...")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS kanji (
            literal      TEXT PRIMARY KEY,
            grade        INTEGER,
            jlpt         INTEGER,
            freq         INTEGER,
            stroke_count INTEGER,
            meanings     TEXT,
            readings_on  TEXT,
            readings_kun TEXT,
            nanori       TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_jlpt ON kanji(jlpt)")
    conn.commit()

    count = 0
    batch = []

    with gzip.open(xml_path, "rt", encoding="utf-8") as f:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag != "character":
                continue

            literal = elem.find("literal").text
            misc = elem.find("misc")

            def _get(parent, tag):
                el = parent.find(tag) if parent is not None else None
                return el.text if el is not None else None

            grade        = _get(misc, "grade")
            jlpt         = _get(misc, "jlpt")
            freq         = _get(misc, "freq")
            stroke_count = _get(misc, "stroke_count")

            meanings, readings_on, readings_kun, nanori = [], [], [], []
            for group in elem.findall("reading_meaning/rmgroup"):
                for m in group.findall("meaning"):
                    if m.get("m_lang") is None:
                        meanings.append(m.text)
                    elif m.get("m_lang") == "pt":
                        meanings.append(f"[PT] {m.text}")
                for r in group.findall("reading"):
                    rtype = r.get("r_type")
                    if rtype == "ja_on":
                        readings_on.append(r.text)
                    elif rtype == "ja_kun":
                        readings_kun.append(r.text)
            nanori = [n.text for n in elem.findall("reading_meaning/nanori")]

            batch.append((
                literal,
                int(grade) if grade else None,
                int(jlpt) if jlpt else None,
                int(freq) if freq else None,
                int(stroke_count) if stroke_count else None,
                json.dumps(meanings, ensure_ascii=False),
                json.dumps(readings_on, ensure_ascii=False),
                json.dumps(readings_kun, ensure_ascii=False),
                json.dumps(nanori, ensure_ascii=False),
            ))
            count += 1
            elem.clear()

            if len(batch) >= 1000:
                c.executemany("INSERT OR REPLACE INTO kanji VALUES (?,?,?,?,?,?,?,?,?)", batch)
                conn.commit()
                batch = []

    if batch:
        c.executemany("INSERT OR REPLACE INTO kanji VALUES (?,?,?,?,?,?,?,?,?)", batch)
        conn.commit()

    conn.close()
    print(f"Indexados {count} kanjis em {db_path}")


if __name__ == "__main__":
    data_dir = os.path.join("data", "raw")
    os.makedirs(data_dir, exist_ok=True)

    xml_path = os.path.join(data_dir, "kanjidic2.xml.gz")
    if not os.path.exists(xml_path):
        download_kanjidic(xml_path)

    db_path = os.path.join("data", "processed", "kanji_dict.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    parse_and_index(xml_path, db_path)

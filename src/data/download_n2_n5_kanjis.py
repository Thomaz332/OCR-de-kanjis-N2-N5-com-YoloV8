import requests
import json
import os

KANJI_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "davidluzgouveia/kanji-data/master/kanji.json"
)


def download_kanji_data():
    print("Baixando base de kanjis (davidluzgouveia/kanji-data)...")
    response = requests.get(KANJI_DATA_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def build_n2_n5_dataset(data, output_path):
    """Filtra N2-N5 e salva JSON com metadados úteis para OCR."""
    n2_n5 = {}

    for kanji, info in data.items():
        level = info.get("jlpt_new")
        if level not in (2, 3, 4, 5):
            continue
        n2_n5[kanji] = {
            "jlpt_level": level,
            "meanings": info.get("meanings", []),
            "readings_on": info.get("readings_on", []),
            "readings_kun": info.get("readings_kun", []),
            "strokes": info.get("strokes"),
            "freq": info.get("freq"),
        }

    print(f"Kanjis N2-N5 encontrados: {len(n2_n5)}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(n2_n5, f, ensure_ascii=False, indent=2)
    print(f"Dataset salvo em: {output_path}")
    return n2_n5


def create_n2_n5_names(data, output_path):
    """
    Gera o arquivo .names no formato YOLO.
    A última classe é UNKNOWN_N1: o modelo aprende a rejeitar
    kanjis N1 e delegar ao modelo especialista.
    """
    print("Filtrando kanjis JLPT N2-N5...")
    kanjis = sorted(
        k for k, v in data.items()
        if v.get("jlpt_new") in (2, 3, 4, 5)
    )
    kanjis.append("UNKNOWN_N1")

    with open(output_path, "w", encoding="utf-8") as f:
        for k in kanjis:
            f.write(k + "\n")

    print(f"Total de classes: {len(kanjis)} ({len(kanjis)-1} kanjis + UNKNOWN_N1)")
    print(f"Arquivo .names salvo em: {output_path}")


def main():
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    data = download_kanji_data()
    build_n2_n5_dataset(data, os.path.join(output_dir, "kanji_n2_n5_dataset.json"))
    create_n2_n5_names(data, os.path.join(output_dir, "n2_n5.names"))


if __name__ == "__main__":
    main()

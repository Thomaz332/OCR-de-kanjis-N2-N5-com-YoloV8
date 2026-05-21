import requests
import json
import os

KANJI_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "davidluzgouveia/kanji-data/master/kanji.json"
)


def download_kanji_data():
    print("Baixando base de kanjis...")
    response = requests.get(KANJI_DATA_URL)
    response.raise_for_status()
    return response.json()


def build_n1_dataset(data, output_path):
    """
    Cria dataset somente com kanjis JLPT N1
    mantendo informações úteis para OCR.
    """

    n1_kanjis = {}

    for kanji, info in data.items():

        # JLPT novo (N1–N5)
        if info.get("jlpt_new") != 1:
            continue

        n1_kanjis[kanji] = {
            "meanings": info.get("meanings", []),
            "readings_on": info.get("readings_on", []),
            "readings_kun": info.get("readings_kun", []),
            "strokes": info.get("strokes"),
            "freq": info.get("freq"),
        }

    print(f"N1 encontrados: {len(n1_kanjis)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(n1_kanjis, f, ensure_ascii=False, indent=2)

    print(f"Arquivo salvo em: {output_path}")

def create_n1_names(data, output_path):
    print("Filtrando kanjis JLPT N1...")

    n1_kanjis = []

    for kanji, info in data.items():
        if info.get("jlpt_new") == 1:
            n1_kanjis.append(kanji)

    n1_kanjis.sort()

    print(f"Total de kanjis N1 encontrados: {len(n1_kanjis)}")

    # salva no formato YOLO (.names)
    with open(output_path, "w", encoding="utf-8") as f:
        for k in n1_kanjis:
            f.write(k + "\n")

    print(f"Arquivo salvo em: {output_path}")

def main():
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    output_file_n1_dataset = os.path.join(output_dir, "kanji_n1_dataset.json")
    output_file_n1_names = os.path.join(output_dir, "n1.names")

    data = download_kanji_data()
    build_n1_dataset(data, output_file_n1_dataset)
    create_n1_names(data, output_file_n1_names)


if __name__ == "__main__":
    main()
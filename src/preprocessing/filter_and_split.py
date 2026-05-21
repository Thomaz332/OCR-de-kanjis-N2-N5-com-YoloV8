#!/usr/bin/env python3
"""
Filtra e separa o Manga109 em conjuntos de validação para o modelo N2-N5.

Produz:
  - data/processed/n2_n5/val/positive   — kanjis N2-N5 isolados (Ground Truth BBox)
  - data/processed/n2_n5/val/negative   — kanjis N1 como classe negativa (UNKNOWN_N1)
  - data/processed/n2_n5/val_complex    — frases completas para validação contextual
  - data/processed/n2_n5/val_complex_metadata.json
"""
import os
import json
import shutil
import argparse
from tqdm import tqdm


def load_class_list(names_path):
    with open(names_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return {name: i for i, name in enumerate(names)}, names


def save_yolo_label(output_path, class_id):
    with open(output_path, "w") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def filter_and_split(metadata_path, jlpt_path, output_dir, n2_n5_names_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(jlpt_path, "r", encoding="utf-8") as f:
        jlpt_map = json.load(f)

    n2_n5_map, n2_n5_names = load_class_list(n2_n5_names_path)
    unknown_n1_id = len(n2_n5_names) - 1  # UNKNOWN_N1 é sempre o último

    dirs = ["n2_n5/val/positive", "n2_n5/val/negative", "n2_n5/val_complex"]
    for d in dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    source_crops = os.path.join(os.path.dirname(metadata_path), "crops")
    stats = {"n2_n5": 0, "n1_negativo": 0, "desconhecido": 0, "complexo": 0}
    complex_gt = []

    for entry in tqdm(metadata, desc="Filtrando N2-N5"):
        text = entry.get("text", "")
        if not text:
            continue
        filename = entry["file"]
        src_path = os.path.join(source_crops, filename)
        if not os.path.exists(src_path):
            continue

        if len(text) == 1:
            if text not in jlpt_map:
                stats["desconhecido"] += 1
                continue

            level = jlpt_map[text]

            if level >= 2 and text in n2_n5_map:
                stats["n2_n5"] += 1
                dst = os.path.join(output_dir, "n2_n5/val/positive", filename)
                shutil.copy(src_path, dst)
                save_yolo_label(dst.replace(".jpg", ".txt"), n2_n5_map[text])

            elif level == 1:
                stats["n1_negativo"] += 1
                dst = os.path.join(output_dir, "n2_n5/val/negative", filename)
                shutil.copy(src_path, dst)
                save_yolo_label(dst.replace(".jpg", ".txt"), unknown_n1_id)

        else:
            kanjis_n2_n5 = [
                c for c in text
                if "一" <= c <= "龯"
                and c in jlpt_map
                and jlpt_map[c] >= 2
                and c in n2_n5_map
            ]
            if not kanjis_n2_n5 or len(complex_gt) >= 500:
                continue
            stats["complexo"] += 1
            dst = os.path.join(output_dir, "n2_n5/val_complex", filename)
            shutil.copy(src_path, dst)
            complex_gt.append({"file": filename, "gt_text": text, "kanjis": kanjis_n2_n5})

    with open(
        os.path.join(output_dir, "n2_n5/val_complex_metadata.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(complex_gt, f, ensure_ascii=False, indent=2)

    print("\nFiltro concluído:")
    print(f"  Positivos N2-N5:   {stats['n2_n5']}")
    print(f"  Negativos (N1):    {stats['n1_negativo']}")
    print(f"  Cenas complexas:   {stats['complexo']}")
    print(f"  Desconhecidos:     {stats['desconhecido']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--jlpt", default="data/raw/jlpt_kanjis.json")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--n2_n5_names", default="data/processed/n2_n5.names")
    args = parser.parse_args()
    filter_and_split(args.metadata, args.jlpt, args.output, args.n2_n5_names)

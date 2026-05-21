#!/usr/bin/env python3
"""
Validação do modelo N2-N5 em cenas complexas do Manga109.

Métrica principal: Recall de Kanjis
  = kanjis detectados presentes na frase GT / total de kanjis únicos na frase GT

Usa esta métrica pois o Manga109 provê ground truth textual (a frase),
não coordenadas individuais por caractere — impossibilitando o cálculo de mAP.
"""
import os
import json
import argparse
from tqdm import tqdm

from ultralytics import YOLO


def validate_complex_scenes(model_path, metadata_path, images_dir, conf=0.25):
    print(f"Carregando modelo: {model_path}")
    model = YOLO(model_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Validando em {len(metadata)} cenas complexas...")

    total_gt = 0
    total_acerto = 0
    log = []

    for entry in tqdm(metadata, desc="Validando"):
        img_path = os.path.join(images_dir, entry["file"])
        if not os.path.exists(img_path):
            continue

        gt_kanjis = set(entry.get("kanjis", []))
        if not gt_kanjis:
            continue

        results = model.predict(img_path, conf=conf, verbose=False)

        pred_kanjis = set()
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label != "UNKNOWN_N1":
                    pred_kanjis.add(label)

        acertos = gt_kanjis & pred_kanjis
        total_gt += len(gt_kanjis)
        total_acerto += len(acertos)

        log.append({
            "file": entry["file"],
            "gt": list(gt_kanjis),
            "pred": list(pred_kanjis),
            "recall": len(acertos) / len(gt_kanjis),
        })

    recall_geral = total_acerto / total_gt if total_gt > 0 else 0.0

    print(f"\n{'='*52}")
    print(f"  VALIDAÇÃO COMPLEXA — MODELO N2-N5")
    print(f"{'='*52}")
    print(f"  Cenas avaliadas:        {len(log)}")
    print(f"  Kanjis GT totais:       {total_gt}")
    print(f"  Kanjis detectados OK:   {total_acerto}")
    print(f"  Recall de Kanjis:       {recall_geral:.4f}  ({recall_geral*100:.2f}%)")
    print(f"{'='*52}")

    out_path = "validation_complex_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": {
                "total_cenas": len(log),
                "total_gt": total_gt,
                "total_acerto": total_acerto,
                "recall_geral": recall_geral,
            }, "por_cena": log},
            f, ensure_ascii=False, indent=2,
        )
    print(f"  Log detalhado: {out_path}")
    return recall_geral


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validação complexa do modelo N2-N5.")
    parser.add_argument("--model", required=True, help="Caminho para best.pt")
    parser.add_argument("--metadata", required=True, help="val_complex_metadata.json")
    parser.add_argument("--images", required=True, help="Pasta com imagens de validação")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    validate_complex_scenes(args.model, args.metadata, args.images, args.conf)

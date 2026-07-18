#!/usr/bin/env python3
"""
Validação em escala do corpus completo Manga109.

Para cada linha de texto com kanjis N2-N5:
  1. Extrai o crop da página
  2. Aplica filtros heurísticos (chars inesperados, área/char suspeita)
  3. Roda YOLO no crop (ou pula com --no-model)
  4. Calcula Recall de Kanjis por linha e agrega por volume/corpus
  5. Salva JSON detalhado e imprime resumo

Uso rápido (2 volumes, sem modelo — apenas parse e filtros):
    python -m src.preprocessing.manga109_corpus \\
        --manga109 data/manga109 --no-model --limit-volumes 2

Uso completo:
    python -m src.preprocessing.manga109_corpus \\
        --manga109 data/manga109 \\
        --model runs/detect/n2_n5_model/weights/best.pt \\
        --conf 0.25 --limit-volumes 10
"""
import os
import sys
import json
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# predict_kanjis vive em validate_complex para evitar duplicação
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from validate_complex import predict_kanjis  # noqa: E402

# ---------------------------------------------------------------------------
# Import lazy de YOLO — só carrega se --model for passado
# ---------------------------------------------------------------------------
_YOLO_CLASS = None


def _get_yolo():
    global _YOLO_CLASS
    if _YOLO_CLASS is None:
        from ultralytics import YOLO
        _YOLO_CLASS = YOLO
    return _YOLO_CLASS


# ---------------------------------------------------------------------------
# Constantes para filtros heurísticos
# ---------------------------------------------------------------------------
_VALID_RANGES = [
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x3000, 0x303F),   # CJK Symbols & Punctuation
    (0xFF00, 0xFFEF),   # Halfwidth & Fullwidth Forms
    (0x0020, 0x007E),   # ASCII básico
]
_MIN_AREA_PER_CHAR = 100       # px² mínimo por caractere
_MAX_AREA_PER_CHAR = 30_000    # px² máximo por caractere

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filtros heurísticos
# ---------------------------------------------------------------------------
def find_unexpected_chars(text):
    """Retorna lista de chars fora dos ranges Unicode esperados para texto de mangá."""
    unexpected = []
    for ch in text:
        cp = ord(ch)
        if not any(lo <= cp <= hi for lo, hi in _VALID_RANGES):
            unexpected.append(ch)
    return unexpected


def check_bbox_area(bbox, text_len):
    """Retorna string descrevendo suspeita de área, ou None se plausível."""
    if text_len == 0:
        return None
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    per_char = area / text_len
    if per_char < _MIN_AREA_PER_CHAR:
        return f"área/char={per_char:.0f}px² < mín {_MIN_AREA_PER_CHAR}"
    if per_char > _MAX_AREA_PER_CHAR:
        return f"área/char={per_char:.0f}px² > máx {_MAX_AREA_PER_CHAR}"
    return None


def _is_cjk(ch):
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)


# ---------------------------------------------------------------------------
# Carrega conjunto N2-N5 do .names
# ---------------------------------------------------------------------------
def load_n2n5_set(names_path):
    kanjis = set()
    with open(names_path, encoding="utf-8") as f:
        for line in f:
            ch = line.strip()
            if ch and ch != "UNKNOWN_N1":
                kanjis.add(ch)
    return kanjis


# ---------------------------------------------------------------------------
# Processamento de um único volume
# ---------------------------------------------------------------------------
def _process_volume(xml_path, images_dir, n2n5_set, model, conf):
    """Parse do XML de um volume; extrai crops e calcula recall por linha."""
    volume_name = xml_path.stem
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pages_result = []
    vol_gt = vol_hit = vol_lines = vol_suspicious = 0

    for page_elem in root.iter("page"):
        page_index = int(page_elem.get("index", 0))
        page_w_attr = int(page_elem.get("width",  0))
        page_h_attr = int(page_elem.get("height", 0))

        # Tenta extensões comuns
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = images_dir / volume_name / f"{page_index:03d}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            logger.warning("Imagem não encontrada: %s/images/%s/%03d.*",
                           images_dir.parent, volume_name, page_index)
            continue

        try:
            page_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Falha ao abrir %s: %s", img_path, exc)
            continue

        page_w, page_h = page_img.size
        lines_result = []

        for text_elem in page_elem.findall(".//text"):
            # transcript pode estar como atributo ou conteúdo de texto
            transcript = (text_elem.get("text", "") or text_elem.text or "").strip()
            if not transcript:
                continue

            x1 = int(text_elem.get("xmin", 0))
            y1 = int(text_elem.get("ymin", 0))
            x2 = int(text_elem.get("xmax", page_w_attr or page_w))
            y2 = int(text_elem.get("ymax", page_h_attr or page_h))

            # Clamp às dimensões reais da imagem
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(page_w, x2), min(page_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Ground truth N2-N5 (kanjis esperados da linha)
            gt_kanjis = {ch for ch in transcript if ch in n2n5_set}

            # Kanjis N1 presentes (para registro no JSON)
            n1_kanjis = {ch for ch in transcript if _is_cjk(ch) and ch not in n2n5_set}

            # Filtros heurísticos — loga suspeitas mas não descarta
            audit = {}
            unexpected = find_unexpected_chars(transcript)
            if unexpected:
                audit["unexpected_chars"] = unexpected
                logger.info("Chars inesperados | %s p%d: %s — '%s'",
                            volume_name, page_index, unexpected, transcript)

            area_problem = check_bbox_area((x1, y1, x2, y2), len(transcript))
            if area_problem:
                audit["bbox_area"] = area_problem
                logger.info("Área suspeita | %s p%d: %s — '%s'",
                            volume_name, page_index, area_problem, transcript)

            is_suspicious = bool(audit)
            if is_suspicious:
                vol_suspicious += 1

            if not gt_kanjis:
                # Linha sem kanjis N2-N5 não contribui para recall
                continue

            vol_lines += 1
            vol_gt += len(gt_kanjis)

            recall = 0.0
            pred_kanjis = set()

            if model is not None:
                crop = page_img.crop((x1, y1, x2, y2))
                pred_kanjis = predict_kanjis(model, crop, conf)
                hits = gt_kanjis & pred_kanjis
                vol_hit += len(hits)
                recall = len(hits) / len(gt_kanjis)

            lines_result.append({
                "page":       page_index,
                "bbox":       [x1, y1, x2, y2],
                "transcript": transcript,
                "gt_n2n5":    sorted(gt_kanjis),
                "gt_n1":      sorted(n1_kanjis),
                "pred":       sorted(pred_kanjis),
                "recall":     recall,
                "suspicious": is_suspicious,
                "audit":      audit,
            })

        if lines_result:
            pages_result.append({"page": page_index, "lines": lines_result})

    vol_recall = vol_hit / vol_gt if vol_gt > 0 else 0.0
    return {
        "volume":           volume_name,
        "total_lines":      vol_lines,
        "total_gt":         vol_gt,
        "total_hit":        vol_hit,
        "total_suspicious": vol_suspicious,
        "recall":           vol_recall,
        "pages":            pages_result,
    }


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------
def run_corpus_validation(
    manga109_root,
    names_path,
    model_path,
    conf,
    limit_volumes,
    volumes_filter,
    output_path,
):
    manga109_root = Path(manga109_root)
    xml_dir    = manga109_root / "annotations"
    images_dir = manga109_root / "images"

    if not xml_dir.exists():
        raise FileNotFoundError(f"Pasta de anotações não encontrada: {xml_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {images_dir}")

    n2n5_set = load_n2n5_set(names_path)
    logger.info("N2-N5 set carregado: %d kanjis", len(n2n5_set))

    model = None
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        YOLO = _get_yolo()
        model = YOLO(model_path)
        logger.info("Modelo carregado: %s", model_path)
    else:
        logger.info("Modo dry-run (--no-model): recall será 0 em todas as linhas")

    xml_files = sorted(xml_dir.glob("*.xml"))
    if volumes_filter:
        xml_files = [x for x in xml_files if x.stem in volumes_filter]
    if limit_volumes:
        xml_files = xml_files[:limit_volumes]

    if not xml_files:
        raise ValueError("Nenhum volume encontrado com os filtros fornecidos.")

    print(f"\nProcessando {len(xml_files)} volume(s)...")

    corpus_gt = corpus_hit = corpus_lines = corpus_suspicious = 0
    volumes_results = []

    for xml_path in tqdm(xml_files, desc="Volumes"):
        result = _process_volume(xml_path, images_dir, n2n5_set, model, conf)
        volumes_results.append(result)
        corpus_gt         += result["total_gt"]
        corpus_hit        += result["total_hit"]
        corpus_lines      += result["total_lines"]
        corpus_suspicious += result["total_suspicious"]

        tqdm.write(
            f"  {result['volume']:30s}  "
            f"linhas={result['total_lines']:4d}  "
            f"recall={result['recall']:.4f}  "
            f"susp={result['total_suspicious']}"
        )

    corpus_recall = corpus_hit / corpus_gt if corpus_gt > 0 else 0.0

    summary = {
        "volumes_avaliados":   len(volumes_results),
        "linhas_com_n2n5":     corpus_lines,
        "kanjis_gt_totais":    corpus_gt,
        "kanjis_detectados_ok": corpus_hit,
        "recall_corpus":       corpus_recall,
        "linhas_suspeitas":    corpus_suspicious,
        "conf":                conf,
        "model":               model_path or "dry-run",
    }

    print(f"\n{'='*60}")
    print(f"  VALIDAÇÃO DO CORPUS MANGA109 — N2-N5")
    print(f"{'='*60}")
    print(f"  Volumes avaliados:       {summary['volumes_avaliados']}")
    print(f"  Linhas com kanjis N2-N5: {summary['linhas_com_n2n5']}")
    print(f"  Kanjis GT totais:        {summary['kanjis_gt_totais']}")
    print(f"  Kanjis detectados OK:    {summary['kanjis_detectados_ok']}")
    print(f"  Recall do corpus:        {corpus_recall:.4f}  ({corpus_recall*100:.2f}%)")
    print(f"  Linhas suspeitas:        {summary['linhas_suspeitas']}")
    print(f"{'='*60}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "volumes": volumes_results},
                  f, ensure_ascii=False, indent=2)
    print(f"  Log detalhado: {output_path}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Validação em escala do corpus Manga109 para o modelo N2-N5."
    )
    parser.add_argument(
        "--manga109",
        default=os.environ.get("KD_MANGA109", "data/manga109"),
        help="Raiz do Manga109 (deve conter annotations/ e images/)",
    )
    parser.add_argument(
        "--names",
        default=os.environ.get("KD_N2N5_NAMES", "data/processed/n2_n5.names"),
        help="Arquivo .names com os kanjis N2-N5",
    )
    parser.add_argument("--model",  default=None, help="Caminho para best.pt")
    parser.add_argument("--no-model", action="store_true",
                        help="Dry-run: faz parse e filtros sem rodar YOLO")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit-volumes", type=int, default=None, metavar="N",
                        help="Limita a N volumes (útil para teste rápido)")
    parser.add_argument("--volumes", default=None,
                        help="Subset de volumes separados por vírgula")
    parser.add_argument("--output", default="validation_corpus_results.json")
    args = parser.parse_args()

    model_path     = None if args.no_model else args.model
    volumes_filter = [v.strip() for v in args.volumes.split(",")] if args.volumes else None

    run_corpus_validation(
        manga109_root  = args.manga109,
        names_path     = args.names,
        model_path     = model_path,
        conf           = args.conf,
        limit_volumes  = args.limit_volumes,
        volumes_filter = volumes_filter,
        output_path    = args.output,
    )

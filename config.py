#!/usr/bin/env python3
"""
Caminhos e configurações centrais do projeto.
Variáveis de ambiente com prefixo KD_ permitem sobrescrever qualquer valor
sem editar o código — útil para desenvolvimento, testes e containers.

Exemplo:
    KD_MANGA109=/mnt/ssd/manga109 python -m src.preprocessing.manga109_corpus ...
"""
import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.environ.get("KD_DATA_DIR",  os.path.join(BASE_DIR, "data"))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")

N2N5_NAMES_PATH = os.environ.get(
    "KD_N2N5_NAMES", os.path.join(PROCESSED_DIR, "n2_n5.names")
)
KANJI_DB_PATH = os.environ.get(
    "KD_KANJI_DB", os.path.join(PROCESSED_DIR, "kanji_dict.db")
)
MANGA109_ROOT = os.environ.get(
    "KD_MANGA109", os.path.join(DATA_DIR, "manga109")
)
N2N5_MODEL_PATH = os.environ.get(
    "KD_N2N5_MODEL",
    os.path.join(BASE_DIR, "runs", "detect", "n2_n5_model", "weights", "best.pt"),
)
N1_MODEL_PATH = os.environ.get(
    "KD_N1_MODEL",
    os.path.join(BASE_DIR, "runs", "detect", "n1_model", "weights", "best.pt"),
)
DEFAULT_CONF = float(os.environ.get("KD_CONF", "0.25"))

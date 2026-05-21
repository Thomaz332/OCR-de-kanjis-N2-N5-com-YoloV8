<div align="center">

# OCR de Kanjis N2–N5 com YOLOv8

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PC9zdmc+)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)](#)

**Detecção e Reconhecimento em Tempo Real de Kanjis N2–N5 em Mangás Japoneses**

*Projeto de Iniciação Científica — Pipeline completo: geração sintética balanceada → treinamento YOLOv8 → API semântica (Kanjidic2) → cliente de captura com overlay*

</div>

---

## Contexto do Projeto

Este repositório faz parte de uma pesquisa de IC em **Visão Computacional aplicada ao Processamento de Documentos Japoneses**, desenvolvida em duas frentes paralelas e complementares:

| Módulo | Escopo | Estratégia | Classes |
|--------|--------|------------|---------|
| **Este repositório** | Kanjis JLPT N2–N5 | *Frequency-Driven Normalization* (Distribuição Head) | ~690 |
| Módulo parceiro (N1) | Kanjis JLPT N1 | *Synthetic Oversampling* (Distribuição Tail) | ~1200 |

O sistema completo opera em **cascata**: o modelo N2–N5 (rápido, generalista) é executado primeiro. Quando encontra um kanji fora de sua distribuição, marca como `UNKNOWN_N1` e delega ao modelo N1 especialista. Juntos cobrem os ~1900 kanjis avaliados no JLPT.

---

## Motivação Científica: Lei de Zipf em Mangás

A distribuição de frequência de kanjis em textos de mangá segue uma **Lei de Zipf**: poucos kanjis aparecem com altíssima frequência enquanto a maioria é rara.

```
Frequência de Aparição em Mangás

  Alto │ ████ N5 (~100 kanjis) 
       │ ███  N4 (~200 kanjis)
       │ ██   N3 (~370 kanjis)
       │ █    N2 (~360 kanjis)
  Baixo│ ░░░░░░░░░░░░░░░ N1 (~1200 kanjis) — "Cauda Longa"
       └─────────────────────────────────────► Nível JLPT
```

Esta assimetria impõe estratégias diferentes por nível:
- **N2–N5 (este projeto):** Os kanjis mais frequentes do mangá. Treinar com dados brutos criaria viés massivo para super-comuns (日, 月, 一). A solução é **normalização artificial**: cada classe recebe exatamente o mesmo número de amostras sintéticas.
- **N1 (projeto parceiro):** Kanjis raros com ~25 exemplos reais por classe no Manga109. Exige *oversampling* sintético massivo.

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PIPELINE COMPLETO N2–N5                         │
├──────────────────┬──────────────────┬──────────────────────────────┤
│  1. DADOS        │  2. TREINAMENTO  │  3. APLICAÇÃO                │
│                  │                  │                              │
│  download_       │  generate_       │  api_server.py               │
│  n2_n5_kanjis.py │  synthetic_      │  (Flask + SQLite)            │
│       ↓          │  images.py       │       ↑                      │
│  n2_n5.names     │       ↓          │  semantic_service.py         │
│  (~690 classes)  │  YOLOv8n         │  (Kanjidic2 local +          │
│       ↓          │  50 épocas       │   kanjiapi.dev fallback)     │
│  download_       │  batch=16        │       ↑                      │
│  fonts.py        │  imgsz=640       │  capture_tool.py             │
│  (NotoSans/      │       ↓          │  (mss + pynput + pystray)    │
│   NotoSerif CJK) │  best.pt         │  F12 → Overlay → On-Hover   │
└──────────────────┴──────────────────┴──────────────────────────────┘
```

### Lógica de Cascata (Head → Tail)

```
Captura de tela (mss)
        │
        ▼
┌──────────────┐   kanji N2-N5?   ┌─────────────────┐
│ Modelo N2-N5 │────── Sim ───────►│ SQLite Kanjidic2│
│ (Generalista)│                   │ → significado   │
└──────┬───────┘                   └─────────────────┘
       │ UNKNOWN_N1
       ▼
┌──────────────┐   kanji N1?      ┌─────────────────┐
│ Modelo N1    │────── Sim ───────►│ SQLite Kanjidic2│
│ (Especialista│                   │ → significado   │
└──────┬───────┘                   └─────────────────┘
       │ Ambos falham
       ▼
   "desconhecido"
```

---

## Estrutura do Repositório

```
ocr-de-kanjis-n2-n5-com-yolov8/
├── README.md
├── requirements.txt
├── assets/
│   └── fonts/                          # Fontes CJK (download automático)
├── data/
│   ├── processed/
│   │   └── n2_n5.names                 # 689 kanjis N2-N5 + UNKNOWN_N1
│   └── raw/                            # Dados brutos (kanjidic2, JSON)
├── docs/
│   └── relatorio_n2_n5.md              # Relatório científico completo
├── notebooks/
│   └── Kaggle_Training_Pipeline.ipynb  # Pipeline Kaggle/Colab (GPU)
└── src/
    ├── data/
    │   ├── download_fonts.py            # Baixa fontes CJK
    │   ├── download_n2_n5_kanjis.py    # Baixa e filtra lista N2-N5
    │   └── generate_synthetic_images.py # Gera dataset balanceado
    ├── preprocessing/
    │   ├── parse_manga109.py            # Parser XML do Manga109
    │   ├── filter_and_split.py          # Filtra por nível JLPT
    │   ├── create_yolo_config.py        # Gera data.yaml
    │   └── validate_complex.py          # Validação em frases reais
    ├── semantic/
    │   ├── build_dictionary.py          # Constrói SQLite do Kanjidic2
    │   ├── semantic_service.py          # Lookup local + fallback online
    │   └── api_server.py               # API Flask REST
    └── capture/
        └── capture_tool.py              # Cliente overlay (F12)
```

---

## Instalação

### Pré-requisitos

- Python 3.8+  
- GPU NVIDIA com CUDA recomendada (para treinamento; inferência funciona em CPU)

```bash
git clone https://github.com/Thomaz332/OCR-de-kanjis-N2-N5-com-YoloV8.git
cd OCR-de-kanjis-N2-N5-com-YoloV8
pip install -r requirements.txt
```

---

## Pipeline Passo a Passo

### Passo 1 — Baixar Recursos

```bash
# Fontes CJK para renderização sintética
python src/data/download_fonts.py

# Lista de kanjis N2-N5 (regenera n2_n5.names se necessário)
python src/data/download_n2_n5_kanjis.py
```

### Passo 2 — Gerar Dataset Sintético

```bash
python src/data/generate_synthetic_images.py
```

Isso gera **~69.000 imagens balanceadas** (100 por classe × ~690 classes), com split automático 80/20 treino/validação e o arquivo `data/synthetic/data.yaml`.

**Augmentations aplicadas:**

| Técnica | Prob. | Justificativa |
|---|---|---|
| Distorção Elástica | 50% | Simula fontes manuscritas e speech bubbles irregulares |
| Variação de Tinta (Dilate/Erode) | 100% | Simula espessura variável de impressão |
| Screentone | 30% | Padrão de impressão típico de mangá japonês |
| Textura de Papel | 50% | Simula scan de volume antigo com manchas |
| Rotação ±10° | 70% | Balões de fala inclinados |
| Ruído Gaussiano | 60% | Qualidade variável de digitalização |
| Blur Gaussiano | 50% | Desfoque por baixa resolução |
| Random Crop | 40% | Kanji parcialmente visível na borda do balão |

### Passo 3 — Treinar o Modelo

```bash
yolo detect train \
    project=yolo_kanji \
    name=n2_n5_model \
    data=data/synthetic/data.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    device=0
```

> Para treino na nuvem (GPU gratuita), use o notebook `notebooks/Kaggle_Training_Pipeline.ipynb` no Kaggle ou Google Colab.

### Passo 4 — Validação (Opcional, requer Manga109)

```bash
# Parsear anotações do Manga109
python src/preprocessing/parse_manga109.py \
    --root /caminho/para/manga109 \
    --output data/processed/manga109

# Filtrar por nível JLPT
python src/preprocessing/filter_and_split.py \
    --metadata data/processed/manga109/metadata.json \
    --jlpt data/raw/jlpt_kanjis.json \
    --output data/processed

# Validação em cenas complexas (Recall de Kanjis)
python src/preprocessing/validate_complex.py \
    --model runs/detect/n2_n5_model/weights/best.pt \
    --metadata data/processed/n2_n5/val_complex_metadata.json \
    --images data/processed/n2_n5/val_complex
```

---

## API Semântica

Consulta de significados para qualquer kanji detectado, com banco local (Kanjidic2) e fallback online (kanjiapi.dev).

```bash
# 1. Construir o banco de dados local (única vez)
python src/semantic/build_dictionary.py

# 2. Iniciar o servidor
python src/semantic/api_server.py
```

**Exemplos de uso:**

```bash
# Consulta simples
curl http://localhost:5000/kanji/明

# Consulta em lote
curl -X POST http://localhost:5000/batch \
     -H 'Content-Type: application/json' \
     -d '{"kanjis": ["明","日","学","語"]}'

# Health check
curl http://localhost:5000/health
```

**Resposta de exemplo:**
```json
{
  "kanji": "明",
  "found": true,
  "source": "local_kanjidic2",
  "jisho_url": "https://jisho.org/search/明%23kanji",
  "data": {
    "jlpt": 3,
    "grade": 2,
    "stroke_count": 8,
    "meanings": ["bright", "light", "clear"],
    "readings_on": ["メイ", "ミョウ"],
    "readings_kun": ["あか.るい", "あ.ける", "あ.かり"]
  }
}
```

---

## Cliente de Captura com Overlay

Ferramenta desktop para leitura de mangás com OCR em tempo real.

```bash
# Terminal 1: API Semântica
python src/semantic/api_server.py

# Terminal 2: Cliente de Captura
python src/capture/capture_tool.py
```

**Controles:**
- **F12** — Liga/desliga o OCR e o overlay
- Ícone na bandeja do sistema para controle
- No terminal: mostra os kanjis detectados e seus significados em tempo real

**Otimizações implementadas:**
- **Detecção de mudança de tela (>30%):** O modelo só roda quando a página muda, reduzindo uso de GPU em ~90% durante leitura estática
- **Lógica de cascata N2-N5 → N1:** Minimiza uso do modelo mais pesado (N1)

---

## Métricas de Avaliação

| Métrica | Dataset | Descrição |
|---|---|---|
| **mAP@50** | Sintético (val) | Precisão média com IoU ≥ 0.5. Métrica padrão YOLO para localização + classificação |
| **Recall de Kanjis** | Manga109 (frases) | % dos kanjis da frase corretamente detectados. Usada quando coordenadas GT individuais não existem |

> **Por que duas métricas?** No dataset sintético temos Ground Truth geométrico (BBoxes exatas), permitindo mAP. No Manga109, o Ground Truth é apenas o texto transcrito da frase — sem coordenadas individuais por caractere, o mAP é matematicamente inaplicável. O Recall de Kanjis mede a utilidade real para o usuário final.

**Configurações de treinamento recomendadas:**

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Modelo base | `yolov8n.pt` | Nano: ideal para inferência em tempo real com GPU doméstica |
| Épocas | 50 | Suficiente para convergência — N2-N5 converge mais rápido que N1 por ter classes mais distintas |
| Batch | 16 | Estável em GPU com 8 GB VRAM |
| imgsz | 640 | Padrão YOLO, melhor tradeoff custo/precisão |
| Confiança (inferência) | 0.25 | Equilibra recall e precisão em cenas reais |

---

## Contexto Acadêmico

Este projeto integra uma **Iniciação Científica** desenvolvida no âmbito de um curso de Ciência da Computação (8º semestre), na área de Visão Computacional aplicada ao processamento de caracteres japoneses.

| Papel | Contribuição |
|---|---|
| **Thomaz** | Modelo N2–N5 (este repositório) — distribuição Head, normalização de frequência |
| **Miguel** | Modelo N1 (repositório parceiro) — distribuição Tail, oversampling sintético |

O relatório científico completo está disponível em [`docs/relatorio_n2_n5.md`](docs/relatorio_n2_n5.md).

---

## Licença e Créditos

Distribuído sob a **Licença MIT**.

- **[Kanjidic2](https://www.edrdg.org/wiki/index.php/KANJIDIC_Project)** — EDRDG, licença CC BY-SA 4.0
- **[kanji-data](https://github.com/davidluzgouveia/kanji-data)** — davidluzgouveia (listas JLPT)
- **[Noto CJK Fonts](https://github.com/notofonts/noto-cjk)** — Google, licença SIL OFL 1.1
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — Ultralytics, licença AGPL-3.0

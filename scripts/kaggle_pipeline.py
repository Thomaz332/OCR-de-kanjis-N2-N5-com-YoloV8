"""Automação do fluxo de treino no Kaggle via linha de comando.

Substitui o fluxo manual (abrir o site, "+ Add Input", editar variáveis no
notebook, "Save & Run All") por um único comando:

    python scripts/kaggle_pipeline.py --run

O que o pipeline faz, em ordem:
  1. Lê kaggle_config.json (raiz do repo) para saber qual dataset sintético e
     qual checkpoint devem ser reaproveitados nesta rodada.
  2. Atualiza notebooks/kernel-metadata.json (campo dataset_sources) para que
     esses datasets sejam anexados como Input automaticamente -- equivalente
     a clicar em "+ Add Input" na interface web.
  3. Sobe uma nova versão do notebook (kaggle kernels push).
  4. Faz polling do status da execução até ela terminar, dar erro ou ser
     cancelada (limite de 12h por commit / cota de GPU semanal).
  5. Baixa a saída da execução (kaggle kernels output).
  6. Cria/versiona um Kaggle Dataset com o checkpoint mais recente (e, na
     primeira vez em que o dataset sintético é gerado do zero, também cria um
     dataset com ele), para que a próxima rodada os reaproveite via Input sem
     precisar mexer na interface web.

Uso:
    python scripts/kaggle_pipeline.py --run
    python scripts/kaggle_pipeline.py --push
    python scripts/kaggle_pipeline.py --status
    python scripts/kaggle_pipeline.py --download
    python scripts/kaggle_pipeline.py --dataset

Pré-requisitos:
    - `pip install kaggle` (feito) e credenciais em
      %USERPROFILE%\\.kaggle\\kaggle.json (Windows).
    - notebooks/Kaggle_Training_Pipeline.ipynb e notebooks/kernel-metadata.json
      devem existir e estar sincronizados com o que você quer rodar no Kaggle.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
NOTEBOOK_FILE = NOTEBOOK_DIR / "Kaggle_Training_Pipeline.ipynb"
KERNEL_METADATA_FILE = NOTEBOOK_DIR / "kernel-metadata.json"
CONFIG_PATH = REPO_ROOT / "kaggle_config.json"
# Fora do repositório de propósito: o repo vive num drive sincronizado
# (Google Drive), e escrever dezenas de milhares de arquivos pequenos lá é
# ordens de magnitude mais lento do que no disco local/temp do sistema.
LOCAL_RUNS_DIR = Path(tempfile.gettempdir()) / "ic_ocr_kaggle_pipeline_runs"

DEFAULT_POLL_INTERVAL = 300          # 5 minutos
DEFAULT_MAX_WAIT = 45000             # 12h30 (o limite do Kaggle é 12h/commit)

TERMINAL_ERROR_STATUSES = ("ERROR",)
TERMINAL_CANCEL_STATUSES = ("CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED")
TERMINAL_SUCCESS_STATUSES = ("COMPLETE",)

CONFIG_DEFAULTS = {
    "kernel_slug": "joaothomaz/ic-ocr",
    "synthetic_dataset_slug": None,
    "synthetic_dataset_path": None,
    "checkpoint_dataset_slug": None,
    "checkpoint_path": None,
}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --------------------------------------------------------------------------
# kaggle CLI helpers
# --------------------------------------------------------------------------

def kaggle_cmd(*args: str) -> list[str]:
    # Usa `python -m kaggle` em vez do executável `kaggle` porque o script
    # instalado pelo pip nem sempre fica no PATH (é o caso no Windows por
    # padrão, a menos que a pasta Scripts do Python seja adicionada ao PATH).
    return [sys.executable, "-m", "kaggle", *args]


def run_kaggle_streamed(*args: str) -> int:
    """Roda um comando kaggle mostrando a saída em tempo real (push/output podem demorar)."""
    log(f"$ kaggle {' '.join(args)}")
    proc = subprocess.run(kaggle_cmd(*args))
    return proc.returncode


def run_kaggle_captured(*args: str) -> tuple[int, str]:
    """Roda um comando kaggle capturando stdout+stderr como texto (para parsing)."""
    proc = subprocess.run(kaggle_cmd(*args), capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


# --------------------------------------------------------------------------
# config / credenciais
# --------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**CONFIG_DEFAULTS, **data}
    return dict(CONFIG_DEFAULTS)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    log(f"kaggle_config.json atualizado: {CONFIG_PATH}")


# --------------------------------------------------------------------------
# 1. push
# --------------------------------------------------------------------------

PIPELINE_CONFIG_CELL_MARKER = "# === PIPELINE_CONFIG ==="


def _nb_source_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def sync_pipeline_config_cell(config: dict) -> None:
    """Escreve synthetic_dataset_path/checkpoint_path diretamente na célula
    PIPELINE_CONFIG do notebook local, como um dict literal.

    O notebook não lê kaggle_config.json em tempo de execução no Kaggle (o
    kernel só tem acesso ao que foi clonado do GitHub ou anexado como Input);
    por isso os valores precisam ser gravados na própria célula do .ipynb
    antes do push, e não apenas no kaggle_config.json local.
    """
    if not NOTEBOOK_FILE.exists():
        raise SystemExit(f"{NOTEBOOK_FILE} não encontrado.")
    with open(NOTEBOOK_FILE, "r", encoding="utf-8") as f:
        nb = json.load(f)

    target_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and "".join(cell["source"]).lstrip().startswith(PIPELINE_CONFIG_CELL_MARKER):
            target_idx = i
            break
    if target_idx is None:
        raise SystemExit(
            f"Célula com marcador {PIPELINE_CONFIG_CELL_MARKER!r} não encontrada em {NOTEBOOK_FILE}. "
            "O notebook foi alterado manualmente? Restaure a célula PIPELINE_CONFIG."
        )

    def _py_literal(value: str | None) -> str:
        return "None" if value is None else repr(value)

    source_text = (
        f"{PIPELINE_CONFIG_CELL_MARKER}\n"
        "# Este dicionário é reescrito diretamente nesta célula por\n"
        "# scripts/kaggle_pipeline.py (a partir de kaggle_config.json) logo antes\n"
        "# de cada `kaggle kernels push` -- não há leitura de arquivo em tempo de\n"
        "# execução, então não há dependência de nada além do próprio notebook.\n"
        "# Rodando manualmente (Save & Run All) sem passar pelo pipeline, os\n"
        "# valores abaixo (None) fazem o notebook gerar o dataset e treinar do\n"
        "# zero, como no comportamento original.\n"
        "PIPELINE_CONFIG = {\n"
        f"    'synthetic_dataset_path': {_py_literal(config.get('synthetic_dataset_path'))},\n"
        f"    'checkpoint_path': {_py_literal(config.get('checkpoint_path'))},\n"
        "}\n"
        "\n"
        "print('PIPELINE_CONFIG:', PIPELINE_CONFIG)"
    )
    nb["cells"][target_idx]["source"] = _nb_source_lines(source_text)
    nb["cells"][target_idx]["outputs"] = []
    nb["cells"][target_idx]["execution_count"] = None

    with open(NOTEBOOK_FILE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    log(
        "PIPELINE_CONFIG gravado no notebook: "
        f"synthetic_dataset_path={config.get('synthetic_dataset_path')!r}, "
        f"checkpoint_path={config.get('checkpoint_path')!r}"
    )


def sync_kernel_metadata(config: dict) -> None:
    if not KERNEL_METADATA_FILE.exists():
        raise SystemExit(f"{KERNEL_METADATA_FILE} não encontrado.")
    with open(KERNEL_METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dataset_sources = [
        slug for slug in (config.get("synthetic_dataset_slug"), config.get("checkpoint_dataset_slug"))
        if slug
    ]
    meta["dataset_sources"] = dataset_sources
    meta["id"] = config["kernel_slug"]

    with open(KERNEL_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if dataset_sources:
        log(f"kernel-metadata.json -> dataset_sources = {dataset_sources}")
    else:
        log("kernel-metadata.json -> nenhum dataset anexado ainda (primeira rodada gera tudo do zero).")


_PUSH_FAILURE_MARKERS = (
    "not valid dataset sources",
    "not valid competition sources",
    "not valid kernel sources",
    "not valid model sources",
    "kernel push error",
)


def push_kernel(config: dict) -> None:
    if not NOTEBOOK_FILE.exists():
        raise SystemExit(f"{NOTEBOOK_FILE} não encontrado. Mantenha o .ipynb local sincronizado antes do push.")
    sync_pipeline_config_cell(config)
    sync_kernel_metadata(config)
    # Usa a versão "captured" (em vez de streamed) porque a CLI as vezes reporta
    # falhas lógicas (dataset invalido, limite de sessoes, etc.) via stdout com
    # exit code 0 -- precisamos inspecionar o texto, não só o código de saída.
    code, output = run_kaggle_captured("kernels", "push", "-p", str(NOTEBOOK_DIR))
    print(output)
    lower_output = output.lower()
    if code != 0:
        raise SystemExit(f"kaggle kernels push falhou (exit code {code}).")
    if any(marker in lower_output for marker in _PUSH_FAILURE_MARKERS):
        raise SystemExit(
            "kaggle kernels push reportou uma falha lógica (dataset inválido, limite de sessões, etc.) "
            "apesar do exit code 0 -- ver saída acima. Corrija e rode de novo antes de prosseguir."
        )
    log(f"Push concluído: {config['kernel_slug']}")


# --------------------------------------------------------------------------
# 2. status (polling)
# --------------------------------------------------------------------------

_STATUS_RE = re.compile(r'has status\s+"([^"]+)"', re.IGNORECASE)
_KNOWN_STATUSES = (
    "CANCEL_ACKNOWLEDGED",
    "CANCEL_REQUESTED",
    "COMPLETE",
    "ERROR",
    "RUNNING",
    "QUEUED",
    "NEW_SCRIPT",
)


def parse_status(raw_output: str) -> str:
    # Só confia no texto que vem explicitamente depois de `has status "..."`
    # (formato do `kaggle kernels status`). Isso evita falso-positivo: uma
    # mensagem de erro de rede/autenticação que contenha a palavra "error"
    # em outro lugar da saída não deve ser confundida com o kernel ter
    # terminado com status ERROR.
    match = _STATUS_RE.search(raw_output)
    if not match:
        return "UNKNOWN"
    captured = match.group(1).upper()
    # captured pode vir como "KernelWorkerStatus.RUNNING" ou só "RUNNING"/"running".
    for token in _KNOWN_STATUSES:
        if token in captured:
            return token
    return "UNKNOWN"


def poll_status(kernel_slug: str, poll_interval: int, max_wait: int) -> str:
    start = time.monotonic()
    log(f"Aguardando execução de {kernel_slug} (poll a cada {poll_interval}s, limite de {max_wait}s)...")
    while True:
        code, output = run_kaggle_captured("kernels", "status", kernel_slug)
        status = parse_status(output)
        elapsed = int(time.monotonic() - start)
        log(f"status={status} (raw: {output.strip()!r}) | decorrido={elapsed}s")

        if status in TERMINAL_SUCCESS_STATUSES:
            log("Execução concluída com sucesso.")
            return status
        if status in TERMINAL_ERROR_STATUSES:
            log("Execução terminou com erro.")
            return status
        if status in TERMINAL_CANCEL_STATUSES:
            log("Execução foi cancelada (provavelmente limite de 12h ou cota de GPU semanal).")
            return status
        if status == "UNKNOWN" and code != 0:
            log("Não foi possível obter status (kernel ainda propagando ou erro de rede); tentando de novo...")

        if elapsed >= max_wait:
            log("Tempo máximo de espera local atingido; parando o polling (a execução pode continuar no Kaggle).")
            return status

        time.sleep(poll_interval)


# --------------------------------------------------------------------------
# 3. download
# --------------------------------------------------------------------------

def _resumable_get(url: str, dest: Path, max_retries: int = 8) -> bool:
    """Baixa uma URL com suporte a retomada via Range -- uma queda de conexão
    continua de onde parou em vez de reiniciar do zero (`kaggle kernels output`
    usa .content de uma vez só e não recupera nada de uma IncompleteRead)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    return True  # já completo
                mode = "ab" if (existing and r.status_code == 206) else "wb"
                with open(dest, mode) as out:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
                return True
        except Exception as e:
            log(f"  retry {attempt}/{max_retries} do download apos erro: {e}")
            time.sleep(min(2 ** attempt, 30))
    return False


def download_output(kernel_slug: str) -> Path:
    """Baixa a saída do kernel para uma pasta LOCAL (fora do repositório).

    Importante: o repositório vive num drive sincronizado (Google Drive via
    "Outros computadores"), e escrever dezenas de milhares de arquivos
    pequenos ali é ordens de magnitude mais lento (chegamos a estimar ~20h
    para extrair um dataset que levou ~8min localmente). Por isso a saída é
    baixada/extraída em LOCAL_RUNS_DIR (temp do sistema), não em kaggle_runs/
    dentro do repo.

    Também usa a API de listagem diretamente (em vez de `kaggle kernels
    output`) porque o Kaggle empacota toda a saída como um único arquivo
    `_output_.zip`, e o download desse zip via `.content` da CLI não
    consegue retomar após uma conexão quebrada -- `_resumable_get` sim.
    """
    owner, ksug = kernel_slug.split("/", 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_root = LOCAL_RUNS_DIR / stamp
    local_root.mkdir(parents=True, exist_ok=True)
    log(f"Baixando saída em pasta local: {local_root}")

    api = KaggleApi()
    api.authenticate()

    files: list[tuple[str, str]] = []
    log_content = None
    with api.build_kaggle_client() as kaggle:
        token = None
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = ksug
            req.page_size = 500
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            for f in resp.files or []:
                files.append((f.file_name, f.url))
            if resp.log:
                log_content = resp.log
            token = resp.next_page_token
            if not token:
                break

    log(f"Saída listada: {len(files)} arquivo(s).")
    if log_content:
        with open(local_root / f"{ksug}.log", "w", encoding="utf-8") as f:
            f.write(log_content)

    all_ok = True
    for name, url in files:
        dest = local_root / name
        ok = _resumable_get(url, dest)
        if not ok:
            log(f"AVISO: falha definitiva ao baixar {name} apos varias tentativas.")
            all_ok = False
            continue
        if name.endswith(".zip"):
            log(f"Extraindo {name}...")
            with zipfile.ZipFile(dest) as z:
                z.extractall(local_root)

    if not all_ok:
        log("AVISO: um ou mais arquivos de saída falharam; a saída local pode estar incompleta.")
    else:
        log(f"Output baixado e extraído em: {local_root}")
    return local_root


# --------------------------------------------------------------------------
# 4. criação/versionamento de datasets
# --------------------------------------------------------------------------

def dataset_exists(slug: str) -> bool:
    code, _ = run_kaggle_captured("datasets", "status", slug)
    return code == 0


def write_dataset_metadata(folder: Path, slug: str, title: str) -> None:
    meta = {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}
    with open(folder / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def create_or_version_dataset(slug: str, folder: Path, title: str, message: str) -> None:
    write_dataset_metadata(folder, slug, title)
    if dataset_exists(slug):
        log(f"Dataset {slug} já existe -> criando nova versão...")
        code = run_kaggle_streamed("datasets", "version", "-p", str(folder), "-m", message, "-r", "zip")
    else:
        log(f"Dataset {slug} não existe -> criando pela primeira vez...")
        code = run_kaggle_streamed("datasets", "create", "-p", str(folder), "-r", "zip")
    if code != 0:
        raise SystemExit(f"Falha ao criar/versionar dataset {slug} (exit code {code}).")


def find_checkpoint_run_dir(output_dir: Path) -> Path | None:
    # O ultralytics salva em <cwd>/runs/detect/<project>/<name> por padrão
    # (confirmado na prática: runs/detect/yolo_kanji/n2_n5_model), não em
    # <project>/<name> direto. Busca ampla via ** cobre isso e variações.
    candidates = (
        list(output_dir.glob("**/yolo_kanji/n2_n5_model"))
        + list(output_dir.glob("**/runs/detect/train*"))
    )
    candidates = [c for c in candidates if (c / "weights" / "last.pt").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "weights" / "last.pt").stat().st_mtime)


def stage_checkpoint(run_dir: Path, staging_root: Path) -> Path:
    # Fica achatado de propósito (sem subpasta n2_n5_model): quando
    # `kaggle datasets create -r zip` zipa uma subpasta e o Kaggle a extrai
    # do lado do servidor, o nome dessa subpasta NÃO é preservado como
    # prefixo -- o conteúdo do zip vira a raiz do dataset diretamente
    # (confirmado na prática via `kaggle datasets files`). Então o que for
    # colocado aqui em staging_dir é exatamente o que aparece em
    # /kaggle/input/<dataset>/ depois.
    staging_dir = staging_root / "checkpoint_upload"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    (staging_dir / "weights").mkdir(parents=True)

    for fname in ("weights/last.pt", "weights/best.pt", "args.yaml", "results.csv"):
        src = run_dir / fname
        if src.exists():
            dst = staging_dir / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return staging_dir


def find_synthetic_dataset_dir(output_dir: Path) -> Path | None:
    # A saída baixada preserva o caminho completo do repo dentro do zip
    # (ex: OCR-de-kanjis-N2-N5-com-YoloV8/data/synthetic/...), então busca
    # recursiva em vez de um caminho fixo relativo à raiz.
    for candidate in output_dir.glob("**/data/synthetic"):
        train_dir = candidate / "images" / "train"
        if train_dir.exists() and any(train_dir.iterdir()):
            return candidate
    return None


def update_datasets(config: dict, output_dir: Path, message: str, skip_synthetic: bool) -> dict:
    # Deriva o username do próprio slug do kernel (owner/kernel-slug) em vez de
    # depender de kaggle.json existir -- essa conta usa o access_token novo,
    # que não tem um "username" legível localmente.
    username = config["kernel_slug"].split("/")[0]

    run_dir = find_checkpoint_run_dir(output_dir)
    if run_dir is not None:
        if not config.get("checkpoint_dataset_slug"):
            config["checkpoint_dataset_slug"] = f"{username}/ic-ocr-checkpoint"
        staging_root = output_dir / "_staging"
        checkpoint_folder = stage_checkpoint(run_dir, staging_root)
        create_or_version_dataset(
            config["checkpoint_dataset_slug"], checkpoint_folder, "IC OCR Kanji Checkpoint", message
        )
        dataset_name = config["checkpoint_dataset_slug"].split("/")[-1]
        config["checkpoint_path"] = f"/kaggle/input/{dataset_name}"
        log(f"checkpoint_path para a próxima rodada: {config['checkpoint_path']}")
    else:
        log("Nenhum checkpoint (weights/last.pt) encontrado na saída baixada; nada para versionar.")

    if not skip_synthetic and not config.get("synthetic_dataset_slug"):
        synth_dir = find_synthetic_dataset_dir(output_dir)
        if synth_dir is not None:
            config["synthetic_dataset_slug"] = f"{username}/ic-ocr-synthetic-dataset"
            create_or_version_dataset(
                config["synthetic_dataset_slug"], synth_dir, "IC OCR Kanji Synthetic Dataset", message
            )
            dataset_name = config["synthetic_dataset_slug"].split("/")[-1]
            config["synthetic_dataset_path"] = f"/kaggle/input/{dataset_name}"
            log(f"synthetic_dataset_path para as próximas rodadas: {config['synthetic_dataset_path']}")
        else:
            log("Nenhum dataset sintético completo encontrado na saída (provavelmente já veio de cache).")

    return config


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Roda o pipeline completo: push + status + download + dataset")
    parser.add_argument("--push", action="store_true", help="Só faz o push de uma nova versão do kernel")
    parser.add_argument("--status", action="store_true", help="Só faz o polling do status do kernel")
    parser.add_argument("--download", action="store_true", help="Só baixa a saída do kernel")
    parser.add_argument("--dataset", action="store_true", help="Só cria/versiona os datasets a partir da última saída baixada")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL, help="Segundos entre checagens de status (padrão: 300)")
    parser.add_argument("--max-wait", type=int, default=DEFAULT_MAX_WAIT, help="Tempo máximo local de espera em segundos (padrão: 45000 = 12h30)")
    parser.add_argument("--message", default=None, help="Mensagem de versão para os datasets (padrão: timestamp)")
    parser.add_argument("--skip-synthetic-dataset", action="store_true", help="Não cria/versiona o dataset sintético automaticamente")
    parser.add_argument("--output-dir", default=None, help="Pasta com saída já baixada, para usar com --dataset isolado")
    args = parser.parse_args()

    if not any([args.run, args.push, args.status, args.download, args.dataset]):
        parser.print_help()
        raise SystemExit(1)

    config = load_config()
    message = args.message or f"pipeline run {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

    output_dir: Path | None = Path(args.output_dir) if args.output_dir else None

    if args.run or args.push:
        push_kernel(config)
        save_config(config)

    final_status = None
    if args.run or args.status:
        final_status = poll_status(config["kernel_slug"], args.poll_interval, args.max_wait)

    if args.run or args.download:
        output_dir = download_output(config["kernel_slug"])

    if args.run or args.dataset:
        if output_dir is None:
            raise SystemExit("Nenhuma pasta de saída disponível. Rode com --download antes, ou passe --output-dir.")
        config = update_datasets(config, output_dir, message, args.skip_synthetic_dataset)
        save_config(config)

    if args.run:
        log(f"Pipeline finalizado. status_final={final_status}")
        if final_status in TERMINAL_ERROR_STATUSES:
            raise SystemExit(2)


if __name__ == "__main__":
    main()

import os
import sys
import time
import threading
import argparse
import numpy as np
import mss
from PIL import Image
from pynput import keyboard
from pystray import Icon, Menu, MenuItem
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic.semantic_service import SemanticService

try:
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QPainter, QColor, QPen, QFont
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

# Escala Kanjidic2 (campo jlpt, legado): 4=N5, 3=N4, 2=N3, 1=N2; 0/None=N1
_JLPT_COLOR_RGB = {
    4: (50,  205, 50),   # N5 — verde
    3: (30,  144, 255),  # N4 — azul
    2: (255, 215, 0),    # N3 — amarelo
    1: (255, 140, 0),    # N2 — laranja
    0: (220, 20,  60),   # N1 / desconhecido — vermelho
}

if _QT_AVAILABLE:
    class OverlayWindow(QWidget):
        """Janela transparente sempre-no-topo que desenha bboxes JLPT sobre a tela."""

        # (lista de detecções, scale_x, scale_y) — chamado da thread de captura
        detection_signal = pyqtSignal(object, float, float)

        def __init__(self):
            super().__init__(parent=None)
            self._detections = []
            self._sx = 1.0
            self._sy = 1.0
            self._setup()
            self.detection_signal.connect(self._on_detections)

        def _setup(self):
            flags = (
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            if sys.platform.startswith("linux"):
                flags |= Qt.WindowType.X11BypassWindowManagerHint
            self.setWindowFlags(flags)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(screen)
            self.show()

        def _on_detections(self, detections, sx, sy):
            self._detections = detections
            self._sx = sx
            self._sy = sy
            self.update()  # agenda repaint sem flicker

        def clear(self):
            self._detections = []
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)

            for d in self._detections:
                bbox = d.get("bbox")
                if not bbox:
                    continue

                # Converte coordenadas físicas (captura) → lógicas (Qt)
                lx1 = int(bbox[0] * self._sx)
                ly1 = int(bbox[1] * self._sy)
                lx2 = int(bbox[2] * self._sx)
                ly2 = int(bbox[3] * self._sy)
                w, h = lx2 - lx1, ly2 - ly1

                jlpt = d.get("jlpt_level") or 0
                rgb  = _JLPT_COLOR_RGB.get(jlpt, _JLPT_COLOR_RGB[0])
                color = QColor(*rgb)

                # Preenchimento semi-transparente
                fill = QColor(*rgb)
                fill.setAlpha(50)
                painter.fillRect(lx1, ly1, w, h, fill)

                # Borda colorida
                painter.setPen(QPen(color, 2))
                painter.drawRect(lx1, ly1, w, h)

                # Confiança acima da bbox
                conf_text = f"{d.get('conf', 0) * 100:.0f}%"
                painter.setPen(QColor(255, 255, 255))
                text_y = max(12, ly1 - 4)
                painter.drawText(lx1 + 2, text_y, conf_text)

                # Significados em português abaixo da bbox
                meanings = d.get("meanings", [])[:2]
                if meanings:
                    painter.drawText(lx1 + 2, ly2 + 14, " · ".join(meanings))

            painter.end()


class KanjiN2N5CaptureClient:
    """
    Cliente de captura de tela para leitura de mangás.

    Atalho F12: liga/desliga o OCR em tempo real.
    Lógica de cascata: N2-N5 primeiro; UNKNOWN_N1 delega ao modelo N1.
    Overlay PyQt6 transparente exibe bboxes coloridas por nível JLPT.
    Detecção de mudança de tela (>30%) para economizar GPU.
    """

    def __init__(self, conf=0.25):
        self.active = False
        self.hotkey = keyboard.Key.f12
        self.conf_threshold = conf
        self.change_threshold = 0.30

        self.model_n2_n5 = None
        self.model_n1    = None
        self.semantic    = None
        self.tray_icon   = None
        self.overlay     = None
        self.last_frame  = None
        self._screen_w   = 0
        self._screen_h   = 0

        self.model_paths = {
            "n2_n5": os.environ.get("N2N5_MODEL", "runs/detect/n2_n5_model/weights/best.pt"),
            "n1":    os.environ.get("N1_MODEL",   "runs/detect/n1_model/weights/best.pt"),
        }

    def load_resources(self):
        print("Carregando modelos e dicionário...")

        if os.path.exists(self.model_paths["n2_n5"]):
            self.model_n2_n5 = YOLO(self.model_paths["n2_n5"])
            print(f"  N2-N5 OK: {self.model_paths['n2_n5']}")
        else:
            print(f"  Aviso: modelo N2-N5 não encontrado em {self.model_paths['n2_n5']}")

        if os.path.exists(self.model_paths["n1"]):
            self.model_n1 = YOLO(self.model_paths["n1"])
            print(f"  N1 OK: {self.model_paths['n1']}")

        db_path = os.environ.get("KD_KANJI_DB", "data/processed/kanji_dict.db")
        try:
            self.semantic = SemanticService(db_path)
            print(f"  Dicionário OK: {db_path}")
        except FileNotFoundError:
            print("  Aviso: dicionário não encontrado. Execute src/semantic/build_dictionary.py")

        print("Recursos carregados. Pressione F12 para ativar.")

    def is_significant_change(self, current_frame):
        """Evita inferência desnecessária: só processa se a tela mudou >30%."""
        if self.last_frame is None:
            self.last_frame = current_frame
            return True
        prev = np.array(self.last_frame.resize((100, 100)).convert("L"), dtype=np.int16)
        curr = np.array(current_frame.resize((100, 100)).convert("L"), dtype=np.int16)
        change = np.mean(np.abs(prev - curr) > 30)
        self.last_frame = current_frame
        return change > self.change_threshold

    def cascade_inference(self, img):
        """
        Cascata Head→Tail:
        1. Modelo N2-N5 detecta kanjis comuns + bbox.
        2. Detecções UNKNOWN_N1 são encaminhadas ao modelo N1.
        Cada detecção: {label, conf, bbox [x1,y1,x2,y2], model, jlpt_level, meanings}
        """
        if self.model_n2_n5 is None:
            return []

        detections = []
        for r in self.model_n2_n5.predict(img, conf=self.conf_threshold, verbose=False):
            for box in r.boxes:
                label = self.model_n2_n5.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                bbox  = box.xyxy[0].tolist()  # [x1, y1, x2, y2] em pixels da imagem capturada

                if label == "UNKNOWN_N1" and self.model_n1 is not None:
                    for r1 in self.model_n1.predict(img, conf=self.conf_threshold, verbose=False):
                        for b1 in r1.boxes:
                            lbl1 = self.model_n1.names[int(b1.cls[0])]
                            if lbl1 != "UNKNOWN_BASIC":
                                detections.append({
                                    "label": lbl1,
                                    "conf":  float(b1.conf[0]),
                                    "bbox":  b1.xyxy[0].tolist(),
                                    "model": "n1",
                                })
                    continue

                if label != "UNKNOWN_N1":
                    detections.append({
                        "label": label,
                        "conf":  conf,
                        "bbox":  bbox,
                        "model": "n2_n5",
                    })

        if self.semantic:
            for d in detections:
                info = self.semantic.get_kanji_info(d["label"])
                data = info.get("data", {})
                d["meanings"]   = data.get("meanings", [])
                d["jlpt_level"] = data.get("jlpt")  # Kanjidic2: 4=N5, 3=N4, 2=N3, 1=N2

        return detections

    def toggle_capture(self):
        self.active = not self.active
        status = "ATIVO" if self.active else "INATIVO"
        print(f"[F12] Kanji OCR N2-N5: {status}")
        if self.tray_icon:
            self.tray_icon.title = f"Kanji OCR N2-N5 [{status}]"
        if not self.active and self.overlay:
            self.overlay.clear()
        if self.active:
            threading.Thread(target=self.capture_loop, daemon=True).start()

    def capture_loop(self):
        with mss.mss() as sct:
            while self.active:
                sct_img = sct.grab(sct.monitors[1])
                phys_w, phys_h = sct_img.width, sct_img.height
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

                if self.is_significant_change(img):
                    detections = self.cascade_inference(img)

                    if detections:
                        print(f"\nDetectados {len(detections)} kanji(s):")
                        for d in detections[:5]:
                            m = ", ".join(d.get("meanings", [])[:2]) or "—"
                            print(f"  {d['label']}  ({d['model']}, {d['conf']:.2f})  →  {m}")

                    if _QT_AVAILABLE and self.overlay is not None:
                        # Scale físico→lógico (relevante em displays HiDPI)
                        sx = self._screen_w / phys_w if phys_w > 0 else 1.0
                        sy = self._screen_h / phys_h if phys_h > 0 else 1.0
                        self.overlay.detection_signal.emit(detections, sx, sy)
                else:
                    time.sleep(0.5)

    def on_press(self, key):
        if key == self.hotkey:
            self.toggle_capture()

    def stop(self, icon=None, item=None):
        self.active = False
        if self.tray_icon:
            self.tray_icon.stop()
        os._exit(0)

    def run_tray(self):
        icon_img = Image.new("RGB", (64, 64), color=(0, 100, 200))
        menu = Menu(
            MenuItem("Ativar/Desativar (F12)", self.toggle_capture),
            MenuItem("Sair", self.stop),
        )
        self.tray_icon = Icon("KanjiOCR-N2N5", icon_img, "Kanji OCR N2-N5 [INATIVO]", menu)
        self.tray_icon.run()

    def start(self):
        self.load_resources()
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        if _QT_AVAILABLE:
            # pystray.run() bloqueia a thread — roda em daemon thread para liberar a main
            tray_thread = threading.Thread(target=self.run_tray, daemon=True)
            tray_thread.start()

            # Qt exige a main thread; o loop de eventos mantém o overlay vivo
            app = QApplication.instance() or QApplication(sys.argv)
            self.overlay = OverlayWindow()
            # Cache das dimensões lógicas para uso thread-safe em capture_loop
            self._screen_w = self.overlay.width()
            self._screen_h = self.overlay.height()
            sys.exit(app.exec())
        else:
            print("PyQt6 não encontrado — instale com: pip install PyQt6>=6.4.0")
            print("Saída apenas em terminal.")
            self.run_tray()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Captura de tela com OCR de kanjis N2-N5 e overlay visual."
    )
    parser.add_argument(
        "--conf", type=float,
        default=float(os.environ.get("KD_CONF", "0.25")),
        help="Confiança mínima YOLO (padrão 0.25; use 0.5+ sem GPU para reduzir falsos positivos)",
    )
    args = parser.parse_args()
    client = KanjiN2N5CaptureClient(conf=args.conf)
    client.start()

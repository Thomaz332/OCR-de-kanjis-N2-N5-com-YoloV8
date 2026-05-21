import os
import sys
import time
import threading
import numpy as np
import mss
from PIL import Image
from pynput import keyboard
from pystray import Icon, Menu, MenuItem
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic.semantic_service import SemanticService


class KanjiN2N5CaptureClient:
    """
    Cliente de captura de tela para leitura de mangás.

    Atalho F12: liga/desliga o OCR em tempo real.
    Lógica de cascata: N2-N5 primeiro; UNKNOWN_N1 delega ao modelo N1.
    Detecção de mudança de tela (>30%) para economizar GPU.
    """

    def __init__(self):
        self.active = False
        self.hotkey = keyboard.Key.f12
        self.conf_threshold = 0.25
        self.change_threshold = 0.30

        self.model_n2_n5 = None
        self.model_n1 = None
        self.semantic = None
        self.tray_icon = None
        self.last_frame = None

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

        db_path = "data/processed/kanji_dict.db"
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
        1. Modelo N2-N5 (Generalista) detecta kanjis comuns.
        2. Detecções UNKNOWN_N1 são encaminhadas ao modelo N1 (Especialista).
        """
        if self.model_n2_n5 is None:
            return []

        detections = []
        for r in self.model_n2_n5.predict(img, conf=self.conf_threshold, verbose=False):
            for box in r.boxes:
                label = self.model_n2_n5.names[int(box.cls[0])]
                conf  = float(box.conf[0])

                if label == "UNKNOWN_N1" and self.model_n1 is not None:
                    for r1 in self.model_n1.predict(img, conf=self.conf_threshold, verbose=False):
                        for b1 in r1.boxes:
                            lbl1 = self.model_n1.names[int(b1.cls[0])]
                            if lbl1 != "UNKNOWN_BASIC":
                                detections.append({"label": lbl1, "conf": float(b1.conf[0]), "model": "n1"})
                    continue

                if label != "UNKNOWN_N1":
                    detections.append({"label": label, "conf": conf, "model": "n2_n5"})

        if self.semantic:
            for d in detections:
                info = self.semantic.get_kanji_info(d["label"])
                d["meanings"] = info.get("data", {}).get("meanings", [])

        return detections

    def toggle_capture(self):
        self.active = not self.active
        status = "ATIVO" if self.active else "INATIVO"
        print(f"[F12] Kanji OCR N2-N5: {status}")
        if self.tray_icon:
            self.tray_icon.title = f"Kanji OCR N2-N5 [{status}]"
        if self.active:
            threading.Thread(target=self.capture_loop, daemon=True).start()

    def capture_loop(self):
        with mss.mss() as sct:
            while self.active:
                sct_img = sct.grab(sct.monitors[1])
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

                if self.is_significant_change(img):
                    detections = self.cascade_inference(img)
                    if detections:
                        print(f"\nDetectados {len(detections)} kanji(s):")
                        for d in detections[:5]:
                            m = ", ".join(d.get("meanings", [])[:2]) or "—"
                            print(f"  {d['label']}  ({d['model']}, {d['conf']:.2f})  →  {m}")
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
        self.run_tray()


if __name__ == "__main__":
    client = KanjiN2N5CaptureClient()
    client.start()

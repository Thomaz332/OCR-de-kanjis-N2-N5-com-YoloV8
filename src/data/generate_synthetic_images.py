import os
import random
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
import cv2
import multiprocessing


def load_fonts(font_dir):
    font_files = glob.glob(os.path.join(font_dir, "*.[ot]tf"))
    if not font_files:
        raise FileNotFoundError(
            f"Nenhuma fonte encontrada em {font_dir}. Rode src/data/download_fonts.py primeiro."
        )
    return font_files


def _compute_dark_pixel_bbox(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # O limiar de "escuro" precisa ser relativo ao fundo real da imagem, não
    # um valor absoluto fixo. bg_color é sorteado em [230, 255] por canal
    # (ver generate_class_images) -- sempre que o fundo sorteado cai abaixo
    # de um limiar fixo tipo 240, a imagem inteira (não só o kanji) contava
    # como "escura" na máscara, inflando a bbox pro quadro inteiro. Isso
    # sempre existiu no código, só ficava mascarado porque o kanji já
    # preenchia quase todo o quadro; ficou visível ao testar escalas
    # pequenas (ver docs/relatorio_n2_n5.md, seção 7.2). Aqui a cor de fundo
    # é estimada a partir da mediana dos 4 cantos (tipicamente livres do
    # kanji) e o limiar é definido bem abaixo dela.
    h, w = gray.shape
    corner_size = max(2, min(h, w) // 20)
    corners = np.concatenate([
        gray[:corner_size, :corner_size].ravel(),
        gray[:corner_size, -corner_size:].ravel(),
        gray[-corner_size:, :corner_size].ravel(),
        gray[-corner_size:, -corner_size:].ravel(),
    ])
    bg_estimate = float(np.median(corners))
    threshold = min(240.0, bg_estimate - 25.0)

    mask = gray < threshold
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        # Nenhum pixel escuro sobrou -- com kanji pequeno e perto de uma
        # borda (faixa de escala ampliada na Seção 7.4), o random_crop às
        # vezes corta o kanji inteiro pra fora do quadro. Cair pro quadro
        # inteiro aqui (como antes) gera um label errado; melhor sinalizar
        # com None e deixar generate_class_images descartar essa amostra.
        return None
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1
    return (xmin, ymin, xmax, ymax)


def draw_speech_bubble(img):
    # Simula o contorno de um balão de fala ao redor da região do kanji.
    # Desenhado só depois da bbox já ter sido medida (ver apply_augmentations),
    # entao nao interfere no calculo do label.
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = random.uniform(0.3, 0.7) * w
    cy = random.uniform(0.3, 0.7) * h
    rw = random.uniform(0.45, 0.95) * w / 2
    rh = random.uniform(0.3, 0.7) * h / 2
    color = (random.randint(0, 60),) * 3
    draw.ellipse((cx - rw, cy - rh, cx + rw, cy + rh), outline=color, width=random.randint(2, 4))
    return img


def draw_panel_border(img):
    # Simula a borda de um painel de mangá em volta do quadro inteiro.
    draw = ImageDraw.Draw(img)
    w, h = img.size
    inset = random.randint(2, 12)
    color = (random.randint(0, 40),) * 3
    draw.rectangle((inset, inset, w - inset, h - inset), outline=color, width=random.randint(2, 5))
    return img


def add_background_clutter(img):
    # Formas simples e claras espalhadas pelo fundo, simulando elementos de
    # cena (traços de ilustração, sombreamento) sem se confundir com o
    # kanji (tom sempre bem mais claro que a tinta do texto).
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(random.randint(2, 6)):
        shade = random.randint(180, 230)
        color = (shade, shade, shade)
        x1, y1 = random.uniform(0, w), random.uniform(0, h)
        x2, y2 = random.uniform(0, w), random.uniform(0, h)
        if random.random() > 0.5:
            draw.line((x1, y1, x2, y2), fill=color, width=random.randint(1, 3))
        else:
            draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), outline=color, width=1)
    return img


def apply_augmentations(img):
    img = ink_variation(img)

    if random.random() > 0.5:
        img = elastic_distortion(img)

    if random.random() > 0.6:
        img = random_crop(img)

    if random.random() > 0.3:
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor="white")

    # A bbox é medida AQUI, depois das augmentations que deslocam/deformam o
    # kanji (elastic distortion, crop, rotação), mas ANTES das que só
    # texturizam o fundo (paper_texture, screentone, ruído gaussiano). Essas
    # últimas escurecem pixels em qualquer lugar da imagem, não só no kanji
    # -- se a bbox fosse medida depois delas, um pixel de fundo escurecido
    # num canto qualquer infla a bbox para quase o quadro inteiro (bug
    # descoberto ao testar kanjis em escala pequena; com o kanji sempre
    # preenchendo o quadro isso ficava mascarado). Ver docs/relatorio_n2_n5.md.
    bbox_exact = _compute_dark_pixel_bbox(img)
    if bbox_exact is None:
        return img, None

    # Contexto de cena (balão, borda de painel, ruído de fundo): testes
    # qualitativos pós-treino (docs/relatorio_n2_n5.md, Secao 7.4) mostraram
    # que mesmo com variação de escala o modelo falhava em imagens reais que
    # têm elementos visuais nunca vistos no treino (contorno de balão, borda
    # de painel, ilustração de fundo). Adicionados aqui, depois da bbox já
    # medida, pelo mesmo motivo do texturizado abaixo.
    if random.random() > 0.65:
        img = draw_speech_bubble(img)

    if random.random() > 0.75:
        img = draw_panel_border(img)

    if random.random() > 0.6:
        img = add_background_clutter(img)

    if random.random() > 0.5:
        img = paper_texture(img)

    if random.random() > 0.7:
        img = screentone(img)

    if random.random() > 0.4:
        img_np = np.array(img)
        noise = np.random.normal(0, 15, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    return img, bbox_exact


def elastic_distortion(img, alpha=6, sigma=20):
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    dx = np.random.rand(h, w) * 2 - 1
    dy = np.random.rand(h, w) * 2 - 1
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    warped = cv2.remap(
        img_np, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return Image.fromarray(warped)


def paper_texture(img):
    noise = np.random.normal(0.95, 0.03, img.size[::-1])
    noise = np.stack([noise] * 3, axis=-1)
    img_np = np.array(img) / 255.0
    img_np *= noise
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def random_crop(img):
    w, h = img.size
    scale = random.uniform(0.75, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    x = random.randint(0, w - nw)
    y = random.randint(0, h - nh)
    img = img.crop((x, y, x + nw, y + nh))
    return img.resize((w, h), Image.BICUBIC)


def ink_variation(img):
    img_np = np.array(img.convert("L"))
    k = random.choice([1, 2])
    kernel = np.ones((k, k), np.uint8)
    if random.random() > 0.5:
        img_np = cv2.dilate(img_np, kernel)
    else:
        img_np = cv2.erode(img_np, kernel)
    return Image.fromarray(img_np).convert("RGB")


def screentone(img):
    img_np = np.array(img.convert("L")).astype(np.float32)
    h, w = img_np.shape
    y, x = np.indices((h, w))
    angle = np.deg2rad(random.uniform(15, 45))
    xr = x * np.cos(angle) + y * np.sin(angle)
    pattern = (xr % random.randint(4, 7)) < 1
    img_np[pattern] *= random.uniform(0.75, 0.9)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np).convert("RGB")


def generate_class_images(args):
    class_id, kanji, num_images_per_class, img_size, output_dir, train_split_ratio, fonts = args

    np.random.seed()
    random.seed()

    generated = 0
    for i in range(num_images_per_class):
        font_path = random.choice(fonts)

        # Variação de escala/posição: por padrão o kanji sempre preenchia o
        # quadro inteiro, centralizado. Testes pós-treino (ver
        # docs/relatorio_n2_n5.md, seção 7.2) mostraram que um modelo
        # treinado só assim falha em reconhecer o mesmo kanji, na mesma
        # fonte, quando aparece pequeno dentro de uma cena maior (balão de
        # fala, página completa) -- exatamente o caso de uso real. Uma
        # fração das imagens agora usa escala pequena e posição aleatória
        # para ensinar essa invariância de escala. O piso da faixa pequena
        # foi ampliado de 12% para 5% do quadro (Seção 7.4): medindo os
        # casos reais que falharam no primeiro reteste, o kanji ocupava
        # ~8-10% do quadro, abaixo do piso original de 12%.
        small_scale = random.random() < 0.55
        if small_scale:
            font_size = random.randint(int(img_size * 0.05), int(img_size * 0.35))
        else:
            font_size = random.randint(int(img_size * 0.5), int(img_size * 0.9))
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            continue

        bg_color = (
            random.randint(230, 255),
            random.randint(230, 255),
            random.randint(230, 255),
        )
        img = Image.new("RGB", (img_size, img_size), bg_color)
        draw = ImageDraw.Draw(img)

        try:
            bbox_text = font.getbbox(kanji)
        except Exception:
            bbox_text = draw.textbbox((0, 0), kanji, font=font)

        if not bbox_text:
            continue

        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        if small_scale:
            margin = 8
            avail_w = max(1, img_size - text_w - 2 * margin)
            avail_h = max(1, img_size - text_h - 2 * margin)
            x = margin + random.uniform(0, avail_w) - bbox_text[0]
            y = margin + random.uniform(0, avail_h) - bbox_text[1]
        else:
            x = (img_size - text_w) / 2 - bbox_text[0] + random.uniform(-20, 20)
            y = (img_size - text_h) / 2 - bbox_text[1] + random.uniform(-20, 20)

        text_color = (
            random.randint(0, 50),
            random.randint(0, 50),
            random.randint(0, 50),
        )
        draw.text((x, y), kanji, font=font, fill=text_color)

        img, bbox_exact = apply_augmentations(img)
        if bbox_exact is None:
            # O random_crop cortou o kanji inteiro pra fora do quadro (mais
            # comum com kanji pequeno perto de uma borda); descarta a
            # amostra em vez de gravar um label incorreto.
            continue
        xmin, ymin, xmax, ymax = bbox_exact

        img_w, img_h = img.size
        width = xmax - xmin
        height = ymax - ymin
        center_x = (xmin + width / 2) / img_w
        center_y = (ymin + height / 2) / img_h
        width /= img_w
        height /= img_h

        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        split = "train" if random.random() < train_split_ratio else "val"
        filename_base = f"{class_id}_{str(i).zfill(5)}"
        img_path = os.path.join(output_dir, "images", split, f"{filename_base}.jpg")
        txt_path = os.path.join(output_dir, "labels", split, f"{filename_base}.txt")

        img.save(img_path)
        with open(txt_path, "w") as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

        generated += 1

    return generated


def create_synthetic_dataset(output_dir, num_images_per_class, kanji_list, train_split_ratio=0.8):
    font_dir = os.path.join("assets", "fonts")
    fonts = load_fonts(font_dir)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    img_size = 640
    print(f"Gerando {num_images_per_class} imagens para {len(kanji_list)} classes...")
    print(f"Total esperado: {len(kanji_list) * num_images_per_class} imagens")

    args_list = [
        (class_id, kanji, num_images_per_class, img_size, output_dir, train_split_ratio, fonts)
        for class_id, kanji in enumerate(kanji_list)
    ]

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Processamento paralelo com {num_cores} núcleo(s)...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        with tqdm(total=len(kanji_list), desc="Progresso") as pbar:
            for _ in pool.imap_unordered(generate_class_images, args_list):
                pbar.update(1)

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(kanji_list)}\n")
        f.write("names:\n")
        for k in kanji_list:
            f.write(f"  - '{k}'\n")
    print(f"data.yaml salvo em: {yaml_path}")


if __name__ == "__main__":
    names_path = os.path.join("data", "processed", "n2_n5.names")

    if not os.path.exists(names_path):
        print(f"Arquivo {names_path} não encontrado!")
        print("Execute primeiro: python src/data/download_n2_n5_kanjis.py")
        import sys
        sys.exit(1)

    with open(names_path, "r", encoding="utf-8") as f:
        kanjis_list = [line.strip() for line in f if line.strip()]

    print(f"Carregados {len(kanjis_list)} kanjis de {names_path}")

    output_folder = os.path.join("data", "synthetic")
    qtd_por_classe = 100

    create_synthetic_dataset(
        output_dir=output_folder,
        num_images_per_class=qtd_por_classe,
        kanji_list=kanjis_list,
        train_split_ratio=0.8,
    )

    print("\nConcluído!")
    print(f"Dataset salvo em: {os.path.abspath(output_folder)}")

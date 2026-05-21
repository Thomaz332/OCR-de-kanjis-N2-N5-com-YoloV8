import os
import random
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
import cv2
import multiprocessing

def load_fonts(font_dir):
    # Procura tanto .ttf quanto .otf
    font_files = glob.glob(os.path.join(font_dir, "*.[ot]tf"))
    if not font_files:
        raise FileNotFoundError(f"Nenhuma fonte encontrada em {font_dir}. Rode download_fonts.py primeiro.")
    return font_files

def apply_augmentations(img):

    img = ink_variation(img)

    if random.random()>0.5:
        img = elastic_distortion(img)

    if random.random()>0.6:
        img = random_crop(img)

    if random.random() > 0.3:
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor="white")

    if random.random()>0.5:
        img = paper_texture(img)
    
    if random.random()>0.7:
        img = screentone(img)

    if random.random() > 0.4:
        img_np = np.array(img)
        noise = np.random.normal(0, 15, img_np.shape) # Ruído Gaussiano
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
        
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    mask = gray < 240

    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        # Evita caixas com dimensão zero
        if xmax == xmin: xmax += 1
        if ymax == ymin: ymax += 1
    else:
        # Fallback caso não encontre nenhum pixel escuro
        xmin, xmax = 0, img.width
        ymin, ymax = 0, img.height

    return img, (xmin, ymin, xmax, ymax)

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
        img_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return Image.fromarray(warped)

def paper_texture(img):
    noise = np.random.normal(0.95, 0.03, img.size[::-1])
    noise = np.stack([noise]*3, axis=-1)

    img_np = np.array(img)/255.0
    img_np *= noise
    img_np = np.clip(img_np*255,0,255).astype(np.uint8)

    return Image.fromarray(img_np)

def random_crop(img):
    w,h = img.size
    scale = random.uniform(0.75,1.0)

    nw, nh = int(w*scale), int(h*scale)
    x = random.randint(0, w-nw)
    y = random.randint(0, h-nh)

    img = img.crop((x,y,x+nw,y+nh))
    return img.resize((w,h), Image.BICUBIC)

def ink_variation(img):
    import cv2
    img_np = np.array(img.convert("L"))

    k = random.choice([1,2])
    kernel = np.ones((k,k), np.uint8)

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

    xr = x*np.cos(angle) + y*np.sin(angle)

    pattern = (xr % random.randint(4,7)) < 1

    img_np[pattern] *= random.uniform(0.75, 0.9)

    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return Image.fromarray(img_np).convert("RGB")

def generate_class_images(args):
    class_id, kanji, num_images_per_class, img_size, output_dir, train_split_ratio, fonts = args
    
    # Reinicia o gerador de números aleatórios para garantir variabilidade em cada processo paralelo
    np.random.seed()
    random.seed()
    
    generated = 0
    for i in range(num_images_per_class):
        # 1. Configuração Aleatória
        font_path = random.choice(fonts)
        
        # Varia o tamanho da fonte entre 50% e 90% da imagem
        font_size = random.randint(int(img_size * 0.5), int(img_size * 0.9))
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            continue
        
        # Fundo: Branco ou levemente amarelado (papel velho)
        bg_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))
        img = Image.new("RGB", (img_size, img_size), bg_color)
        draw = ImageDraw.Draw(img)
        
        # 2. Centralizar Texto
        # getbbox retorna (left, top, right, bottom)
        try:
            bbox_text = font.getbbox(kanji)
        except:
            # Fallback para versões antigas do Pillow se der erro
            bbox_text = draw.textbbox((0, 0), kanji, font=font)

        if not bbox_text: 
            continue 
        
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        
        # Posição central calculada
        x = (img_size - text_w) / 2 - bbox_text[0]
        y = (img_size - text_h) / 2 - bbox_text[1]
        
        # Deslocamento aleatório pequeno (Shift)
        x += random.uniform(-20, 20)
        y += random.uniform(-20, 20)
        
        # Cor do texto (Preto ou cinza escuro)
        text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        draw.text((x, y), kanji, font=font, fill=text_color)
        
        # 3. Augmentations e recuperação da Bounding Box exata
        img, bbox_exact = apply_augmentations(img)
        xmin, ymin, xmax, ymax = bbox_exact
        
        # 4. Cálculo do BBox no formato YOLO
        img_w, img_h = img.size
        width = xmax - xmin
        height = ymax - ymin
        center_x = xmin + (width / 2)
        center_y = ymin + (height / 2)
        
        # Normaliza para 0 a 1
        center_x /= img_w
        center_y /= img_h
        width /= img_w
        height /= img_h
        
        # Garante limites
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        # 5. Salvar em Train ou Val
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
    
    # Estrutura YOLO: images/(train|val) e labels/(train|val)
    splits = ["train", "val"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
    
    img_size = 640
    
    print(f"Gerando {num_images_per_class} imagens para {len(kanji_list)} classes...")
    print(f"Total de imagens esperadas: {len(kanji_list) * num_images_per_class}")
    
    # Preparar argumentos para o multiprocessamento
    args_list = []
    for class_id, kanji in enumerate(kanji_list):
        args_list.append((
            class_id, kanji, num_images_per_class, img_size, output_dir, train_split_ratio, fonts
        ))
        
    # Usar todos os núcleos físicos disponíveis menos 1 (para o sistema)
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Iniciando processamento paralelo com {num_cores} núcleos (CPUs)...")
    
    # Barra de progresso baseada nas classes!
    with multiprocessing.Pool(processes=num_cores) as pool:
        with tqdm(total=len(kanji_list), desc="Progresso") as pbar:
            for _ in pool.imap_unordered(generate_class_images, args_list):
                pbar.update(1)

    # 6. Criar data.yaml automaticamente
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        # No Kaggle, o caminho será /kaggle/working/data/synthetic se rodar lá
        # Ou um caminho absoluto. Para funcionar bem em qualquer lugar:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(kanji_list)}\n")
        f.write("names:\n")
        for k in kanji_list:
            f.write(f"  - '{k}'\n")
    print(f"data.yaml salvo em: {yaml_path}")


if __name__ == "__main__":
    # Ler a lista completa de Kanjis N1
    names_path = os.path.join("data", "processed", "n1.names")
    
    if os.path.exists(names_path):
        with open(names_path, "r", encoding="utf-8") as f:
            kanjis_list = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Carregados {len(kanjis_list)} kanjis de {names_path}")
    else:
        print(f"Arquivo {names_path} não encontrado!")
        print("Certifique-se de rodar download_n1_kanjis.py primeiro.")
        import sys
        sys.exit(1)
        
    # Pasta de saída
    output_folder = os.path.join("data", "synthetic")
    
    # Quantidade de imagens por Kanji
    qtd_por_classe = 100 # Mude para 100+ para treinamento forte, mas pode levar algumas horas localmente
    
    create_synthetic_dataset(
        output_dir=output_folder, 
        num_images_per_class=qtd_por_classe, 
        kanji_list=kanjis_list, 
        train_split_ratio=0.8
    )
    
    print("\nConcluído!")
    print(f"Dataset salvo em: {os.path.abspath(output_folder)}")
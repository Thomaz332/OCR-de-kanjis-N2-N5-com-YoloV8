import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
import json
from tqdm import tqdm


def parse_manga109(manga109_root, output_dir):
    """Parseia o Manga109, extrai recortes de texto e salva metadados JSON."""
    annotations_dir = os.path.join(manga109_root, "annotations")
    images_dir = os.path.join(manga109_root, "images")
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    metadata = []
    books = sorted(f for f in os.listdir(annotations_dir) if f.endswith(".xml"))
    print(f"Encontrados {len(books)} volumes em {annotations_dir}")

    for book_file in tqdm(books, desc="Parseando volumes"):
        book_title = os.path.splitext(book_file)[0]
        xml_path = os.path.join(annotations_dir, book_file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            pages = root.find("pages")
            if pages is None:
                continue

            for page in pages.findall("page"):
                page_index = page.get("index")
                image_path = os.path.join(
                    images_dir, book_title, f"{str(page_index).zfill(3)}.jpg"
                )
                if not os.path.exists(image_path):
                    continue

                try:
                    img = Image.open(image_path)
                except Exception:
                    continue

                for text_elem in page.findall("text"):
                    text_id = text_elem.get("id")
                    xmin = int(text_elem.get("xmin"))
                    ymin = int(text_elem.get("ymin"))
                    xmax = int(text_elem.get("xmax"))
                    ymax = int(text_elem.get("ymax"))
                    content = text_elem.text

                    if not content or xmin >= xmax or ymin >= ymax:
                        continue

                    try:
                        crop = img.crop((xmin, ymin, xmax, ymax))
                        crop_filename = f"{book_title}_{page_index}_{text_id}.jpg"
                        crop.save(os.path.join(crops_dir, crop_filename))
                        metadata.append({
                            "id": text_id,
                            "book": book_title,
                            "page": page_index,
                            "bbox": [xmin, ymin, xmax, ymax],
                            "text": content,
                            "file": crop_filename,
                            "width": xmax - xmin,
                            "height": ymax - ymin,
                        })
                    except Exception:
                        continue

        except Exception as e:
            print(f"Erro ao parsear {book_title}: {e}")
            continue

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nConcluído: {len(metadata)} regiões de texto salvas em {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser do Manga109 para recortes de texto.")
    parser.add_argument("--root", required=True, help="Diretório raiz do Manga109")
    parser.add_argument("--output", default="data/processed/manga109", help="Diretório de saída")
    args = parser.parse_args()
    parse_manga109(args.root, args.output)

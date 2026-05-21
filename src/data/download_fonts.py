import os
import requests


def download_file(url, output_path):
    print(f"Baixando: {os.path.basename(output_path)}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(" -> Sucesso!")
        return True
    except Exception as e:
        print(f" -> Erro: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def main():
    output_dir = os.path.join("assets", "fonts")
    os.makedirs(output_dir, exist_ok=True)

    fonts = {
        "NotoSansCJKjp-Regular.otf":
            "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
        "NotoSerifCJKjp-Regular.otf":
            "https://github.com/notofonts/noto-cjk/raw/main/Serif/OTF/Japanese/NotoSerifCJKjp-Regular.otf",
    }

    print(f"Iniciando download de {len(fonts)} fontes para: {output_dir}")
    print("-" * 50)

    sucessos = 0
    for filename, url in fonts.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"Pulado (já existe): {filename}")
            sucessos += 1
            continue
        if download_file(url, filepath):
            sucessos += 1

    print("-" * 50)
    print(f"Concluído: {sucessos}/{len(fonts)} fontes prontas.")


if __name__ == "__main__":
    main()

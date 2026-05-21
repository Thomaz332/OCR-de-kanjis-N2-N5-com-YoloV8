import os
import requests

def download_file(url, output_path):
    print(f"Baixando: {os.path.basename(output_path)}...")
    try:
        # User-Agent para evitar bloqueio do GitHub
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f" -> Sucesso!")
        return True
    except Exception as e:
        print(f" -> Erro: {e}")
        # Remove arquivo corrompido/incompleto se existir
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def main():
    # Diretório de destino
    output_dir = os.path.join("assets", "fonts")
    os.makedirs(output_dir, exist_ok=True)

    # Dicionário de Fontes: Nome -> URL Direta (Raw)
    # Selecionamos fontes com estética variada para enriquecer o treino do YOLO
    fonts = {
        # Noto CJK (Essenciais - Japonês Padrão)
        "NotoSansCJKjp-Regular.otf": 
        "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
        
        "NotoSerifCJKjp-Regular.otf": 
        "https://github.com/notofonts/noto-cjk/raw/main/Serif/OTF/Japanese/NotoSerifCJKjp-Regular.otf",
    }

    print(f"Iniciando download de {len(fonts)} fontes para: {output_dir}")
    print("-" * 50)
    
    sucessos = 0
    for filename, url in fonts.items():
        # Caminho final do arquivo
        filepath = os.path.join(output_dir, filename)
        
        # Verifica se já existe para não baixar de novo
        if os.path.exists(filepath):
            print(f"Pulado (já existe): {filename}")
            sucessos += 1
            continue
            
        if download_file(url, filepath):
            sucessos += 1
            
    print("-" * 50)
    print(f"Concluído: {sucessos}/{len(fonts)} fontes prontas para uso.")

if __name__ == "__main__":
    main()
import os
import yaml
import argparse


def create_config(names_path, dataset_dir, output_yaml):
    with open(names_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    data = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }

    os.makedirs(os.path.dirname(output_yaml) or ".", exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Configuração YOLO salva em: {output_yaml}")
    print(f"  Classes (nc): {len(names)}")
    print(f"  Train: {os.path.join(dataset_dir, 'images/train')}")
    print(f"  Val:   {os.path.join(dataset_dir, 'images/val')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", default="data/processed/n2_n5.names")
    parser.add_argument("--dataset_dir", default="data/synthetic")
    parser.add_argument("--output_yaml", default="data/synthetic/data.yaml")
    args = parser.parse_args()
    create_config(args.names, args.dataset_dir, args.output_yaml)

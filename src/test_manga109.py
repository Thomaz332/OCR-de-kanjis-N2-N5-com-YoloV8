import os
import random
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on random Manga109 images")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to your best.pt weights file")
    parser.add_argument("--num_images", type=int, default=5, help="Number of random images to test")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection")
    args = parser.parse_args()

    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    manga109_dir = BASE_DIR / "data" / "raw" / "Manga109" / "images"
    output_dir = BASE_DIR / "runs" / "inference" / "manga109"

    if not manga109_dir.exists():
        print(f"Error: Manga109 images directory not found at {manga109_dir}")
        print("Please ensure you moved the dataset correctly.")
        return

    if not Path(args.model).exists():
        print(f"Error: Model weights not found at {args.model}")
        print("Please provide the correct path using --model flag (e.g. --model runs/detect/train/weights/best.pt)")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image paths
    all_images = []
    for manga_book_dir in manga109_dir.iterdir():
        if manga_book_dir.is_dir():
            for img_path in manga_book_dir.glob("*.jpg"):
                all_images.append(img_path)

    if not all_images:
        print("No images found in the dataset.")
        return

    # Select random images
    selected_images = random.sample(all_images, min(args.num_images, len(all_images)))

    # Load YOLO model
    print(f"Loading model {args.model}...")
    model = YOLO(args.model)

    print(f"\nRunning inference on {len(selected_images)} random images...")
    
    # Run inference
    for img_path in selected_images:
        print(f"Processing {img_path.name} from {img_path.parent.name}...")
        
        # Run prediction
        results = model(str(img_path), conf=args.conf)
        
        # Save results
        for idx, result in enumerate(results):
            # Generate save path
            save_name = f"{img_path.parent.name}_{img_path.stem}_pred.jpg"
            save_path = output_dir / save_name
            
            # Save the image with boxes drawn
            result.save(filename=str(save_path))
            print(f"  -> Saved detection to: {save_path}")

    print(f"\n✅ Testing complete! Check the {output_dir} folder for results.")

if __name__ == "__main__":
    main()

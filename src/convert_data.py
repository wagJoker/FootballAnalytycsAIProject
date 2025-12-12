import os
import pandas as pd
import glob
from pathlib import Path

# Config
DATA_DIR = Path('E:/MyProject/FootballAI/data/football_kaggle')
CLASS_MAP = {
    'ball': 0,
    'goalkeeper': 1,
    'player': 2,
    'referee': 3
}

def convert_split(split_name):
    images_dir = DATA_DIR / 'images' / split_name
    labels_dir = DATA_DIR / 'labels' / split_name
    
    # Create labels directory
    os.makedirs(labels_dir, exist_ok=True)
    
    # Find CSV file
    csv_files = glob.glob(str(images_dir / '*.csv'))
    if not csv_files:
        print(f"No CSV found in {images_dir}")
        return
        
    csv_path = csv_files[0]
    print(f"Processing {split_name} from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Check classes
    unique_classes = df['class'].unique()
    print(f"Found classes: {unique_classes}")
    for cls in unique_classes:
        if cls not in CLASS_MAP:
            print(f"WARNING: Unknown class '{cls}' will be skipped!")

    # Group by filename
    grouped = df.groupby('filename')
    
    count = 0
    for filename, group in grouped:
        # Image might have different extension in csv vs reality, but usually matches basename
        # YOLO expects distinct file for each image image.jpg -> image.txt
        
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = labels_dir / txt_filename
        
        lines = []
        for _, row in group.iterrows():
            cls_name = row['class']
            if cls_name not in CLASS_MAP:
                continue
                
            cls_id = CLASS_MAP[cls_name]
            
            # Dimensions
            img_w = row['width']
            img_h = row['height']
            
            # Box
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            
            # Normalize to center x, y, w, h
            dw = 1.0 / img_w
            dh = 1.0 / img_h
            
            w = xmax - xmin
            h = ymax - ymin
            x_center = xmin + w / 2.0
            y_center = ymin + h / 2.0
            
            x_center *= dw
            w *= dw
            y_center *= dh
            h *= dh
            
            # Clip to 0-1 just in case
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines))
        count += 1
        
    print(f"Created {count} label files for {split_name}.")

def main():
    convert_split('train')
    convert_split('val')
    print("Conversion complete.")

if __name__ == '__main__':
    main()

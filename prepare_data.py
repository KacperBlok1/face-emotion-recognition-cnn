import os
from utils.helpers import EMOTIONS, create_dir_if_not_exists

def main():
    print("=== FER-2013 Data Preparation Utility ===")
    
    base_data_dir = "data"
    subfolders = ["train", "test"]
    
    print("\nCreating expected folder structure...")
    
    for folder in subfolders:
        folder_path = os.path.join(base_data_dir, folder)
        create_dir_if_not_exists(folder_path)
        print(f"Directory verified: {folder_path}")
        
        # Create directories for each emotion (lowercased)
        for emotion in EMOTIONS:
            emotion_folder = os.path.join(folder_path, emotion.lower())
            create_dir_if_not_exists(emotion_folder)
            
    print("\nExpected directory tree:")
    print("data/")
    print("  |-- train/")
    for emotion in EMOTIONS:
        print(f"  |   |-- {emotion.lower()}/")
    print("  \\-- test/")
    for emotion in EMOTIONS:
        print(f"      |-- {emotion.lower()}/")
        
    print("\n[SUCCESS] Directories prepared successfully!")
    print("Please download the FER-2013 dataset from Kaggle and place your training and test images into these respective subfolders.")

if __name__ == "__main__":
    main()

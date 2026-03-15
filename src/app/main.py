import os
import sys

# Add src to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocess.video_processor import process_video
from features.extractor import extract_features
from matcher.engine import MatchEngine

def main():
    print("--- SceneSeek Prototype ---")
    # Placeholder for main execution flow
    print("Welcome to SceneSeek. This prototype identifies movies from video clips.")

if __name__ == "__main__":
    main()

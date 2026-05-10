import os
import sys

# Get the absolute path to the root of our project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the 'src' directory to Python's module path so we can import our own scripts easily
sys.path.append(os.path.join(BASE_DIR, "src"))

# Import our custom functions from the other modules we built
from preprocess.extract_frames import extract_frames
from features.extract_features import extract_features
from matcher.match_features import match_features

def run_pipeline(test_video_path):
    """
    Runs the full end-to-end SceneSeek pipeline:
    1. Extracts frames from the test video.
    2. Converts those frames into feature vectors using our AI model.
    3. Matches the features against the real reference database to identify the movie.
    """
    print("\n" + "="*50)
    print("🚀 STARTING SCENESEEK PIPELINE")
    print("="*50)
    print(f"Analyzing video: {test_video_path}")
    
    if not os.path.exists(test_video_path):
        print(f"\n❌ Error: Test video not found at {test_video_path}")
        print("Please add a test video to the data/test/ folder before running the pipeline.")
        return

    # --- Setup Directories ---
    # Where to save the temporarily extracted frames
    frames_dir = os.path.join(BASE_DIR, "outputs", "frames")
    
    # Where to save the AI-extracted features for our test clip
    features_dir = os.path.join(BASE_DIR, "outputs", "features")
    features_path = os.path.join(features_dir, "features.npy")
    
    # Where to find our real pre-built database of movie features
    reference_db_dir = os.path.join(BASE_DIR, "outputs", "reference_features")

    # --- Step 1: Extract Frames ---
    print("\n[STEP 1/3] Extracting frames from test clip...")
    # We extract 5 frames from the test clip to keep things fast
    extract_frames(test_video_path, frames_dir, num_frames=5)

    # --- Step 2: Extract Features ---
    print("\n[STEP 2/3] Extracting AI features from frames...")
    # This will read the images in frames_dir and save a features.npy to features_dir
    extract_features(frames_dir, features_dir)

    # --- Step 3: Match Features ---
    print("\n[STEP 3/3] Matching clip against the Reference Database...")
    # This function reads our new features.npy and compares it to all movies in reference_db_dir
    best_match = match_features(features_path, reference_db_dir)
    
    print("\n✨ PIPELINE COMPLETE ✨")
    print("="*50 + "\n")
    
    return best_match

if __name__ == "__main__":
    # --- Configuration ---
    
    # Define the path to a test video you want to identify.
    # Feel free to change "test_clip.mp4" to whatever your file is named!
    test_video = os.path.join(BASE_DIR, "data", "test", "test_avatar_1.mp4")
    
    # Run the entire pipeline with a single function call
    run_pipeline(test_video)

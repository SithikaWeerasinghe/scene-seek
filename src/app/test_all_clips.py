import os
import sys
import shutil

# Get the absolute path to the root of our project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the 'src' directory to Python's module path so we can import our own scripts easily
sys.path.append(os.path.join(BASE_DIR, "src"))

# Import our custom functions from the other modules we built
from app.run_pipeline import run_pipeline

def clear_directory(directory_path):
    """
    Safely deletes all files in a directory to ensure a clean slate for the next test.
    Creates the directory if it doesn't exist.
    """
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(directory_path, exist_ok=True)

def test_all_clips():
    """
    Tests all 5 test clips through the SceneSeek pipeline and prints a summary.
    """
    print("\n" + "="*50)
    print("🎬 SCENESEEK: BATCH TESTING ALL CLIPS 🎬")
    print("="*50)
    
    # Define our test cases
    # The dictionary maps the filename to the expected movie name
    test_cases = {
        "test_avatar_1.mp4": "avatar",
        "test_interstellar_1.mp4": "interstellar",
        "test_titanic_1.mp4": "titanic",
        "test_inception_1.mp4": "inception",
        "test_harry_potter_1.mp4": "harry_potter"
    }
    
    # Store results for the final summary table
    results = []
    correct_count = 0
    
    # Setup directories to clear
    frames_dir = os.path.join(BASE_DIR, "outputs", "frames")
    features_dir = os.path.join(BASE_DIR, "outputs", "features")
    test_data_dir = os.path.join(BASE_DIR, "data", "test")
    
    for idx, (filename, expected_movie) in enumerate(test_cases.items(), 1):
        print("\n" + "-"*50)
        print(f"[{idx}/{len(test_cases)}] Testing Clip: {filename}")
        print("-"*50)
        
        test_video_path = os.path.join(test_data_dir, filename)
        
        # Check if the file actually exists
        if not os.path.exists(test_video_path):
            print(f"❌ Error: Test video not found at {test_video_path}")
            results.append({
                "clip": filename,
                "expected": expected_movie,
                "predicted": "File Not Found",
                "is_correct": False
            })
            continue
            
        # Clear output directories safely before each run to avoid using old files
        print("🧹 Cleaning output directories (frames & features)...")
        clear_directory(frames_dir)
        clear_directory(features_dir)
        
        # Run the full pipeline!
        print("🚀 Running full pipeline...")
        predicted_movie = run_pipeline(test_video_path)
        
        # Verify correctness
        is_correct = (predicted_movie == expected_movie)
        if is_correct:
            correct_count += 1
            print(f"✅ Prediction is CORRECT!")
        else:
            print(f"❌ Prediction is INCORRECT (Expected: {expected_movie})")
            
        # Store result
        results.append({
            "clip": filename,
            "expected": expected_movie,
            "predicted": str(predicted_movie), # Handle None case
            "is_correct": is_correct
        })
        
    # --- Print Final Summary Table ---
    print("\n" + "="*75)
    print("📊 FINAL SUMMARY REPORT")
    print("="*75)
    
    # Print table header
    print(f"{'Test Clip Name':<30} | {'Expected Movie':<15} | {'Predicted Movie':<15} | {'Result':<10}")
    print("-" * 75)
    
    # Print each result row
    for res in results:
        clip_name = res["clip"]
        expected = res["expected"]
        predicted = res["predicted"]
        result_text = "✅ Pass" if res["is_correct"] else "❌ Fail"
        
        print(f"{clip_name:<30} | {expected:<15} | {predicted:<15} | {result_text:<10}")
        
    print("-" * 75)
    print(f"Total Correct: {correct_count} out of {len(test_cases)}")
    
    accuracy = (correct_count / len(test_cases)) * 100
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print("="*75 + "\n")

if __name__ == "__main__":
    test_all_clips()

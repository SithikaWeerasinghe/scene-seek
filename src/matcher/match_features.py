import numpy as np
import os

def load_input_features(filepath):
    """Loads the extracted features from the input video numpy file."""
    # Check if the file exists before trying to load it
    if not os.path.exists(filepath):
        print(f"Error: Feature file not found at '{filepath}'")
        print("Please ensure you have run the feature extraction step first.")
        return None
    
    # Load the numpy array containing our frame features
    features = np.load(filepath)
    print(f"Loaded input features with shape: {features.shape}")
    return features

def load_reference_database(reference_dir):
    """
    Loads the real reference database of movie features.
    Reads each movie's feature vector from its folder.
    """
    print("\n--- Loading Reference Database ---")
    
    if not os.path.exists(reference_dir):
        print(f"Error: Reference database directory not found at '{reference_dir}'")
        print("Please run the build_reference_database.py script first.")
        return None
        
    reference_db = {}
    
    # Get all subdirectories (each represents a movie)
    movie_folders = [f.path for f in os.scandir(reference_dir) if f.is_dir()]
    
    for movie_folder in movie_folders:
        movie_name = os.path.basename(movie_folder)
        feature_file = os.path.join(movie_folder, "features.npy")
        
        if os.path.exists(feature_file):
            # Load the pre-calculated feature vector for this movie
            movie_vector = np.load(feature_file)
            reference_db[movie_name] = movie_vector
        else:
            print(f"  Warning: No features.npy found for movie '{movie_name}'")
            
    print(f"Loaded {len(reference_db)} movies from the reference database.")
    return reference_db

def calculate_cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors using numpy.
    Cosine similarity measures the angle between two vectors, resulting in a value 
    between -1 and 1, where 1 means they are exactly the same direction (most similar).
    """
    # Ensure vectors are 1D arrays
    v1 = vec1.flatten()
    v2 = vec2.flatten()
    
    # Cosine similarity formula: (A dot B) / (||A|| * ||B||)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid division by zero just in case
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return dot_product / (norm_v1 * norm_v2)

def match_features(input_features_path, reference_db_dir):
    """Main function to perform feature matching."""
    print("=== SceneSeek: Feature Matching Prototype ===\n")

    # 1. Load the features extracted from the input video
    input_features = load_input_features(input_features_path)
    if input_features is None:
        return

    # 2. Average the frame features into one clip-level feature vector
    # input_features shape is typically (num_frames, feature_dim)
    # We average across the frames (axis=0) to get a single 1D vector representing the whole clip
    print("\nAveraging frame features to create a clip-level signature...")
    clip_feature = np.mean(input_features, axis=0)
    
    print(f"Averaged clip feature shape: {clip_feature.shape}")

    # 3. Load the real database of movie features to match against
    reference_db = load_reference_database(reference_db_dir)
    if not reference_db:
        return

    # 4. Compare input clip features against each movie in the database
    print("\n--- Matching Results ---")
    best_match = None
    highest_similarity = -1.0

    for movie, db_feature in reference_db.items():
        # Calculate cosine similarity between our input clip and the database movie
        similarity_score = calculate_cosine_similarity(clip_feature, db_feature)
        
        print(f"Similarity with {movie:15}: {similarity_score:.4f}")
        
        # Keep track of the best match seen so far
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            best_match = movie

    # 5. Output the final prediction
    print("\n=============================================")
    if best_match:
        print(f"🌟 BEST MATCH: {best_match} (Score: {highest_similarity:.4f}) 🌟")
    else:
        print("No matches found. Is the reference database empty?")
    print("=============================================")

if __name__ == "__main__":
    # --- Configuration ---
    
    # Get the root directory of the project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Define paths based on the project root
    input_features_path = os.path.join(base_dir, "outputs", "features", "features.npy")
    reference_db_dir = os.path.join(base_dir, "outputs", "reference_features")
    
    # Run the matcher
    match_features(input_features_path, reference_db_dir)

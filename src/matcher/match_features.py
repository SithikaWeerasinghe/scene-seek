import numpy as np
import os

# Define the path to our input features
# Assuming the script runs from the project root
INPUT_FEATURES_PATH = "outputs/features/features.npy"

# We'll use this list of movies for our simulated database
MOVIES = [
    "Avatar",
    "Interstellar",
    "Titanic",
    "Inception",
    "Harry Potter"
]

def load_input_features(filepath):
    """Loads the extracted features from the numpy file."""
    # Check if the file exists before trying to load it
    if not os.path.exists(filepath):
        print(f"Error: Feature file not found at '{filepath}'")
        print("Please ensure you have run the feature extraction step first.")
        return None
    
    # Load the numpy array containing our frame features
    features = np.load(filepath)
    print(f"Loaded input features with shape: {features.shape}")
    return features

def create_mock_database(feature_dim):
    """
    Creates a simulated database of movie features.
    In a real app, these would be pre-calculated from full movies and loaded from a database.
    Here, we just generate random features of the same dimension for demonstration.
    """
    print("\n--- Creating Mock Database ---")
    mock_db = {}
    
    # We set a fixed random seed so the "random" database is the same every time you run it,
    # which makes testing easier and more consistent for beginners.
    np.random.seed(42) 
    
    for movie in MOVIES:
        # Generate a random feature vector of the correct size
        random_features = np.random.rand(feature_dim)
        
        # Normalize the random features (important for cosine similarity)
        norm = np.linalg.norm(random_features)
        if norm > 0:
            normalized_features = random_features / norm
        else:
            normalized_features = random_features
            
        mock_db[movie] = normalized_features
        
    print(f"Created mock database with {len(MOVIES)} movies.")
    return mock_db

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

def match_features():
    """Main function to perform feature matching."""
    print("=== SceneSeek: Feature Matching Prototype ===\n")

    # 1. Load the features extracted from the input video
    input_features = load_input_features(INPUT_FEATURES_PATH)
    if input_features is None:
        return

    # 2. Average the frame features into one clip-level feature vector
    # input_features shape is typically (num_frames, feature_dim)
    # We average across the frames (axis=0) to get a single 1D vector representing the whole clip
    print("\nAveraging frame features to create a clip-level signature...")
    clip_feature = np.mean(input_features, axis=0)
    
    # Get the dimension of the feature vector (e.g., 2048 for ResNet50)
    feature_dim = clip_feature.shape[0]
    print(f"Averaged clip feature shape: {clip_feature.shape}")

    # 3. Create a simulated database of movie features to match against
    mock_db = create_mock_database(feature_dim)

    # 4. Compare input clip features against each movie in the database
    print("\n--- Matching Results ---")
    best_match = None
    highest_similarity = -1.0

    for movie, db_feature in mock_db.items():
        # Calculate cosine similarity between our input clip and the database movie
        similarity_score = calculate_cosine_similarity(clip_feature, db_feature)
        
        print(f"Similarity with {movie:15}: {similarity_score:.4f}")
        
        # Keep track of the best match seen so far
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            best_match = movie

    # 5. Output the final prediction
    print("\n=============================================")
    print(f"🌟 BEST MATCH: {best_match} (Score: {highest_similarity:.4f}) 🌟")
    print("=============================================")

if __name__ == "__main__":
    match_features()

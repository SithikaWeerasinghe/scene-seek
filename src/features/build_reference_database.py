import os
import glob
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def get_resnet_model():
    """Loads and prepares the pretrained ResNet18 model."""
    print("Loading pretrained ResNet18 model...")
    # Use ResNet18 as in the previous step
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Remove the final classification layer to get raw features instead of class predictions
    modules = list(model.children())[:-1] 
    model = torch.nn.Sequential(*modules)
    model.eval() 
    return model

def get_transforms():
    """Defines the image transformations expected by PyTorch."""
    return transforms.Compose([
        transforms.Resize(256),                  # Resize smaller edge to 256
        transforms.CenterCrop(224),              # Crop the center 224x224 pixels
        transforms.ToTensor(),                   # Convert image to a PyTorch Tensor
        transforms.Normalize(                    # Normalize using ImageNet statistics
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

def extract_frames_from_video(video_path, num_frames=5):
    """
    Reads a video and extracts a fixed number of evenly spaced frames.
    Returns a list of PIL Images. We extract frames directly in memory 
    without saving them to disk first.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Warning: Could not open video {video_path}")
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"    Warning: Video {video_path} has 0 frames.")
        cap.release()
        return []
        
    # Calculate indices of frames to extract
    # We use linspace to get evenly spaced frames across the entire video length
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    extracted_images = []
    
    for idx in frame_indices:
        # Jump to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # OpenCV loads images in BGR color format by default. 
            # We need to convert it to RGB format for PyTorch.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert the numpy array to a PIL Image
            pil_img = Image.fromarray(frame_rgb)
            extracted_images.append(pil_img)
            
    cap.release()
    return extracted_images

def build_database(reference_dir, output_base_dir):
    """
    Main function to process all movies and build the reference database.
    """
    print(f"Looking for movie folders in: {reference_dir}")
    
    # Check if the reference directory exists
    if not os.path.exists(reference_dir):
        print(f"Error: The directory {reference_dir} does not exist.")
        print("Please create it and add some movie folders with .mp4 clips.")
        return
        
    # Get all subdirectories (each subdirectory represents a different movie)
    movie_folders = [f.path for f in os.scandir(reference_dir) if f.is_dir()]
    
    if not movie_folders:
        print("No movie folders found in the reference directory.")
        return
        
    print(f"Found {len(movie_folders)} movie(s) to process.")
    
    # Prepare the AI model and image transformations
    model = get_resnet_model()
    preprocess = get_transforms()
    
    # Process each movie one by one
    for movie_folder in movie_folders:
        movie_name = os.path.basename(movie_folder)
        print(f"\n--- Processing Movie: {movie_name} ---")
        
        # Find all .mp4 clips in this movie's folder
        clip_paths = glob.glob(os.path.join(movie_folder, "*.mp4"))
        
        if not clip_paths:
            print(f"  Warning: No .mp4 clips found for {movie_name}")
            continue
            
        print(f"  Found {len(clip_paths)} clips.")
        
        all_movie_features = []
        
        # We don't need to track gradients during feature extraction
        with torch.no_grad():
            for clip_path in clip_paths:
                clip_name = os.path.basename(clip_path)
                print(f"  Extracting features from clip: {clip_name}")
                
                # Extract a few frames from the video (5 frames per clip)
                frames = extract_frames_from_video(clip_path, num_frames=5)
                
                for img in frames:
                    # Preprocess the frame and add a fake batch dimension expected by PyTorch
                    input_tensor = preprocess(img).unsqueeze(0)
                    
                    # Pass the image through the model to get features
                    output = model(input_tensor)
                    
                    # Flatten the output to a 1D vector (512 numbers for ResNet18)
                    feature_vector = output.squeeze().numpy()
                    all_movie_features.append(feature_vector)
                    
        # If we successfully extracted features for this movie
        if all_movie_features:
            # 1. Convert the list to a numpy array
            features_array = np.array(all_movie_features)
            
            # 2. Average all frames across all clips into a single vector 
            # This single vector will represent the entire movie!
            movie_vector = np.mean(features_array, axis=0)
            
            # 3. Save the result
            # Create a specific output folder for this movie
            movie_out_dir = os.path.join(output_base_dir, movie_name)
            os.makedirs(movie_out_dir, exist_ok=True)
            
            out_file = os.path.join(movie_out_dir, "features.npy")
            np.save(out_file, movie_vector)
            
            print(f"  ✅ Saved movie feature vector of shape {movie_vector.shape} to: {out_file}")
        else:
            print(f"  ❌ Failed to extract any features for {movie_name}")
            
    print("\n=== Database Building Complete! ===")

if __name__ == "__main__":
    # --- Configuration ---
    
    # Get the root directory of the project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Where to find the reference movies
    # We expect this structure: data/reference/Movie_Name/clip.mp4
    reference_dir = os.path.join(base_dir, "data", "reference")
    
    # Where to save the database features
    output_features_dir = os.path.join(base_dir, "outputs", "reference_features")
    
    # --- Execution ---
    print("=== SceneSeek: Building Reference Database ===")
    build_database(reference_dir, output_features_dir)

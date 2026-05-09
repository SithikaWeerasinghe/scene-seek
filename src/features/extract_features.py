import os
import glob
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def extract_features(frames_dir, output_dir):
    """
    Reads frame images from a directory, uses a pretrained ResNet model to extract 
    feature vectors, and saves the vectors and filenames.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, "features.npy")
    names_path = os.path.join(output_dir, "frame_names.txt")
    
    # 2. Find all .jpg images in the frames directory
    image_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    if not image_paths:
        print(f"Warning: No .jpg images found in {frames_dir}")
        print("Please run the extract_frames.py script first.")
        return
        
    print(f"Found {len(image_paths)} images to process.")

    # 3. Load a pretrained ResNet model
    print("Loading pretrained ResNet18 model...")
    # ResNet18 is a good, lightweight model for beginner projects.
    # We use DEFAULT weights which means it was pre-trained on ImageNet.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # By default, ResNet outputs a "class prediction" (e.g., "dog", "car").
    # We don't want the prediction; we want the raw mathematical features before the prediction.
    # So we remove the final Fully Connected ('fc') classification layer.
    modules = list(model.children())[:-1] 
    model = torch.nn.Sequential(*modules)
    
    # Set model to evaluation mode (turns off training-specific behavior like dropout)
    model.eval() 

    # 4. Define how images should be prepared for the model
    # PyTorch models expect images to be a specific size and format.
    preprocess = transforms.Compose([
        transforms.Resize(256),                  # Resize smaller edge to 256
        transforms.CenterCrop(224),              # Crop the center 224x224 pixels
        transforms.ToTensor(),                   # Convert image to a PyTorch Tensor
        transforms.Normalize(                    # Normalize using ImageNet statistics
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    features_list = []
    frame_names_list = []

    # 5. Process each image
    print("Extracting features from images...")
    
    # We use torch.no_grad() because we are not training the model. 
    # This saves memory and makes the process faster.
    with torch.no_grad(): 
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            print(f"  Processing {filename}...")
            
            try:
                # Load the image and ensure it has 3 color channels (RGB)
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"    Error reading {filename}: {e}")
                continue
                
            # Preprocess the image
            input_tensor = preprocess(img)
            
            # PyTorch models expect a "batch" of images, even if it's just one.
            # unsqueeze(0) adds a fake batch dimension: shape changes from [C, H, W] to [1, C, H, W]
            input_batch = input_tensor.unsqueeze(0) 
            
            # Pass the image through the model to get features
            output = model(input_batch)
            
            # The output shape is [1, 512, 1, 1]. 
            # squeeze() removes dimensions of size 1 to make it a flat 1D vector of 512 numbers.
            feature_vector = output.squeeze().numpy()
            
            features_list.append(feature_vector)
            frame_names_list.append(filename)

    # 6. Save the results
    print(f"Saving {len(features_list)} feature vectors...")
    
    # Save the list of feature vectors as a fast-loading numpy array (.npy)
    np.save(features_path, np.array(features_list))
    
    # Save the filenames to a simple text file so we know which feature belongs to which frame
    with open(names_path, 'w') as f:
        for name in frame_names_list:
            f.write(f"{name}\n")
            
    print(f"Saved features to: {features_path}")
    print(f"Saved frame names to: {names_path}")
    print("Feature extraction complete!")

if __name__ == "__main__":
    # --- Configuration ---
    
    # Get the root directory of the project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Where to find the frames we extracted earlier
    input_frames_dir = os.path.join(base_dir, "outputs", "frames")
    
    # Where to save the new features
    output_features_dir = os.path.join(base_dir, "outputs", "features")
    
    # --- Execution ---
    print("--- SceneSeek: Feature Extraction ---")
    extract_features(input_frames_dir, output_features_dir)

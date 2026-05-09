import cv2
import os

def extract_frames(video_path, output_dir, num_frames=5):
    """
    Extracts a specific number of frames from a video at equal intervals.
    
    Args:
        video_path (str): Path to the input video file (.mp4).
        output_dir (str): Directory where the extracted frames will be saved.
        num_frames (int): The number of frames to extract from the video.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensuring output directory exists: {output_dir}")

    # 2. Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        print("Please check if the file exists and is a valid video format.")
        return

    # 3. Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    if total_frames == 0:
        print("Error: The video seems to be empty.")
        cap.release()
        return

    # 4. Calculate the interval between frames to extract
    # This ensures we get frames spread evenly throughout the video
    interval = max(1, total_frames // num_frames)
    
    # 5. Extract and save the frames
    frames_extracted = 0
    
    for i in range(num_frames):
        # Calculate the frame index we want to capture
        frame_index = i * interval
        
        # Ensure we don't go past the end of the video
        if frame_index >= total_frames:
            break
            
        # Set the video position to the target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        success, frame = cap.read()
        
        if success:
            # Construct the filename, e.g., frame_001.jpg
            frame_filename = os.path.join(output_dir, f"frame_{frames_extracted:03d}.jpg")
            
            # Save the frame as a .jpg image
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            
            frames_extracted += 1
        else:
            print(f"Warning: Failed to extract frame at index {frame_index}")

    # 6. Release the video object to free up resources
    cap.release()
    print(f"Successfully extracted {frames_extracted} frames to {output_dir}")

if __name__ == "__main__":
    # --- Configuration ---
    # These variables make it easy to adjust the script's behavior.
    
    # Get the root directory of the project (two levels up from this script)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Input video path (Placeholder: replace with an actual video later)
    # Assuming video will be placed in a 'data' folder
    input_video_path = os.path.join(base_dir, "data", "reference", "avatar", "clip1.mp4")
    
    # Output directory for the extracted frames
    output_frames_dir = os.path.join(base_dir, "outputs", "frames")
    
    # Number of frames to extract
    target_num_frames = 5
    
    # --- Execution ---
    print("--- SceneSeek: Frame Extraction ---")
    extract_frames(input_video_path, output_frames_dir, num_frames=target_num_frames)

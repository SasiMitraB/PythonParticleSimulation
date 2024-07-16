import cv2
import os
from matplotlib import pyplot as plt


def generate_animation(folder_name, output_video_name='output_video.mp4', fps=24):
    """
    Generate an animated video from a sequence of PNG images in a specified folder.

    Parameters:
    - folder_name (str): The name of the folder containing PNG images named in sequential order (1.png, 2.png, ..., n.png).
    - output_video_name (str): The name of the output animated video file. Default is 'output_video.mp4'.
    - fps (int): Frames per second for the output video. Default is 24.

    Returns:
    None

    Raises:
    - FileNotFoundError: If no PNG image files are found in the specified folder.

    Example:
    >>> generate_animation('images_folder', output_video_name='output_animation.mp4', fps=30)

    This function reads PNG images from the specified folder, stitches them together in sequential order, and generates
    an animated video in the MP4 format. The resulting video is saved with the specified output file name.

    Note: Ensure that the images are named in sequential order (1.png, 2.png, ..., n.png) and are in the PNG format.

    """
    # Get a list of image files in the folder
    image_files = sorted([f for f in os.listdir(folder_name) if f.endswith('.png')])

    # Check if there are any image files
    if not image_files:
        raise FileNotFoundError("No PNG image files found in the folder.")

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(folder_name, image_files[0]))
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    # Iterate through images and write to video
    for image_file in image_files:
        image_path = os.path.join(folder_name, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print(f"Video generated: {output_video_name}")



def hello_world():
    print("Hello World!")
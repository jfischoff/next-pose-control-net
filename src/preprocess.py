
from PIL import Image
import ffmpeg
import os
from pathlib import Path
import argparse

def video_to_frames(input_video, 
                    output_dir, 
                    height, 
                    width=None, 
                    x_offset=0, 
                    y_offset=0):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get the original video's width and height
    probe = ffmpeg.probe(input_video)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    orig_width = int(video_info['width'])
    orig_height = int(video_info['height'])

    # compute the new width given the height
    aspect_ratio = orig_width / orig_height
    scaled_width = int(height * aspect_ratio)

    in_file = ffmpeg.input(input_video)
    out_file = f"{output_dir}/%010d.png"

    video = in_file.filter('scale', scaled_width, height)

    # If the width is specified, crop the video
    if width is not None:
        video = video.crop(x_offset, y_offset, width, height)

    print("video: ", video)

    video.output(out_file).run(overwrite_output=True)

# Image merging code I've used before, but it must be extended for more channels.
# 
# def merge_images(root_folder):
#   initial_images_folder = os.path.join(root_folder, "initial_images_folder")
#   next_outline_images_folder = os.path.join(root_folder, "next_outline_images_folder")
# 
#   # Iterate through the PNG files in the initial_images_folder
#   for filename in os.listdir(initial_images_folder):
#       if filename.endswith(".png"):
#           # Open the initial image and the outline image
#           initial_image_path = os.path.join(initial_images_folder, filename)
#           outline_image_path = os.path.join(next_outline_images_folder, filename)
#           initial_image = Image.open(initial_image_path).convert("RGBA")
#           outline_image = Image.open(outline_image_path).convert("L")
# 
#           initial_image.putalpha(outline_image)
# 
#           initial_image.save(initial_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to frames with optional cropping")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_dir", help="Directory to save the output frames")
    parser.add_argument("height", type=int, help="Height for uniformly scaling the frames")
    parser.add_argument("--width", type=int, help="Width for cropping (optional)")
    parser.add_argument("--x_offset", type=int, default=0, help="X offset for cropping (optional)")
    parser.add_argument("--y_offset", type=int, default=0, help="Y offset for cropping (optional)")

    args = parser.parse_args()
    print("args: ", args)

    video_to_frames(args.input_video, args.output_dir, args.height, args.width, args.x_offset, args.y_offset)
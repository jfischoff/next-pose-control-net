
from PIL import Image
import ffmpeg
from pathlib import Path
import argparse
import controlnet_hinter

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
def merge_images(image1_path, image2_path, output_path):
    # Open the input images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Ensure both images have the same dimensions
    if image1.size != image2.size:
        print("Error: Images should have the same dimensions.")
        return

    # Split the input images into their RGB channels
    image1_r, image1_g, image1_b = image1.split()
    image2_r, image2_g, image2_b = image2.split()

    # Create a new image with 6 channels (3 from image1 and 3 from image2)
    merged_image = Image.merge("RGBRGB", (image1_r, image1_g, image1_b, image2_r, image2_g, image2_b))

    # Save the merged 6-channel image
    merged_image.save(output_path)

def create_openpose_image(input_path, output_path):
    pose_image = controlnet_hinter.hint_openpose(Image.open(input_path))
    pose_image.save(output_path)

# A function for iterating through a folder of pngs and creating openpose images
def create_openpose_images(input_dir, 
                           output_dir, 
                           should_skip_existing=True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for path in Path(input_dir).iterdir():
        if path.is_file() and path.suffix == ".png":
            output_path = f"{output_dir}/{path.name}"

            if should_skip_existing and Path(output_path).is_file():
                continue
            
            create_openpose_image(path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image and video processing tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for the extract-frames command
    parser_extract_frames = subparsers.add_parser("extract-frames", help="Convert video to frames with optional cropping")
    parser_extract_frames.add_argument("input_video", help="Path to the input video file")
    parser_extract_frames.add_argument("output_dir", help="Directory to save the output frames")
    parser_extract_frames.add_argument("height", type=int, help="Height for uniformly scaling the frames")
    parser_extract_frames.add_argument("--width", type=int, help="Width for cropping (optional)")
    parser_extract_frames.add_argument("--x_offset", type=int, default=0, help="X offset for cropping (optional)")
    parser_extract_frames.add_argument("--y_offset", type=int, default=0, help="Y offset for cropping (optional)")

    # Subparser for the merge-images command
    parser_merge_images = subparsers.add_parser("merge-images", help="Merge RGB channels from two images into a 6-channel image")
    parser_merge_images.add_argument("image1_path", help="Path to the first input image")
    parser_merge_images.add_argument("image2_path", help="Path to the second input image")
    parser_merge_images.add_argument("output_path", help="Path to save the output 6-channel image")

    # Subparser for the create-openpose-image command
    parser_create_openpose_image = subparsers.add_parser("create-openpose-image", help="Create an OpenPose image from a single input image")
    parser_create_openpose_image.add_argument("input_path", help="Path to the input image")
    parser_create_openpose_image.add_argument("output_path", help="Path to save the output OpenPose image")

    # Subparser for the create-openpose-images command
    parser_create_openpose_images = subparsers.add_parser("create-openpose-images", help="Create OpenPose images for a folder of input images")
    parser_create_openpose_images.add_argument("input_dir", help="Path to the input directory containing images")
    parser_create_openpose_images.add_argument("output_dir", help="Directory to save the output OpenPose images")
    parser_create_openpose_images.add_argument("--do_not_skip_existing", action="store_false", default=True, help="Do not skip creating OpenPose images for existing output files")



    args = parser.parse_args()
    print("args: ", args)

    if args.command == "extract-frames":
        video_to_frames(args.input_video, args.output_dir, args.height, args.width, args.x_offset, args.y_offset)
    elif args.command == "merge-images":
        merge_images(args.image1_path, args.image2_path, args.output_path)
    elif args.command == "create-openpose-image":
        create_openpose_image(args.input_path, args.output_path)
    elif args.command == "create-openpose-images":
        create_openpose_images(args.input_dir, args.output_dir, args.do_not_skip_existing)
    else:
        parser.print_help()
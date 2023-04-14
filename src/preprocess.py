
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

def create_openpose_image(input_path, output_path):
    source_image = Image.open(input_path)
    pose_image = controlnet_hinter.hint_openpose(source_image)
    # resize the image to 
    pose_image = pose_image.resize(source_image.size)

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
    elif args.command == "create-openpose-image":
        create_openpose_image(args.input_path, args.output_path)
    elif args.command == "create-openpose-images":
        create_openpose_images(args.input_dir, args.output_dir, args.do_not_skip_existing)
    else:
        parser.print_help()
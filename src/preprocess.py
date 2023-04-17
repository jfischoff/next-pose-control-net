
from PIL import Image
import ffmpeg
from pathlib import Path
import os
import argparse
import controlnet_hinter
import jsonlines

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


def get_captioning_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    captioning_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
    )
    captioning_model = captioning_model.to(device)
    return processor, captioning_model

def generate_captions(images, captioner=None):
    """Generates captions for a batch of images.

    Args:
        images: A batch of images in the RGB format.

    Returns:
        A list of generated captions.
    """
    if captioner is None:
        processor, captioning_model = get_captioning_model()
    else:
        processor, captioning_model = captioner
        
    inputs = processor(images=images, return_tensors="pt").to(captioning_model.device, torch.float16)

    generated_ids = captioning_model.generate(**inputs)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    generated_texts = [text.strip() for text in generated_texts]
    return generated_texts

IMAGE_FILE_EXTENSIONS = ['.png', '.jpg']

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

def crop_image(input_path,
               output_path,
               height,
               width=None,
               x_offset=0,
               y_offset=0):


    image = Image.open(input_path)
    orig_width, orig_height = image.size
    orig_aspect_ratio = orig_width / orig_height
    scaled_width = int(height * orig_aspect_ratio)
    

    image = image.resize((scaled_width, height), Image.Resampling.BICUBIC)
    

    if width is not None:
        image = image.crop((x_offset, y_offset, x_offset + width, y_offset + height))

    image.save(output_path)
    
def crop_frames(input_dir, 
                    output_dir, 
                    height, 
                    width=None, 
                    x_offset=0, 
                    y_offset=0,
                    should_skip_existing=True):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for path in Path(input_dir).iterdir():
        if path.is_file() and path.suffix == ".png":
            output_path = f"{output_dir}/{path.name}"

            if should_skip_existing and Path(output_path).is_file():
                continue
            
            crop_image(path, output_path, height, width, x_offset, y_offset)



def create_jsonl(image_input_dir,
                 pose_input_dir,
                 output_file,
                 text="A photo of a person dancing",
                 use_captioning_model=False):

    if use_captioning_model:
        captioner = get_captioning_model()
    
    with jsonlines.open(output_file, 'w') as writer:  #
        input_images = []
        for path in Path(image_input_dir).iterdir():
            if path.is_file() and path.suffix in IMAGE_FILE_EXTENSIONS:
                input_images.append(path)
        input_images.sort()
        for prev_image, curr_image in zip(input_images[:-1], input_images[1:]):
            _, tail = os.path.split(curr_image)
            pose_image = os.path.join(pose_input_dir, tail)
            
            if use_captioning_model:
                text = generate_captions(Image.open(curr_image), captioner=captioner)
                
            writer.write( { 'text': text,
                            'image': str(curr_image),
                            'conditioning_image': str(prev_image),
                            'pose_conditioning_image': str(pose_image),
                        })
            
    
                   
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
        if path.is_file() and path.suffix in IMAGE_FILE_EXTENSIONS:
            output_path = f"{output_dir}/{path.name}"

            if should_skip_existing and Path(output_path).is_file():
                continue
            
            create_openpose_image(path, output_path)


def runner(input_video, 
           frames_dir,
           openpose_dir,
           train_jsonl_file,
           width,
           height,
           x_offset,
           y_offset,
            ):
    # Extract frames from the video
    video_to_frames(input_video, frames_dir, height, width, x_offset, y_offset)

    # Create openpose images from the frames
    create_openpose_images(frames_dir, openpose_dir)

    # Create the jsonl file
    create_jsonl(frames_dir, openpose_dir, train_jsonl_file)                              
    

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

    parser_crop_frames = subparsers.add_parser("crop-frames", help="Crop directory of images")
    parser_crop_frames.add_argument("input_dir", help="Path to the input video file")
    parser_crop_frames.add_argument("output_dir", help="Directory to save the output frames")
    parser_crop_frames.add_argument("height", type=int, help="Height for uniformly scaling the frames")
    parser_crop_frames.add_argument("--width", type=int, help="Width for cropping (optional)")
    parser_crop_frames.add_argument("--x_offset", type=int, default=0, help="X offset for cropping (optional)")
    parser_crop_frames.add_argument("--y_offset", type=int, default=0, help="Y offset for cropping (optional)")
    parser_crop_frames.add_argument("--do_not_skip_existing", action="store_false", default=True, help="Do not skip creating OpenPose images for existing output files")

    
    # Subparser for the create-openpose-image command
    parser_create_openpose_image = subparsers.add_parser("create-openpose-image", help="Create an OpenPose image from a single input image")
    parser_create_openpose_image.add_argument("input_path", help="Path to the input image")
    parser_create_openpose_image.add_argument("output_path", help="Path to save the output OpenPose image")

    # Subparser for the create-openpose-images command
    parser_create_openpose_images = subparsers.add_parser("create-openpose-images", help="Create OpenPose images for a folder of input images")
    parser_create_openpose_images.add_argument("input_dir", help="Path to the input directory containing images")
    parser_create_openpose_images.add_argument("output_dir", help="Directory to save the output OpenPose images")
    parser_create_openpose_images.add_argument("--do_not_skip_existing", action="store_false", default=True, help="Do not skip creating OpenPose images for existing output files")


    parser_create_jsonl = subparsers.add_parser("create-jsonl", help="Create jsonl for a folder of input images and a folder of OpenPose images")
    parser_create_jsonl.add_argument("image_input_dir", help="Path to the input directory containing images")
    parser_create_jsonl.add_argument("pose_input_dir", help="Path to the input directory containing images")
    parser_create_jsonl.add_argument("--text", default="A photo of a person dancing", help="Description of images")
    parser_create_jsonl.add_argument("--use_captioning_model", action="store_true", help="Use captioning model to describe images")
    parser_create_jsonl.add_argument("output_file", help="Output filename")


    parser_run = subparsers.add_parser("run", help="Run the full preprocessing pipeline")
    parser_run.add_argument("input_video", help="Path to the input video file")
    parser_run.add_argument("frames_dir", help="Directory to save the output frames")
    parser_run.add_argument("openpose_dir", help="Directory to save the output OpenPose images")
    parser_run.add_argument("train_jsonl_file", help="Output filename")
    parser_run.add_argument("width", type=int, help="Width for cropping (optional)")
    parser_run.add_argument("height", type=int, help="Height for uniformly scaling the frames")
    parser_run.add_argument("x_offset", type=int, default=0, help="X offset for cropping (optional)")
    parser_run.add_argument("y_offset", type=int, default=0, help="Y offset for cropping (optional)")
    

    args = parser.parse_args()
    print("args: ", args)

    if args.command == "extract-frames":
        video_to_frames(args.input_video, args.output_dir, args.height, args.width, args.x_offset, args.y_offset)
    elif args.command == "create-openpose-image":
        create_openpose_image(args.input_path, args.output_path)
    elif args.command == "create-openpose-images":
        create_openpose_images(args.input_dir, args.output_dir, args.do_not_skip_existing)
    elif args.command == "create-jsonl":
        create_jsonl(args.image_input_dir, args.pose_input_dir, args.output_file, args.text, args.use_captioning_model)
    elif args.command == "crop-frames":
        crop_frames(args.input_dir, args.output_dir, args.height, args.width, args.x_offset, args.y_offset, args.do_not_skip_existing)
    elif args.command == "run":
        runner(args.input_video, args.frames_dir, args.openpose_dir, args.train_jsonl_file, args.width, args.height, args.x_offset, args.y_offset)
    else:
        parser.print_help()

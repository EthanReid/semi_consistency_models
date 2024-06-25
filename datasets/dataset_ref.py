import os
import numpy as np
from PIL import Image
import argparse
from multiprocessing import Pool

def process_image(args):
    image_path, image_size = args
    with Image.open(image_path) as img:
        img = img.resize((image_size, image_size))
        return np.array(img)

def load_and_resize_images(image_dir, num_images, image_size, num_workers):
    image_files = os.listdir(image_dir)[:num_images]
    image_paths = [os.path.join(image_dir, image_file) for image_file in image_files]
    pool_args = [(image_path, image_size) for image_path in image_paths]
    
    with Pool(num_workers) as pool:
        image_list = pool.map(process_image, pool_args)
    
    return np.array(image_list)

def save_images_as_npz(images, output_file):
    np.savez(output_file, images=images)

def main(image_dir, num_images, image_size, output_file, num_workers):
    images = load_and_resize_images(image_dir, num_images, image_size, num_workers)
    save_images_as_npz(images, output_file)
    print(f"Saved {num_images} images to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and save LSUN images as NPZ file")
    parser.add_argument("--image_dir", type=str, help="Directory containing LSUN images")
    parser.add_argument("--num_images", type=int, help="Number of images to process")
    parser.add_argument("--image_size", type=int, help="Size to resize images to (DxD)")
    parser.add_argument("--output_file", type=str, help="Output NPZ file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    main(args.image_dir, args.num_images, args.image_size, args.output_file, args.num_workers)
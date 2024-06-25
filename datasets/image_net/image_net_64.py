import os
import pickle
import argparse
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Function to save a single image as PNG
def save_image_as_png(args):
    image_array, label, filename, output_dir, batch_index, i = args
    image = Image.fromarray(image_array)
    label_dir = os.path.join(output_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)
    image.save(os.path.join(label_dir, f"{filename}_{batch_index}_{i}.png"))

# Function to process a single batch file
def process_batch(batch_file, output_dir, batch_index, num_workers):
    data_batch = unpickle(batch_file)
    
    # Debugging: print the keys of the unpickled dictionary
    print(f"Keys in the batch file '{batch_file}': {list(data_batch.keys())}")

    data = data_batch['data']
    labels = data_batch['labels']

    data = data.reshape(-1, 3, 64, 64)
    data = data.transpose(0, 2, 3, 1)  # Convert to HWC

    # Generate filenames
    filenames = [f'image_{batch_index}_{i}.png' for i in range(len(data))]

    args = [(data[i], labels[i], filenames[i], output_dir, batch_index, i) for i in range(len(data))]
    with Pool(num_workers) as p:
        p.map(save_image_as_png, args)

def main(input_dir, output_dir, num_workers):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load and save images for each batch using multiple workers
    for batch_index in range(1, 6):
        batch_file = os.path.join(input_dir, f'train_data_batch_{batch_index}')
        process_batch(batch_file, output_dir, batch_index, num_workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Imagenet64 data batches to PNG images.')
    parser.add_argument('--input_dir', type=str, help='Input directory containing the data batch files.')
    parser.add_argument('--output_dir', type=str, help='Output directory to save the PNG images.')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='Number of workers to use for parallel processing.')

    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_workers = args.num_workers

    if input_dir == '.':
        input_dir = os.getcwd()

    main(input_dir, output_dir, num_workers)
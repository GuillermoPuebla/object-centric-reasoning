import os
import csv
import cv2
import numpy as np
from PIL import Image
import random
import itertools


def check_path(path):
    """Method for creating directory if it doesn't exist yet."""
    if not os.path.exists(path):
        os.mkdir(path)

def make_SOSD_sample(top_path, bottom_path):
    """"Makes a sample of second order same-different (SOSD)."""
    # Coordinates
    left1, top1 = 32, 0
    left2, top2 = 32, 64
    # Make full image
    top_img = Image.open(top_path)
    bottom_img = Image.open(bottom_path)
    full_img = Image.new(top_img.mode, (128, 128), (255, 255, 255))
    full_img.paste(top_img, (left1, top1))
    full_img.paste(bottom_img, (left2, top2))
    return full_img

def get_random_triplet_for_SOSD(main_base, base_index, start=0, stop=35000, data_dir='data/svrt_task1_64x64'):
    """Get a 'random' triplet of images to form two samples of SOSD.
    Args:
        main_base: 0 (different) or 1 (same).
        base_index: integer from 0 to 34999 that indicates the base image id.
        data_dir: data directory path.
    """
    non_match_base = 1 if main_base == 0 else 0
    match_samples = list(range(start, stop))
    match_samples.remove(base_index)
    main_numbers = (base_index, random.sample(match_samples, 1)[0])
    a = str(main_numbers[0]).zfill(4) if main_numbers[0] < 9999 else str(main_numbers[0]).zfill(4)
    b = str(main_numbers[1]).zfill(4) if main_numbers[1] < 9999 else str(main_numbers[1]).zfill(4)
    c = random.randint(start, stop-1)
    c = str(c).zfill(4) if c < 9999 else str(c).zfill(4)
    
    main_path = f'{data_dir}/sample_{main_base}_{a}.png'
    match_path = f'{data_dir}/sample_{main_base}_{b}.png'
    non_match_path = f'{data_dir}/sample_{non_match_base}_{c}.png'
    
    return main_path, match_path, non_match_path

def make_svrt1_sosd_dataset():
    """Total: 70_000. 
    Train (70%): 49_000, val (10%): 7_000, test (20%): 14_000"""
    # Make folders
    root_dir = f'data/svrt1_sosd'
    check_path(root_dir)
    train_dir = f'data/svrt1_sosd/train'
    check_path(train_dir)
    val_dir = f'data/svrt1_sosd/val'
    check_path(val_dir)
    test_dir = f'data/svrt1_sosd/test'
    check_path(test_dir)

    # Train data
    ds_train_file = f"{root_dir}/train_annotations.csv"
    ds_train_file_header = ['ID', 'top label', 'bottom label', 'label']
    
    # Get permutations without repetition
    train_size = 49_000
    val_size = 7_000
    test_size = 14_000
    
    # Open the file in write mode
    with open(ds_train_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(ds_train_file_header)
        
        # Get 49000*2 train images
        counter = 0
        for top_label in range(2):
            for j in range(24500): # 49000/2
                # Get all 64x64 images
                main_path, match_path, non_match_path = get_random_triplet_for_SOSD(main_base=top_label, base_index=j, start=0, stop=24500)
                
                # Save second order 'same' image
                img_so_s = make_SOSD_sample(top_path=main_path, bottom_path=match_path)
                bottom_label = top_label
                row = [counter, top_label, bottom_label, 1]
                writer.writerow(row)
                img_so_s.save(f'{train_dir}/{counter}.png')
                counter += 1
                
                # Save second order 'different' image
                img_so_d = make_SOSD_sample(top_path=main_path, bottom_path=non_match_path)
                bottom_label = 1 if top_label == 0 else 0
                row = [counter, top_label, bottom_label, 0]
                writer.writerow(row)
                img_so_d.save(f'{train_dir}/{counter}.png')
                counter += 1
        
    print('total train images: ', counter)

    # Validation data
    ds_val_file = f"{root_dir}/val_annotations.csv"
    ds_val_file_header = ['ID', 'top label', 'bottom label', 'label']
    # Open the file in write mode
    with open(ds_val_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(ds_val_file_header)
        # Get 7000 validation images
        counter = 0
        for top_label in range(2):
            for j in range(24500, 28000): # 7000/2 = 3500
                # Get all 64x64 images
                main_path, match_path, non_match_path = get_random_triplet_for_SOSD(main_base=top_label, base_index=j, start=24500, stop=28000)
                
                # Save second order 'same' image
                img_so_s = make_SOSD_sample(top_path=main_path, bottom_path=match_path)
                bottom_label = top_label
                row = [counter, top_label, bottom_label, 1]
                writer.writerow(row)
                img_so_s.save(f'{val_dir}/{counter}.png')
                counter += 1
                
                # Save second order 'different' image
                img_so_d = make_SOSD_sample(top_path=main_path, bottom_path=non_match_path)
                bottom_label = 1 if top_label == 0 else 0
                row = [counter, top_label, bottom_label, 0]
                writer.writerow(row)
                img_so_d.save(f'{val_dir}/{counter}.png')
                counter += 1

    print('total validation images: ', counter)
    
    # Test data
    ds_test_file = f"{root_dir}/test_annotations.csv"
    ds_test_file_header = ['ID', 'top label', 'bottom label', 'label']
    # Open the file in write mode
    with open(ds_test_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(ds_test_file_header)
        # Get 14000 test images
        counter = 0
        for top_label in range(2):
            for j in range(28000, 35000): # 14000/2 = 7000
                # Get all 64x64 images
                main_path, match_path, non_match_path = get_random_triplet_for_SOSD(main_base=top_label, base_index=j, start=28000, stop=35000)
                
                # Save second order 'same' image
                img_so_s = make_SOSD_sample(top_path=main_path, bottom_path=match_path)
                bottom_label = top_label
                row = [counter, top_label, bottom_label, 1]
                writer.writerow(row)
                img_so_s.save(f'{test_dir}/{counter}.png')
                counter += 1
                
                # Save second order 'different' image
                img_so_d = make_SOSD_sample(top_path=main_path, bottom_path=non_match_path)
                bottom_label = 1 if top_label == 0 else 0
                row = [counter, top_label, bottom_label, 0]
                writer.writerow(row)
                img_so_d.save(f'{test_dir}/{counter}.png')
                counter += 1

    print('total test images: ', counter)
    
    return


if __name__ == '__main__':
    make_svrt1_sosd_dataset()
    
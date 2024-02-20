import os
import csv
import cv2
import numpy as np
from PIL import Image
import random


def check_path(path):
    """Method for creating directory if it doesn't exist yet."""
    if not os.path.exists(path):
        os.mkdir(path)

def get_individual_objects_translated(img):
    """Returns individual objects in separated images"""
    # Bad sample flag
    bad_sample = False
    # Find contours
    image1 = img.copy()
    image2 = img.copy()
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # this is to deal with different opencv versions
    # If there is a single contour return with flag
    if len(cnts) == 1:
        bad_sample = True
        return Image.fromarray(image1), Image.fromarray(image2), bad_sample
    else:
        # Delete objects
        cv2.drawContours(image1, cnts, 0, color=[255,255,255], thickness=-1)
        cv2.drawContours(image2, cnts, 1, color=[255,255,255], thickness=-1)
        return Image.fromarray(image1), Image.fromarray(image2), bad_sample

def make_RMS_sample(base_path, left_path, right_path):
    # Coordinates
    left1, top1 = 32, 0
    left2, top2 = 0, 64
    left3, top3 = 64, 64
    # Make full image
    base_img = Image.open(base_path)
    left_img = Image.open(left_path)
    right_img = Image.open(right_path)
    full_img = Image.new(base_img.mode, (128, 128), (255, 255, 255))
    full_img.paste(base_img, (left1, top1))
    full_img.paste(left_img, (left2, top2))
    full_img.paste(right_img, (left3, top3))
    # Make individual images
    base1_small, base2_small, _ = get_individual_objects_translated(np.array(base_img))
    base_1 = Image.new(base1_small.mode, (128, 128), (255, 255, 255))
    base_2 = Image.new(base2_small.mode, (128, 128), (255, 255, 255))
    # print(type(base1_small), type(base_1))
    # print(base1_small.size, base_1.size)
    base_1.paste(base1_small, (left1, top1))
    base_2.paste(base2_small, (left1, top1))

    left1_small, left2_small, _ = get_individual_objects_translated(np.array(left_img))
    left_1 = Image.new(left1_small.mode, (128, 128), (255, 255, 255))
    left_2 = Image.new(left2_small.mode, (128, 128), (255, 255, 255))
    # print(type(left1_small), type(left1))
    # print(left1_small.size, left1.size)
    left_1.paste(left1_small, (left2, top2))
    left_2.paste(left2_small, (left2, top2))

    right1_small, right2_small, _ = get_individual_objects_translated(np.array(right_img))
    right_1 = Image.new(right1_small.mode, (128, 128), (255, 255, 255))
    right_2 = Image.new(right2_small.mode, (128, 128), (255, 255, 255))
    right_1.paste(right1_small, (left3, top3))
    right_2.paste(right2_small, (left3, top3))

    return full_img, [base_1, base_2, left_1, left_2, right_1, right_2]

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
    
def get_random_triplets(main_base, base_index, start=0, stop=35000, data_dir='data/svrt_task1_64x64'):
    """
    Args:
        main_base: 0 (different) or 1 (same).
        base_index: integer from 0 to 34999 that indicates the base image id.
        data_dir: data directory path.
    """
    non_match_base = 1 if main_base == 0 else 0
    match_candidates = list(range(start, stop))
    match_candidates.remove(base_index)
    match_samples = random.sample(match_candidates, 2)
    a = str(base_index).zfill(4) if base_index <= 9999 else str(base_index)
    b1 = str(match_samples[0]).zfill(4) if match_samples[0] <= 9999 else str(match_samples[0])
    b2 = str(match_samples[1]).zfill(4) if match_samples[1] <= 9999 else str(match_samples[1])
    
    non_match_candidates = list(range(start, stop))
    non_match_samples = random.sample(non_match_candidates, 2)
    c1 = str(non_match_samples[0]).zfill(4) if non_match_samples[0] <= 9999 else str(non_match_samples[0])
    c2 = str(non_match_samples[1]).zfill(4) if non_match_samples[1] <= 9999 else str(non_match_samples[1])
    
    main_path = f'{data_dir}/sample_{main_base}_{a}.png'
    match_1 = f'{data_dir}/sample_{main_base}_{b1}.png'
    match_2 = f'{data_dir}/sample_{main_base}_{b2}.png'
    non_match_1 = f'{data_dir}/sample_{non_match_base}_{c1}.png'
    non_match_2 = f'{data_dir}/sample_{non_match_base}_{c2}.png'
    
    return main_path, match_1, match_2, non_match_1, non_match_2

def make_svrt1_rmts_dataset_big():
    """Total: 70_000. Train (70%): 49000, val (10%): 7000, test (20%): 14000"""
    # Make folders
    root_dir = f'data/svrt1_rmts'
    check_path(root_dir)
    train_dir = f'data/svrt1_rmts/train'
    check_path(train_dir)
    val_dir = f'data/svrt1_rmts/val'
    check_path(val_dir)
    test_dir = f'data/svrt1_rmts/test'
    check_path(test_dir)

    # Train data
    ds_train_file = f"{root_dir}/train_annotations.csv"
    ds_train_file_header = ['ID', 'base', 'label']
    # Open the file in write mode
    with open(ds_train_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(ds_train_file_header)
        # Get 49000 train images
        counter = 0
        for base in range(2):
            for j in range(24500): # 49000/2
                # First triplet
                main_path, match_1, match_2, non_match_1, non_match_2 = get_random_triplets(main_base=base, base_index=j, start=0, stop=24500)
                label = 0
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=match_1, right_path=non_match_1)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{train_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
                # Invert left/right assignment
                label = 1
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=non_match_1, right_path=match_1)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{train_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
                
                # Second triplet
                label = 0
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=match_2, right_path=non_match_2)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{train_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
                # Invert left/right assignment
                label = 1
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=non_match_2, right_path=match_2)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{train_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1            
    print(counter)
    
    # Validation data
    ds_val_file = f"{root_dir}/val_annotations.csv"
    ds_val_file_header = ['ID', 'base', 'label']
    # Open the file in write mode
    with open(ds_val_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(ds_val_file_header)
        # Get 7000 validation images
        counter = 0
        for base in range(2):
            for j in range(24500, 28000): # 7000/2 = 3500
                # First triplet
                main_path, match_1, match_2, non_match_1, non_match_2 = get_random_triplets(main_base=base, base_index=j, start=24500, stop=28000)
                label = 0
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=match_1, right_path=non_match_1)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{val_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
                # Invert left/right assignment
                label = 1
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=non_match_1, right_path=match_1)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{val_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
                
                # Second triplet
                label = 0
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=match_2, right_path=non_match_2)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{val_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
                # Invert left/right assignment
                label = 1
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=non_match_2, right_path=match_2)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{val_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
    print(counter)
    
    # Test data
    ds_test_file = f"{root_dir}/test_annotations.csv"
    ds_test_file_header = ['ID', 'base', 'label']
    # Open the file in write mode
    with open(ds_test_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(ds_test_file_header)
        # Get 14000 test images
        counter = 0
        for base in range(2):
            for j in range(28000, 35000): # 14000/2 = 7000
                # First triplet
                main_path, match_1, match_2, non_match_1, non_match_2 = get_random_triplets(main_base=base, base_index=j, start=28000, stop=35000)
                label = 0
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=match_1, right_path=non_match_1)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{test_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
                # Invert left/right assignment
                label = 1
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=non_match_1, right_path=match_1)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{test_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
                
                # Second triplet
                label = 0
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=match_2, right_path=non_match_2)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{test_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
                # Invert left/right assignment
                label = 1
                full_img, image_list = make_RMS_sample(base_path=main_path, left_path=non_match_2, right_path=match_2)
                row = [counter, base, label]
                # Save data
                full_img.save(f'{test_dir}/{counter}.png')
                writer.writerow(row)
                counter += 1
    print(counter)
    
    return


if __name__ == '__main__':
    make_svrt1_rmts_dataset_big()
    
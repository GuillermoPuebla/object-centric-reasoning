# Imports
import os
import csv
import pandas as pd
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from project_utils.dataloaders import SD_dataset, check_path
from models.RViT import transformer_econvviut_hires_multiloss_medium, accuracy


# Test datasets
def get_ds_names_and_loaders(batch_size, normalization='samplewise3'):
    """Returns names and dataloaders of new datastets for testing"""
    names_and_loaders = []

    # SVRT1 translated
    ds = SD_dataset(
        root_dir='data/svrt1_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('SVRT1', loader))

    # Irregular translated
    ds = SD_dataset(
        root_dir='data/irregular_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Irregular', loader))

    # Regular translated
    ds = SD_dataset(
        root_dir='data/regular_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Regular', loader))

    # Open translated
    ds = SD_dataset(
        root_dir='data/open_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Open', loader))

    # Wider translated
    ds = SD_dataset(
        root_dir='data/wider_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Wider', loader))

    # Scrambled translated
    ds = SD_dataset(
        root_dir='data/scrambled_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Scrambled', loader))

    # Random translated
    ds = SD_dataset(
        root_dir='data/random_sd',
        split='test',
        normalization='randomcolorsimple'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Random', loader)) # Scrambled.1

    # Filled translated
    ds = SD_dataset(
        root_dir='data/filled_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Filled', loader))

    # Lines translated
    ds = SD_dataset(
        root_dir='data/lines_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Lines', loader))

    # Arrows translated
    ds = SD_dataset(
        root_dir='data/arrows_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Arrows', loader))

    # Rectangles translated
    ds = SD_dataset(
        root_dir='data/rectangles_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Rectangles', loader))

    # Straight lines translated
    ds = SD_dataset(
        root_dir='data/slines_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('S-lines', loader))

    # Connected squares translated
    ds = SD_dataset(
        root_dir='data/csquares_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-squares', loader))

    # Connected circles translated
    ds = SD_dataset(
        root_dir='data/ccircles_sd',
        split='test',
        normalization=normalization
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-circles', loader))

    return names_and_loaders

# Train function
def train_SD_RViT(
    run,
    model,
    model_name,
    device,
    optimizer,
    scheduler,
    epochs,
    train_loader,
    val_loader,
    current_dir
    ):
    # Create filename for saving training progress
    train_file = f"{current_dir}/model_{model_name}_run_{run}_train.csv"
    header = [
        'Model', 
        'Run', 
        'Epoch',
        'Loss',
        'Accuracy',
        'Val loss',
        'Val accuracy'
        ]
    # Open the file in write mode
    with open(train_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(header)
        # Iterate over epoches and batches
        best_val_loss = 1_000_000.0
        for epoch in range(1, epochs + 1):
            # Make sure gradient tracking is on
            model.train(True)
            running_train_acc = 0.0
            running_train_loss = 0.0
            for i, (x, target) in enumerate(train_loader):
                # Load data to device
                x = x.to(device)
                target = target.to(device)
                # Zero out gradients for optimizer 
                optimizer.zero_grad()
                # Run model 
                loss, output, n_updates = model(x, target)
                # Loss
                train_loss = loss.item()
                running_train_loss += train_loss
                # Accuracy
                train_acc = accuracy(output, target, topk=(1,))[0] * 100.0
                train_acc /= len(output)
                running_train_acc += train_acc
                # Update model
                loss.backward()
                optimizer.step()

            scheduler.step()

            avg_train_loss = running_train_loss / (i + 1)
            avg_train_acc = running_train_acc / (i + 1)

            # Validation accuracy
            model.train(False)
            running_val_acc = 0.0
            running_val_loss = 0.0
            for j, (x, target) in enumerate(val_loader):
                # Load data to device
                x = x.to(device)
                target = target.to(device)
                # Run model 
                loss, output, n_updates = model(x, target)
                # Loss
                val_loss = loss.item()
                running_val_loss += val_loss
                # Accuracy
                val_acc = accuracy(output, target, topk=(1,))[0] * 100.0
                val_acc /= len(output)
                running_val_acc += val_acc

            avg_val_loss = running_val_loss / (j + 1)
            avg_val_acc = running_val_acc / (j + 1)

            # Save info to file
            row = [
                model_name, # 'Model'
                run, # 'Run'
                epoch, # 'Epoch'
                avg_train_loss, # 'Loss'
                avg_train_acc, # 'Accuracy'
                avg_val_loss, # 'Val loss'
                avg_val_acc # 'Val accuracy'
                ]
            writer.writerow(row)

            # Report
            if epoch % 1 == 0:
                print('[Epoch: ' + str(epoch) + '] ' + \
                        '[Train loss = ' + '{:.4f}'.format(avg_train_loss) + '] ' + \
                        '[Val loss = ' + '{:.4f}'.format(avg_val_loss) + '] ' + \
                        '[Train acc = ' + '{:.2f}'.format(avg_train_acc) + '] ' + \
                        '[Val acc = ' + '{:.2f}'.format(avg_val_acc) + '] ')
            
            # Track best performance, and save the model's state
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path =  f"{current_dir}/model_{model_name}_run_{run}_best_epoch_weights.pt"
                torch.save(model.state_dict(), model_path)
            
            if avg_val_acc > 99.0:
                break

# Test function
def test_SD_RViT(
    run,
    model,
    model_name,
    device,
    ds_names_and_loaders,
    current_dir
    ):
    # Create filename for saving training progress
    test_file = f"{current_dir}/model_{model_name}_run_{run}_test.csv"
    header = ['Model', 'Run']
    for name, loader in ds_names_and_loaders:
        loss_name = f"Loss {name}"
        acc_name = f"Accuracy {name}"
        header.append(loss_name)
        header.append(acc_name)
    # Open the file in write mode
    with open(test_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(header)
        # Initialize row
        row = [model_name, run]
        # Iterate over datasets
        for name, data_loader in ds_names_and_loaders:
            running_test_acc = 0.0
            running_test_loss = 0.0
            for j, (x, target) in enumerate(data_loader):
                # Load data to device
                x = x.to(device)
                target = target.to(device)
                # Run model 
                loss, output, n_updates = model(x, target)
                # Loss
                test_loss = loss.item()
                running_test_loss += test_loss
                # Accuracy
                test_acc = accuracy(output, target, topk=(1,))[0] * 100.0
                test_acc /= len(output)
                running_test_acc += test_acc

            avg_test_loss = running_test_loss / (j + 1)
            avg_test_acc = running_test_acc / (j + 1)

            # Save info to row
            row.append(avg_test_loss)
            row.append(avg_test_acc)
        # Write all data to file
        writer.writerow(row)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # Limit number of  threads 
    torch.set_num_threads(10)
    # Training parameters
    EPOCHS = 200
    BATCH_SIZE = 64
    BATCH_SIZE_TEST = 64
    WORKERS = 8
    VIT_DEPTH = 0
    U_DEPHT = 9
    LR = 0.0001
    STEP_SIZE = 160
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.00001 # 0.0001

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUNS = 10
    
    # Train on SVRT-1
    current_dir = 'results/SD_RViT_new'
    check_path(current_dir)
            
    train_ds = SD_dataset(
        root_dir='data/svrt1_sd',
        split='train',
        normalization='samplewise3'
        )
    val_ds = SD_dataset(
        root_dir='data/svrt1_sd',
        split='val',
        normalization='samplewise3'
        )
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE,
        shuffle=True
        )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE_TEST,
        shuffle=False
        )
    test_names_loaders = get_ds_names_and_loaders(
            batch_size=BATCH_SIZE_TEST,
            normalization='samplewise3'
            )
    
    for i in range(1, RUNS+1):
        # Insantiate model
        model = transformer_econvviut_hires_multiloss_medium(
            pretrained=False,
            map_location=None,
            depth=VIT_DEPTH,
            u_depth=U_DEPHT
            )
        # Set to training mode 
        model.train()
        model.to(device)
        # Create optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer.zero_grad()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE)

        # Train
        train_SD_RViT(
            run=i,
            model=model,
            model_name='RViT',
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=EPOCHS,
            train_loader=train_loader,
            val_loader=val_loader,
            current_dir=current_dir
            )
        # Set to evaluation mode
        model.eval()
        # Test
        test_SD_RViT(
            run=i,
            model=model,
            model_name='RViT',
            device=device,
            ds_names_and_loaders=test_names_loaders,
            current_dir=current_dir
            )

    # Merge files into datasets
    test_df = []
    train_df = []
    for root, dirs, files in os.walk(current_dir, topdown=False):
        for name in files:
            path_to_file = os.path.join(root, name)
            if name.endswith('test.csv'):
                df = pd.read_csv(path_to_file)
                test_df.append(df)
            elif name.endswith('train.csv'):
                df = pd.read_csv(path_to_file)
                train_df.append(df)

    test_df = pd.concat(test_df)
    test_df_name = f'{current_dir}/RViT_test_SD.csv'
    test_df.to_csv(test_df_name)

    train_df = pd.concat(train_df)
    train_df_name = f'{current_dir}/RViT_train_SD.csv'
    train_df.to_csv(train_df_name)

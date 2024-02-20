# Imports
import os
import csv
import pandas as pd
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from project_utils.dataloaders import MTS_dataset, check_path
from models.CLIPViT import get_CLIP_ViT16


# Test datasets
def get_ds_names_and_loaders(batch_size, normalization='CLIPViT'):
    """Returns names and dataloaders of new datastets for testing"""
    names_and_loaders = []

    # SVRT1
    ds = MTS_dataset(
        root_dir='data/svrt1_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('SVRT1', loader))

    # Irregular translated
    ds = MTS_dataset(
        root_dir='data/irregular_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Irregular', loader))

    # Regular translated
    ds = MTS_dataset(
        root_dir='data/regular_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Regular', loader))

    # Open translated
    ds = MTS_dataset(
        root_dir='data/open_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Open', loader))

    # Wider translated
    ds = MTS_dataset(
        root_dir='data/wider_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Wider', loader))

    # Scrambled translated
    ds = MTS_dataset(
        root_dir='data/scrambled_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Scrambled', loader))

    # Random translated
    ds = MTS_dataset(
        root_dir='data/random_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Random', loader)) # Scrambled.1

    # Filled translated
    ds = MTS_dataset(
        root_dir='data/filled_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Filled', loader))

    # Lines translated
    ds = MTS_dataset(
        root_dir='data/lines_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Lines', loader))

    # Arrows translated
    ds = MTS_dataset(
        root_dir='data/arrows_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Arrows', loader))

    # Rectangles translated
    ds = MTS_dataset(
        root_dir='data/rectangles_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Rectangles', loader))

    # Straight lines translated
    ds = MTS_dataset(
        root_dir='data/slines_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('S-lines', loader))

    # Connected squares translated
    ds = MTS_dataset(
        root_dir='data/csquares_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-squares', loader))

    # Connected circles translated
    ds = MTS_dataset(
        root_dir='data/ccircles_mts',
        split='test',
        normalization=normalization,
        squeeze=True,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-circles', loader))

    return names_and_loaders

# Training function
def train_SD_CLIPViT(
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
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    # Set to training mode 
    model.train()
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
            # load batch from dataloader 
            for i, (x, y) in enumerate(train_loader):
                # Load data to device
                x = x.to(device)
                y = y.to(device)
                # Zero out gradients for optimizer 
                optimizer.zero_grad()
                # Run model 
                outputs = model(x)
                _, preds = torch.max(outputs, 1)

                # Update model
                train_loss = loss_fn(outputs, y)
                train_loss.backward()
                optimizer.step()

                # Loss
                running_train_loss += train_loss.item()
                
                # Accuracy
                running_train_acc += torch.sum(preds == y.data)
            
            avg_train_loss = running_train_loss / (i + 1)
            avg_train_acc = (running_train_acc.item() / len(train_loader.dataset)) * 100

            # Val accuracy
            model.train(False)
            running_val_acc = 0.0
            running_val_loss = 0.0
            for j, (x, y) in enumerate(val_loader):
                # Load data to device
                x = x.to(device)
                y = y.to(device)
                # Run model 
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                # Loss
                val_loss = loss_fn(outputs, y)
                running_val_loss += val_loss.item() 
                # Accuracy
                running_val_acc += torch.sum(preds == y.data)

            avg_val_loss = running_val_loss / (j + 1)
            avg_val_acc = (running_val_acc.item() / len(val_loader.dataset)) * 100

            scheduler.step(val_loss)
            
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
def test_SD_CLIPViT(
    run,
    model,
    model_name,
    device,
    ds_names_and_loaders,
    current_dir
    ):
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
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
        for name, test_loader in ds_names_and_loaders:
            running_test_acc = 0.0
            running_test_loss = 0.0
            for j, (x, y) in enumerate(test_loader):
                # Load data to device
                x = x.to(device)
                y = y.to(device)
                # Run model 
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                # Loss
                loss = loss_fn(outputs, y)
                running_test_loss += loss.item() 
                # Accuracy
                running_test_acc += torch.sum(preds == y.data)


            avg_test_loss = running_test_loss / (j + 1)
            avg_test_acc = (running_test_acc.item() / len(test_loader.dataset)) * 100
            
            # Save info to row
            row.append(avg_test_loss)
            row.append(avg_test_acc)
        # Write all data to file
        writer.writerow(row)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # Limit number of  threads 
    torch.set_num_threads(10)
    # SVRT-1 parameters
    EPOCHS = 100
    BATCH_SIZE = 64
    BATCH_SIZE_TEST = 64
    LR = 0.000001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUNS = 10
    
    # Train on SVRT-1
    current_dir = 'results/MTS_CLIPViT_LR_0_000001'
    check_path(current_dir)
    
    train_ds = MTS_dataset(
        root_dir='data/svrt1_mts',
        split='train',
        normalization='CLIPViT',
        squeeze=True,
        invert=False,
        data_format=None
        )        
    val_ds = MTS_dataset(
        root_dir='data/svrt1_mts',
        split='val',
        normalization='CLIPViT',
        squeeze=True,
        invert=False,
        data_format=None
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
            normalization='CLIPViT'
            )
    for i in range(1, RUNS+1):
        # Insantiate model
        model = get_CLIP_ViT16(device=device, freeze_encoder=False)
        
        # Set to training mode 
        model.train()
        model.to(device)
        # Create optimizer and scheduler
        # optimizer = optim.Adam(model.parameters(), lr=LR)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
        # Do the training
        train_SD_CLIPViT(
            run=i,
            model=model,
            model_name='CLIPViT',
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
        test_SD_CLIPViT(
            run=i,
            model=model,
            model_name='CLIPViT',
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
    test_df_name = f'{current_dir}/CLIPViT_test_MTS.csv'
    test_df.to_csv(test_df_name)

    train_df = pd.concat(train_df)
    train_df_name = f'{current_dir}/CLIPViT_train_MTS.csv'
    train_df.to_csv(train_df_name)


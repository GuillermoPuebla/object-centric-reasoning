# Imports
import os
import csv
import pandas as pd
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from project_utils.dataloaders import RMTS_dataset, check_path
from project_utils.multi_task_batch_scheduler import BatchSchedulerSampler
from models.GAMR import MAREO

# Test datasets
def get_ds_names_and_loaders(batch_size, normalization='samplewise3'):
    """Returns names and dataloaders of new datastets for testing"""
    names_and_loaders = []

    # SVRT1
    ds = RMTS_dataset(
        root_dir='data/svrt1_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('SVRT1', loader))

    # Irregular translated
    ds = RMTS_dataset(
        root_dir='data/irregular_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Irregular', loader))

    # Regular translated
    ds = RMTS_dataset(
        root_dir='data/regular_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Regular', loader))

    # Open translated
    ds = RMTS_dataset(
        root_dir='data/open_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Open', loader))

    # Wider translated
    ds = RMTS_dataset(
        root_dir='data/wider_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Wider', loader))

    # Scrambled translated
    ds = RMTS_dataset(
        root_dir='data/scrambled_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Scrambled', loader))

    # Random translated
    ds = RMTS_dataset(
        root_dir='data/random_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Random', loader))

    # Filled translated
    ds = RMTS_dataset(
        root_dir='data/filled_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Filled', loader))

    # Lines translated
    ds = RMTS_dataset(
        root_dir='data/lines_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Lines', loader))

    # Arrows translated
    ds = RMTS_dataset(
        root_dir='data/arrows_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Arrows', loader))

    # Rectangles translated
    ds = RMTS_dataset(
        root_dir='data/rectangles_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Rectangles', loader))

    # Straight lines translated
    ds = RMTS_dataset(
        root_dir='data/slines_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('S-lines', loader))

    # Connected squares translated
    ds = RMTS_dataset(
        root_dir='data/csquares_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-squares', loader))

    # Connected circles translated
    ds = RMTS_dataset(
        root_dir='data/ccircles_rmts',
        split='test',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-circles', loader))

    return names_and_loaders

# Training function
def train_MAREO(
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
                y_pred_linear, y_pred = model(x, device)
                # Loss
                loss = loss_fn(y_pred_linear, target)
                running_train_loss += loss.item()
                # Update model
                loss.backward()
                optimizer.step()
                # Accuracy
                train_acc = torch.eq(y_pred, target).float().mean().item() * 100.0
                running_train_acc += train_acc
            
            scheduler.step()

            avg_train_loss = running_train_loss / (i + 1)
            avg_train_acc = running_train_acc / (i + 1)

            # Validation
            model.train(False)
            running_val_acc = 0.0
            running_val_loss = 0.0
            for j, (x, target) in enumerate(val_loader):
                # Load data to device
                x = x.to(device)
                target = target.to(device)
                # Run model
                y_pred_linear, y_pred = model(x, device)
                # Loss
                loss = loss_fn(y_pred_linear, target)
                running_val_loss += loss.item()
                # Accuracy
                train_acc = torch.eq(y_pred, target).float().mean().item() * 100.0
                running_val_acc += train_acc

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
def test_MAREO(
    run,
    model,
    model_name,
    device,
    ds_names_and_loaders,
    current_dir
    ):
    loss_fn = nn.CrossEntropyLoss()
    model.train(False)

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
                y_pred_linear, y_pred = model(x, device)
                # Loss
                loss = loss_fn(y_pred_linear, target)
                running_test_loss += loss.item()
                # Accuracy
                train_acc = torch.eq(y_pred, target).float().mean().item() * 100.0
                running_test_acc += train_acc

            avg_test_loss = running_test_loss / (j + 1)
            avg_test_acc = running_test_acc / (j + 1)

            # Save info to row
            row.append(avg_test_loss)
            row.append(avg_test_acc)
        # Write all data to file
        writer.writerow(row)

# Training and val dataset functions
def get_MTL_dataloader(batch_size, normalization='samplewise3'):
    """
    Returns multitask learning dataloader.
    Yields batches from a single dataset at the time.
    """
    ds_names = []
    ds_list = []

    # SVRT1
    ds = RMTS_dataset(
        root_dir='data/svrt1_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('SVRT1')
    ds_list.append(ds)

    # Irregular translated
    ds = RMTS_dataset(
        root_dir='data/irregular_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Irregular')
    ds_list.append(ds)

    # Regular translated
    ds = RMTS_dataset(
        root_dir='data/regular_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Regular')
    ds_list.append(ds)

    # Open translated
    ds = RMTS_dataset(
        root_dir='data/open_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Open')
    ds_list.append(ds)

    # Wider translated
    ds = RMTS_dataset(
        root_dir='data/wider_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Wider')
    ds_list.append(ds)

    # Scrambled translated
    ds = RMTS_dataset(
        root_dir='data/scrambled_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Scrambled')
    ds_list.append(ds)

    # Random translated
    ds = RMTS_dataset(
        root_dir='data/random_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Random')
    ds_list.append(ds)

    # Filled translated
    ds = RMTS_dataset(
        root_dir='data/filled_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Filled')
    ds_list.append(ds)

    # Lines translated
    ds = RMTS_dataset(
        root_dir='data/lines_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Lines')
    ds_list.append(ds)

    # Arrows translated
    ds = RMTS_dataset(
        root_dir='data/arrows_rmts',
        split='train',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Arrows')
    ds_list.append(ds)
    
    # Dataloader with BatchSchedulerSampler
    concat_dataset = ConcatDataset(ds_list)
    sampler = BatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset=concat_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False
        )
    return dataloader, ds_names

def get_val_dataloader(batch_size, normalization='samplewise3'):
    """
    Returns concatenated validation dataloader.
    """
    ds_names = []
    ds_list = []

    # SVRT1
    ds = RMTS_dataset(
        root_dir='data/svrt1_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('SVRT1')
    ds_list.append(ds)

    # Irregular translated
    ds = RMTS_dataset(
        root_dir='data/irregular_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Irregular')
    ds_list.append(ds)

    # Regular translated
    ds = RMTS_dataset(
        root_dir='data/regular_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Regular')
    ds_list.append(ds)

    # Open translated
    ds = RMTS_dataset(
        root_dir='data/open_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Open')
    ds_list.append(ds)

    # Wider translated
    ds = RMTS_dataset(
        root_dir='data/wider_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Wider')
    ds_list.append(ds)

    # Scrambled translated
    ds = RMTS_dataset(
        root_dir='data/scrambled_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Scrambled')
    ds_list.append(ds)

    # Random translated
    ds = RMTS_dataset(
        root_dir='data/random_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Random')
    ds_list.append(ds)

    # Filled translated
    ds = RMTS_dataset(
        root_dir='data/filled_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Filled')
    ds_list.append(ds)

    # Lines translated
    ds = RMTS_dataset(
        root_dir='data/lines_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Lines')
    ds_list.append(ds)

    # Arrows translated
    ds = RMTS_dataset(
        root_dir='data/arrows_rmts',
        split='val',
        normalization=normalization,
        invert=False,
        squeeze=False, 
        data_format=None
        )
    ds_names.append('Arrows')
    ds_list.append(ds)
    
    # Dataloader with BatchSchedulerSampler
    concat_dataset = ConcatDataset(ds_list)
    return DataLoader(
        dataset=concat_dataset,
        batch_size=batch_size,
        shuffle=False
        )


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # Limit number of threads 
    torch.set_num_threads(10)
    # SVRT-1 parameters
    max_epochs = 20 #200
    LR = 0.00001 #0.00001
    BATCH_SIZE = 64
    BATCH_SIZE_TEST = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUNS = 10 #10
    MAREO_TIME_STEPS = 8 #6 #4

    # Train on SVRT-1
    current_dir = 'results/RTE_RMTS_MAREO'
    check_path(current_dir)

    train_loader, ds_names = get_MTL_dataloader(
        batch_size=BATCH_SIZE, 
        normalization='samplewise3'
        )
    val_loader = get_val_dataloader(
        batch_size=BATCH_SIZE,
        normalization='samplewise3'
        )
    test_names_loaders = get_ds_names_and_loaders(
            batch_size=BATCH_SIZE_TEST,
            normalization='samplewise3'
            )
    # Train
    for i in range(1, RUNS+1):
        # Insantiate model
        model = MAREO(
            encoder='custom',
            norm_type='contextnorm', 
            steps=MAREO_TIME_STEPS
            )
        # Set to training mode 
        model.train()
        model.to(device)
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
        # Train
        train_MAREO(
            run=i,
            model=model,
            model_name='MAREO',
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=max_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            current_dir=current_dir
            )
        # Set to evaluation mode
        model.eval()
        # Test
        test_MAREO(
            run=i,
            model=model,
            model_name='MAREO',
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
    test_df_name = f'{current_dir}/MAREO_test_RTE_RMTS.csv'
    test_df.to_csv(test_df_name)

    train_df = pd.concat(train_df)
    train_df_name = f'{current_dir}/MAREO_train_RTE_RMTS.csv'
    train_df.to_csv(train_df_name)
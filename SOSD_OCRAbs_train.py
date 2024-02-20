# Imports
import os
import csv
import pandas as pd
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from project_utils.dataloaders import SOSD_dataset, check_path
from models.OCRAbs import SlotAttentionAutoEncoder, scoring_model


def load_slot_checkpoint(slot_model,checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn model
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn model with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path)
    slot_model.load_state_dict(model_ckp['slot_model_state_dict'])
    return slot_model

# Test datasets
def get_ds_names_and_loaders(batch_size, normalization='OCRAbs'):
    """Returns names and dataloaders of new datastets for testing"""
    names_and_loaders = []
        
    # SVRT1
    ds = SOSD_dataset(
        root_dir='data/svrt1_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('SVRT1', loader))

    # Irregular translated
    ds = SOSD_dataset(
        root_dir='data/irregular_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Irregular', loader))

    # Regular translated
    ds = SOSD_dataset(
        root_dir='data/regular_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Regular', loader))

    # Open translated
    ds = SOSD_dataset(
        root_dir='data/open_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Open', loader))

    # Wider translated
    ds = SOSD_dataset(
        root_dir='data/wider_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Wider', loader))

    # Scrambled translated
    ds = SOSD_dataset(
        root_dir='data/scrambled_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Scrambled', loader))

    # Random translated
    ds = SOSD_dataset(
        root_dir='data/random_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Random', loader)) # Scrambled.1

    # Filled translated
    ds = SOSD_dataset(
        root_dir='data/filled_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Filled', loader))

    # Lines translated
    ds = SOSD_dataset(
        root_dir='data/lines_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Lines', loader))

    # Arrows translated
    ds = SOSD_dataset(
        root_dir='data/arrows_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Arrows', loader))

    # Rectangles translated
    ds = SOSD_dataset(
        root_dir='data/rectangles_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Rectangles', loader))

    # Straight lines translated
    ds = SOSD_dataset(
        root_dir='data/slines_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('S-lines', loader))

    # Connected squares translated
    ds = SOSD_dataset(
        root_dir='data/csquares_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-squares', loader))

    # Connected circles translated
    ds = SOSD_dataset(
        root_dir='data/ccircles_sosd',
        split='test',
        normalization=normalization,
        squeeze=False,
        invert=False,
        data_format=None
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-circles', loader))

    return names_and_loaders

# Training function
def train_sd_OCRAbs(
    run,
    slot_model,
    ocrabs_model,
    model_name,
    device,
    optimizer,
    scheduler,
    epochs,
    train_loader,
    val_loader,
    current_dir
    ):
    # Loss functions
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    sigmoid_activation = nn.Sigmoid()

    # Create filename for saving training progress
    train_file = f"{current_dir}/model_{model_name}_run_{run}_train.csv"
    header = [
        'Model', 
        'Run', 
        'Epoch', 
        'Loss 1',
        'Loss 2',
        'Accuracy',
        'Val loss 1',
        'Val loss 2',
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
            slot_model.eval()
            ocrabs_model.train()
            running_train_acc = 0.0
            running_train_loss_1 = 0.0
            running_train_loss_2 = 0.0
            # load batch from dataloader 
            for i, (x, y) in enumerate(train_loader):
                # Load data to device
                x = x.to(device).float()
                y = y.to(device).float()
                
                # Zero out gradients for optimizer 
                optimizer.zero_grad()
                
                # Run model
                recon_combined, recons, masks, feat_slots, pos_slots, attn = slot_model(x, device)
                score = ocrabs_model(feat_slots, pos_slots, device)
                pred = torch.round(sigmoid_activation(score)).int()

                # Update model
                loss = bce_criterion(sigmoid_activation(score), y)
                loss.backward()
                optimizer.step()

                # Loss
                running_train_loss_1 += mse_criterion(recon_combined, x).item()
                running_train_loss_2 += loss.item()
                # Accuracy
                running_train_acc += torch.eq(pred, y).float().mean().item() * 100
            

            avg_train_loss_1 = running_train_loss_1 / (i + 1)
            avg_train_loss_2 = running_train_loss_2 / (i + 1)
            avg_train_acc = running_train_acc / (i + 1)


			########### I', here ############

            # Val accuracy
            ocrabs_model.train(False)
            running_val_acc = 0.0
            running_val_loss_1 = 0.0
            running_val_loss_2 = 0.0
            for j, (x, y) in enumerate(val_loader):
                # Load data to device
                x = x.to(device).float()
                y = y.to(device).float()
                
                # Run model 
                recon_combined, recons, masks, feat_slots, pos_slots, attn = slot_model(x, device)
                score = ocrabs_model(feat_slots, pos_slots, device)
                pred = torch.round(sigmoid_activation(score)).int()
                
                # Loss
                loss = bce_criterion(sigmoid_activation(score), y)
                running_val_loss_1 += mse_criterion(recon_combined, x).item()
                running_val_loss_2 += loss.item()
                # Accuracy
                running_val_acc += torch.eq(pred, y).float().mean().item() * 100

            avg_val_loss_1 = running_val_loss_1 / (j + 1)
            avg_val_loss_2 = running_val_loss_2 / (j + 1)
            avg_val_acc = running_val_acc / (j + 1)

            scheduler.step(avg_val_loss_2)
            
            # Save info to file
            row = [
                model_name, # 'Model'
                run, # 'Run'
                epoch, # 'Epoch'
                avg_train_loss_1, # 'Loss 1'
                avg_train_loss_2, # 'Loss 1'
                avg_train_acc, # 'Accuracy'
                avg_val_loss_1, # 'Val loss 1'
                avg_val_loss_2, # 'Val loss 2'
                avg_val_acc # 'Val accuracy'
                ]
            writer.writerow(row)

            # Report
            if epoch % 1 == 0:
                print('[Epoch: ' + str(epoch) + '] ' + \
                        '[t_l_1 = ' + '{:.4f}'.format(avg_train_loss_1) + '] ' + \
                        '[v_l_1 = ' + '{:.4f}'.format(avg_val_loss_1) + '] ' + \
                        '[t_l_2 = ' + '{:.4f}'.format(avg_train_loss_2) + '] ' + \
                        '[v_l_2 = ' + '{:.4f}'.format(avg_val_loss_2) + '] ' + \
                        '[t_acc = ' + '{:.2f}'.format(avg_train_acc) + '] ' + \
                        '[v_acc = ' + '{:.2f}'.format(avg_val_acc) + '] ')
            
            # Track best performance, and save the model's state
            if avg_val_loss_2 < best_val_loss:
                best_val_loss = avg_val_loss_2
                model_path =  f"{current_dir}/model_{model_name}_run_{run}_best_epoch_weights.pt"
                torch.save(ocrabs_model.state_dict(), model_path)
            
            if avg_val_acc > 99.0:
                break


# Test function
def test_sd_OCRAbs(
    run,
    slot_model,
    ocrabs_model,
    model_name,
    device,
    ds_names_and_loaders,
    current_dir
    ):
    # Loss function
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    sigmoid_activation = nn.Sigmoid()
    
    # Create filename for saving training progress
    test_file = f"{current_dir}/model_{model_name}_run_{run}_test.csv"
    header = ['Model', 'Run']
    for name, loader in ds_names_and_loaders:
        loss_name_1 = f"Loss 1 {name}"
        loss_name_2 = f"Loss 2 {name}"
        acc_name = f"Accuracy {name}"
        header.append(loss_name_1)
        header.append(loss_name_2)
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
            running_test_loss_1 = 0.0
            running_test_loss_2 = 0.0
            for j, (x, y) in enumerate(test_loader):
                # Load data to device
                x = x.to(device).float()
                y = y.to(device).float()
                
                # Run model 
                recon_combined, recons, masks, feat_slots, pos_slots, attn = slot_model(x, device)
                score = ocrabs_model(feat_slots, pos_slots, device)
                pred = torch.round(sigmoid_activation(score)).int()
                
                # Loss
                loss = bce_criterion(sigmoid_activation(score), y)
                running_test_loss_1 += mse_criterion(recon_combined, x).item()
                running_test_loss_2 += loss.item()
                
                # Accuracy
                running_test_acc += torch.eq(pred, y).float().mean().item() * 100

            avg_test_loss_1 = running_test_loss_1 / (j + 1)
            avg_test_loss_2 = running_test_loss_2 / (j + 1)
            avg_test_acc = running_test_acc / (j + 1)
            
            # Save info to row
            row.append(avg_test_loss_1)
            row.append(avg_test_loss_2)
            row.append(avg_test_acc)
        # Write all data to file
        writer.writerow(row)



if __name__ == "__main__":
    mp.set_start_method('spawn')
    # Limit number of threads 
    torch.set_num_threads(10)
    # Model parameters 
    IMG_SIZE = 128
    NUM_SLOTS = 6 # Number of slots in Slot Attention
    NUM_ITERATIONS = 3 # Number of attention iterations
    HID_DIM = 64 # hidden dimension size
    DEPTH = 24 # transformer number of layers
    HEADS = 8 # transformer number of heads
    MLP_DIM = 512 # transformer mlp dimension
    slot_weights = 'pre_weights/slot_attention_autoencoder_augmentations_6slots_svrt_1000samples_best.pth.tar'
    # SVRT-1 parameters
    EPOCHS = 50
    BATCH_SIZE = 32
    BATCH_SIZE_TEST = 32
    LR = 0.000004 #0.00004
    WEIGHT_DECAY = 0.0
    SCHEDULER_GAMMA = 0.5
    WARMUP_STEPS_PCT = 0.02
    DECAY_STEPS_PCT = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUNS = 10
    
    # Train on SVRT-1
    current_dir = 'results/SOSD_OCRAbs'
    check_path(current_dir)
    
    train_ds = SOSD_dataset(
        root_dir='data/svrt1_sosd',
        split='train',
        normalization='OCRAbs', 
        squeeze=False, 
        invert=False
        )
    val_ds = SOSD_dataset(
        root_dir='data/svrt1_sosd',
        split='val',
        normalization='OCRAbs', 
        squeeze=False, 
        invert=False
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
            normalization='OCRAbs'
            )
    for i in range(1, RUNS+1):
        # Insantiate model
        slot_model = SlotAttentionAutoEncoder((IMG_SIZE, IMG_SIZE), NUM_SLOTS, NUM_ITERATIONS, HID_DIM).to(device)
        ocrabs_model = scoring_model(HID_DIM, DEPTH, HEADS, MLP_DIM, NUM_SLOTS, True).to(device)
        slot_model = load_slot_checkpoint(slot_model, slot_weights)
        # model= nn.DataParallel(model)
        
        # Load weights
        model_name = 'OCRAbs'
        run = i
        weights_path = f"{current_dir}/model_{model_name}_run_{run}_best_epoch_weights.pt"
        ocrabs_model.load_state_dict(torch.load(weights_path))

        # Set to training mode 
        slot_model.eval()
        ocrabs_model.train()
        
        # Create optimizer and scheduler
        params = [{'params': ocrabs_model.parameters()}]
        optimizer = optim.Adam(params, lr=LR)
        total_steps = EPOCHS * len(train_loader) # Total steps for scheduler

        def warm_and_decay_lr_scheduler(step):
            warmup_steps = WARMUP_STEPS_PCT * total_steps
            decay_steps = DECAY_STEPS_PCT * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= SCHEDULER_GAMMA ** (step / decay_steps)
            return factor

        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
        
        # Do the training
        train_sd_OCRAbs(
            run=i,
            slot_model=slot_model,
            ocrabs_model=ocrabs_model,
            model_name='OCRAbs',
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=EPOCHS,
            train_loader=train_loader,
            val_loader=val_loader,
            current_dir=current_dir
            )
        # Set to evaluation mode
        ocrabs_model.eval()

        # Test
        test_sd_OCRAbs(
            run=i,
            slot_model=slot_model,
            ocrabs_model=ocrabs_model,
            model_name='OCRAbs',
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
    test_df_name = f'{current_dir}/OCRAbs_test_SOSD.csv'
    test_df.to_csv(test_df_name)
    
    train_df = pd.concat(train_df)
    train_df_name = f'{current_dir}/OCRAbs_train_SOSD.csv'
    train_df.to_csv(train_df_name)

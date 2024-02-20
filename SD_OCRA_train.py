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
from models.OCRA import OCRA, evaluate, param2args, loss_fn


# Test datasets
def get_ds_names_and_loaders(batch_size, normalization=None):
    """Returns names and dataloaders of new datastets for testing"""
    names_and_loaders = []
    
    # SVRT1 translated
    ds = SD_dataset(
        root_dir='data/svrt1_sd', 
        split='test', 
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('svrt1_translated', loader))

    # Irregular translated
    ds = SD_dataset(
        root_dir='data/irregular_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Irregular', loader))

    # Regular translated
    ds = SD_dataset(
        root_dir='data/regular_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Regular', loader))

    # Open translated
    ds = SD_dataset(
        root_dir='data/open_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Open', loader))

    # Wider translated
    ds = SD_dataset(
        root_dir='data/wider_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Wider', loader))

    # Scrambled translated
    ds = SD_dataset(
        root_dir='data/scrambled_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Scrambled', loader))

    # Random translated
    ds = SD_dataset(
        root_dir='data/random_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Random', loader))

    # Filled translated
    ds = SD_dataset(
        root_dir='data/filled_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Filled', loader))

    # Lines translated
    ds = SD_dataset(
        root_dir='data/lines_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Lines', loader))

    # Arrows translated
    ds = SD_dataset(
        root_dir='data/arrows_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Arrows', loader))

    # Rectangles translated
    ds = SD_dataset(
        root_dir='data/rectangles_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('Rectangles', loader))

    # Straight lines translated
    ds = SD_dataset(
        root_dir='data/slines_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('S-lines', loader))

    # Connected squares translated
    ds = SD_dataset(
        root_dir='data/csquares_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-squares', loader))

    # Connected circles translated
    ds = SD_dataset(
        root_dir='data/ccircles_sd',
        split='test',
        normalization=normalization, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    loader = DataLoader(ds, batch_size=batch_size)
    names_and_loaders.append(('C-circles', loader))

    return names_and_loaders

# Train function
def train_sd_OCRA(
    run,
    model,
    model_name,
    device,
    optimizer,
    epochs,
    train_loader,
    val_loader,
    current_dir,
    args,
    loss_fn
    ):
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
            for i, (x, y, yloss_w) in enumerate(train_loader):                
                # if one target and y is not in one-hot format, convert it to one-hot encoding
                if args.num_targets == 1:
                    if len(y.shape) < 2: 
                        y = y.type(torch.int64)
                        y = torch.zeros(y.size(0), args.num_classes).scatter_(1, y.view(-1, 1), 1.) 
                # load dataset on device
                x = x.view(x.shape[0], -1).to(device)
                y = y.to(device)
                yloss_w = yloss_w.to(device)
                # forward pass
                objcaps_len_step, read_x_step, c_step, readout_logits  = model(x)
                # compute loss for this batch and append it to training loss
                loss, _ , _ = loss_fn(objcaps_len_step, read_x_step, c_step, readout_logits, x, y, yloss_w, args, device)
                train_loss = loss.item()
                running_train_loss += train_loss                
                # zero out previous gradients and backward pass
                optimizer.zero_grad()
                loss.backward()
                # clip grad norm to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), 10)
                # update param
                optimizer.step()
                # Train accuracy
                batch_loss, batch_L_recon, batch_L_margin, batch_acc_exact, batch_acc_partial, read_x_step, \
                c_each_step, y_pred, objcaps_len_step, readout_logits =  evaluate(model, x, y, loss_fn, args, device)
                running_train_acc += batch_acc_exact * 100
            
            avg_train_loss = running_train_loss / (i + 1)
            avg_train_acc = running_train_acc / len(train_loader.dataset)

            # Val accuracy
            model.train(False)
            running_val_acc = 0.0
            running_val_loss = 0.0
            for j, (x, y, yloss_w) in enumerate(val_loader):
                # load dataset on device
                x = x.view(x.shape[0], -1).to(device)
                y = y.to(device)
                yloss_w = yloss_w.to(device)
                # forward pass
                objcaps_len_step, read_x_step, c_step, readout_logits  = model(x)
                # compute loss for this batch and append it to training loss
                loss, _ , _ = loss_fn(objcaps_len_step, read_x_step, c_step, readout_logits, x, y, yloss_w, args, device)
                val_loss = loss.item()

                running_val_loss += val_loss
                # Val accuracy
                batch_loss, batch_L_recon, batch_L_margin, batch_acc_exact, batch_acc_partial, read_x_step, \
                c_each_step, y_pred, objcaps_len_step, readout_logits =  evaluate(model, x, y, loss_fn, args, device)
                running_val_acc += batch_acc_exact * 100

            avg_val_loss = running_val_loss / (j + 1)
            avg_val_acc = running_val_acc / len(val_loader.dataset)

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
def test_sd_OCRA(
    run,
    model,
    model_name,
    device,
    ds_names_and_loaders,
    current_dir,
    args,
    loss_fn
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
        for name, test_loader in ds_names_and_loaders:
            running_test_acc = 0.0
            running_test_loss = 0.0
            for j, (x, y, yloss_w) in enumerate(test_loader):
                # load dataset on device
                x = x.view(x.shape[0], -1).to(device)
                y = y.to(device)
                yloss_w = yloss_w.to(device)
                print(y) # tensor([[1., 1., 0., 1.], [1., 1., 1., 0.]], device='cuda:0')
                print(y.shape) # torch.Size([2, 4])
                input()
                # forward pass
                objcaps_len_step, read_x_step, c_step, readout_logits  = model(x)
                # compute loss for this batch and append it to training loss
                loss, _ , _ = loss_fn(objcaps_len_step, read_x_step, c_step, readout_logits, x, y, yloss_w, args, device)
                test_loss = loss.item()

                running_test_loss += test_loss
                # Val accuracy
                batch_loss, batch_L_recon, batch_L_margin, batch_acc_exact, batch_acc_partial, read_x_step, \
                c_each_step, y_pred, objcaps_len_step, readout_logits =  evaluate(model, x, y, loss_fn, args, device)
                running_test_acc += batch_acc_exact * 100

            avg_test_loss = running_test_loss / (j + 1)
            avg_test_acc = running_test_acc / len(test_loader.dataset)
            
            # Save info to row
            row.append(avg_test_loss)
            row.append(avg_test_acc)
        # Write all data to file
        writer.writerow(row)

# Method for creating directory if it doesn't exist yet
def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    # SVRT-1 parameters
    EPOCHS = 100
    BATCH_SIZE = 128
    BATCH_SIZE_TEST = 2 #128
    LR = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUNS = 10
    
    # Set up OCRA parameters
    ocra_params = {
        # task info
        "task": "svrt_task1", 
        "num_classes": 4,
        "num_targets": 3,
        "image_dims": (1, 128, 128),
        "cat_dup": False, # if True, objects can come from the same category 

        # training and testing
        "n_epochs": 100,
        "lr": 0.001,
        "train_batch_size": 128,
        "test_batch_size": 128,
        "cuda": 0, # cuda device number
        'device': None,

        # directories and logging
        "data_dir": "./data/",
        "output_dir": "./results/svrt_task1/", # where best performing model will be saved and log_dir is created 
        'log_dir': './results/svrt_task1/Oct22_2022',
        "restore_file": None, # checkpoint file to restore and resume training, if none, set as None

        "save_checkpoint": True, # save checkpoint
        "record_gradnorm": False, # whether log model gradnorm to writer
        "record_attn_hooks": False, # whether record forward and backward attention hooks
        "validate_after_howmany_epochs": 1, # validate after how many epoch
        "best_val_acc": 0, # only save model that achieved val_acc above this value
        "verbose": True, # whether print validation loss/acc

        ## model architecture

        # read and write operations
        "use_read_attn": True,
        "read_size": 18, # 18
        "use_write_attn": True,
        "write_size": 18, # 18

        # whether to apply convolutional layers to read images 
        "use_backbone": "conv_med", # if False, feed raw pixels to encoder
        "conv1_nfilters": 32,
        "conv2_nfilters": 32,

        # the number of complete cycle of encoder-decoder
        "time_steps": 10,  

        # encoder/decoder RNN size and whether to include xhat as input to encoder
        "include_xhat": False,
        "lstm_size": 512, # 512
        "decoder_encoder_feedback" : True,

        # dynamic routing capsules
        "use_capsnet": True, # if false, just linear readout will be used instead of capsulenet representations
        "num_zcaps": 40, # size of linear layer from encoder to primary caps
        "dim_zcaps": 8, # primary caps dim, note z_size/dim_zcaps = num_primarycaps
        "routings": 3, # the number of dynamic routing
        "dim_objectcaps": 16, # final class object caps
        "backg_objcaps": 0, # the number of capsules for background

        # decoder/reconstruction
        "mask_objectcaps": False, # if True, use masked objectcaps for decoder 
        "class_cond_mask": False, # if True, groundtruth info will be use to mask objectcaps, if False, use the most active one and mask others.   
        "recon_model": True, # if True, decoder generates a reconstruction of the input (lam_recon should be also set as zero)


        # loss function
        "lam_recon": 60, # weight for reconstruction error 3 for 3 steps, 10 for 10 steps
        "clip_c": True, # if True, clip the predicted cumulative canvas to 1 
        "use_recon_mask": False, # if True, only consider input regions reconstructed by the model for calculating reconstruction loss
        }
    args = param2args(ocra_params)

    # Train on SVRT-1
    current_dir = 'results/SD_OCRA'
    check_path(current_dir)

    train_ds = SD_dataset(
        root_dir='data/svrt1_single_img', 
        split='train', 
        normalization=None, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
        )
    val_ds = SD_dataset(
        root_dir='data/svrt1_single_img', 
        split='val', 
        normalization=None, 
        squeeze=False, 
        invert=True,
        data_format='OCRA'
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
            normalization=None
            )
    for i in range(1, RUNS+1):
        # Insantiate model
        model = OCRA(args)

        # Set to training mode 
        model.train()
        model.to(device)
        # Create optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
        # Do the training
        train_sd_OCRA(
            run=i,
            model=model,
            model_name='OCRA',
            device=device,
            optimizer=optimizer,
            epochs=EPOCHS,
            train_loader=train_loader,
            val_loader=val_loader,
            current_dir=current_dir,
            args=args,
            loss_fn=loss_fn
            )
        # Set to evaluation mode
        model.eval()
        # Test
        test_sd_OCRA(
            run=i,
            model=model,
            model_name='OCRA',
            device=device,
            ds_names_and_loaders=test_names_loaders,
            current_dir=current_dir,
            args=args,
            loss_fn=loss_fn
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
    test_df_name = f'{current_dir}/OCRA_test_SD.csv'
    test_df.to_csv(test_df_name)

    train_df = pd.concat(train_df)
    train_df_name = f'{current_dir}/OCRA_train_SD.csv'
    train_df.to_csv(train_df_name)

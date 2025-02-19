import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from HG_models import FanCustom
import time
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    ########################### Hyper-Parameters for training ###########################
    batch_size = 8
    learning_rate = 1e-5 
    w_decay = 1e-6 
    total_layers = 6  # The number of stages for FanCustom model only
    epochs = 150
    save_model = True
    if save_model:
        save_path = './models'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    experiment_name = 'actys_train' # Also the name of the model to be saved
    best_val_loss = 9999
    #####################################################################################
    # Load Dataset for training and validation
    # Your Dataset class must return the input image and the ground truth data. 
    # The input and g.t. must have the same spatial dimensions.
    val_set = ...
    train_set = ...
    input_channels = 3 # 3 for RGB, 1 for Grayscale
    #####################################################################################


    print('Training set: ', len(train_set), 'samples')
    print('Validation set:  ', len(val_set), 'samples')

    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
  
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Will train on ', device)

    # Load Network Architecture
    net = FanCustom(total_layers, in_channels=input_channels)
    # Send Network to GPU (if any)
    net.to(device)
    
    # Define a Loss function and optimizer
    cret = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=w_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5)

    train_loss = []
    val_loss = []
    # Train the network
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch + 1) + '/' + str(epochs))
        start = time.time()

        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            input_data, gt_depth = data
            
            input_data, gt_depth = input_data.to(device), gt_depth.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            est_depths = net(input_data)

            depth_est_loss = sum([cret(est_depths[j], gt_depth) for j in range(total_layers)])/total_layers

            running_loss += depth_est_loss.item()
            depth_est_loss.backward()
            optimizer.step()

        running_loss = running_loss / (i + 1)
        train_loss.append(running_loss)
        scheduler.step()
        net.eval()

        test_error = 0.0
        j = 1.0
        with torch.no_grad():
            for val_data in val_loader:
                val_input_data, val_gt_depth = val_data
                val_input_data, val_gt_depth = val_input_data.to(device), val_gt_depth.to(device)
                outputs = net(val_input_data)
                # Take the output from the last stacked hourglass
                test_error += F.mse_loss(val_gt_depth, outputs[-1])
                j += 1
        test_error = test_error / (j)
        val_loss.append(test_error)

        print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: S1:{:.8f}'.format(epoch + 1, epochs, running_loss, test_error))
        if save_model:
            if test_error < best_val_loss:
                best_val_loss = test_error
                print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(epoch + 1, test_error))
                torch.save(net.state_dict(), save_path+'/'+experiment_name + '.pt')

        end = time.time() - start
        print('Finished Epoch ' + str(epoch + 1) + ' in ' + str(end) + ' seconds')

    fig=plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, epochs+1), train_loss, label="Train loss")
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title("Train Loss Plot")
    plt.show()

    fig=plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, epochs+1), val_loss, label="Validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title("Validation Loss Plot")
    plt.show()
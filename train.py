import time
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from data_loader import get_dataloader
from model import EnhancedFaceParsingNet
from loss import my_criterion
from utils import compute_multiclass_fscore

def train(
        device, 
        model=EnhancedFaceParsingNet(), 
        epochs=50, 
        loss_option="CE", 
        lr = 0.001, 
        scheduler_step = 10, 
        scheduler_gamma = 0.2,
    ):

    print(f"Using device: {device}")
    model = model.to(device)

    # get training loader
    train_loader = get_dataloader(is_train=True)

    # initialize hyperparameters
    criterion = my_criterion(loss_option, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    for ith_epoch in range(epochs):
        print('-' * 30)
        print(f'       Epoch {ith_epoch + 1}/{epochs}')
        print('-' * 30)

        time_start = time.time()

        # first, train
        model.train()
        for inputs, labels in train_loader:
            # move to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero gradient
            optimizer.zero_grad()

            # compute loss, gradients, and perform backpropagation
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # increment scheduler after one full epoch training
        scheduler.step()

        # next, evaluate the current performance of the model using customized metrics
        f1_list = []

        model.eval()
        with torch.no_grad():
            # still use training data for now
            for inputs, labels in train_loader:
                # move to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # get predictions from logits tensor([batch_size, 19, 512, 512]) into tensor([batch_size, 512, 512])
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)

                # compute F1 score per image in this mini-batch, and append to the total list
                batch_size = labels.shape[0]
                for j in range(batch_size):
                    f1 = compute_multiclass_fscore(labels[j].cpu().numpy(), preds[j].cpu().numpy())
                    f1_list.append(f1)

            # average score for all images in the validation dataset
            print(f"Average F1 score: {np.mean(f1_list)}")

        time_end = time.time()

        print(f"Learning rate: {scheduler.get_last_lr()[0]}")
        print(f"Epoch {ith_epoch + 1} completed in {time_end - time_start:.2f} seconds\n")

    # output trained model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth")
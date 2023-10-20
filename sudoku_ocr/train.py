import torch
import torch.nn as nn
import time
from torch.optim import Adam, lr_scheduler
from ocr_dataset import create_dataloader
import numpy as np
from sudoku_ocr.ocr_model import DigitOCR

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DigitOCR()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=0.)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=1e-1)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    n_epochs = 15
    train_loader, test_loader = create_dataloader()
    for epoch in range(n_epochs):
        print('-' * 10)
        print('Epoch {}, lr = {}'.format(epoch, scheduler.get_last_lr()[0]))
        start = time.time()
        model.train()
        losses = []

        for inputs, targets in train_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        losses = np.array(losses)
        print('Epoch average training loss = {}'.format(losses.mean()))
        model.eval()
        losses = []

        for inputs, targets in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)
            loss = torch.mean((torch.argmax(outputs, dim=1) == targets).double())
            losses.append(loss.item())

        losses = np.array(losses)
        print('Epoch average validation accuracy = {}'.format(losses.mean()))
        elapsed = time.time() - start
        print('{:.0f}m{:.0f}s elapsed'.format(elapsed // 60, elapsed % 60))
        scheduler.step()

    # Save
    torch.save(model.state_dict(), "ocr_digit.pth")

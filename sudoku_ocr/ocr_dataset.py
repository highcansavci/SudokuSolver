import os
import torch
import csv
import io
from PIL import Image
from torch.utils import data
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import base64
from sudoku_ocr.ocr_model import GaussianNoise


def base64ToPIL(x):
    return Image.open(io.BytesIO(base64.b64decode(x.encode())))

def create_dataloader():
    # Training set
    norm = transforms.ToTensor()

    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        norm,
        GaussianNoise(mean=0., std=0.15)
    ])

    train_dataset = AQMNIST(transform=transform, what='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=True)

    test_dataset = AQMNIST(transform=norm, what="test")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False)

    return train_loader, test_loader


class AQMNIST(Dataset):
    """Augmented QMNIST.

	The augmentation consists in adding empty images, i.e. without digit,
	as well as printed digits (not handwritten) with various fonts.

	References
	----------
	Cold Case: The Lost MNIST Digits
	(Yadav C. and Bottou L., 2019)
	https://arxiv.org/abs/1905.10498
	"""

    black = Image.new('L', (28, 28))

    def __init__(self, transform=lambda x: x, what='train'):
        super().__init__()

        self.transform = transform
        self.qmnist = datasets.QMNIST(root='.', what=what, download=True)

        self.printed = []

        if os.path.exists('printed_digits.csv'):
            with open('printed_digits.csv', 'r') as f:
                reader = csv.reader(f, delimiter=',')

                for row in reader:
                    self.printed.append((
                        base64ToPIL(row[2]).convert('L'),
                        int(row[1])
                    ))

        self.printed = [
            x
            for i, x in enumerate(self.printed)
            if (i % 5 == 0) == (what == 'test')
        ]

    def __len__(self):
        return len(self.qmnist) + len(self.printed) + len(self.qmnist) // 10

    def __getitem__(self, i):
        if i < len(self.qmnist):
            inpt, targt = self.qmnist[i]
        elif i < len(self.qmnist) + len(self.printed):
            inpt, targt = self.printed[i - len(self.qmnist)]
        else:
            inpt, targt = self.black, 10

        return self.transform(inpt), targt

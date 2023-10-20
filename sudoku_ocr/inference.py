import torch
import torchvision.transforms as transforms
import cv2

def ocr_inference(img, model):
    transform = transforms.ToTensor()

    model.eval()

    with torch.no_grad():
        batch = torch.stack([
            transform(x).float() for x in img
        ])

        if torch.cuda.is_available():
            batch = batch.cuda()

        output = model(batch)

    dgts = torch.argmax(output, dim=1).cpu().reshape(9, 9).numpy()
    dgts[dgts == 10] = 0

    return dgts

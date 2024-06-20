from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
import cv2
import torch
import tqdm
import numpy as np

import os

input_path = "data/Stone331/origin"
output_path = "outputs"

if not os.path.exists(output_path):
    os.mkdir(output_path)

model = DeepCrack()
device = torch.device("cpu")

model.load_state_dict(torch.load("checkpoints/DeepCrack_CT260_FT1.pth", map_location=None if torch.cuda.is_available() else 'cpu'))

input_list = []
for root, dirs, files in os.walk(input_path):
    for file in files:
        input_list.append(os.path.join(root, file))

# set the model to evaluation mode
model.eval()

with torch.no_grad():
    for input_file in tqdm.tqdm(input_list):
        # read the image and preprocess it
        img = cv2.imread(input_file).astype(np.float32) / 255
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        test_pred = model(img)
        test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
        save_name = os.path.join(output_path, os.path.split(input_file)[1])
        save_pred = test_pred.numpy() * 255
        cv2.imwrite(save_name, save_pred.astype(np.uint8))
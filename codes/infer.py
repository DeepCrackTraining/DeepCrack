from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
import cv2
import torch
import tqdm
import numpy as np

import os

input_path = "inputs"
output_path = "outputs"

if not os.path.exists(output_path):
    os.mkdir(output_path)

model = DeepCrack()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load("checkpoints/DeepCrack_CT260_FT1/checkpoints/DeepCrack_CT260_FT1_DeepCrack_CT260_FT1_epoch(1)_0000072_2024-06-20-18-45-14.pth", map_location=None if torch.cuda.is_available() else 'cpu'))

input_list = []
for root, dirs, files in os.walk(input_path):
    for file in files:
        input_list.append(os.path.join(root, file))

# set the model to evaluation mode
model.eval()

with torch.no_grad():
    for input_file in tqdm.tqdm(input_list):
        # read the image and preprocess it
        img = cv2.imread(input_file, cv2.IMREAD_COLOR)
        size = img.shape[:2]
        img = cv2.resize(img, (1440, 1440))
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        img = torch.from_numpy(img).float()

        # make the prediction
        output = model(img.to(device))
        output = torch.sigmoid(output[0].cpu().squeeze())
        save_pred = output
        save_name = os.path.join(output_path, os.path.split(input_file)[1])
        save_pred = save_pred.numpy() * 255

        # resize the output to the original size
        save_pred = cv2.resize(save_pred, (size[1], size[0]))
        cv2.imwrite(save_name, save_pred)

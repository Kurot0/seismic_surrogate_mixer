import os
import torch
import numpy as np
from scipy.ndimage import zoom


dataName = "data/prep_data/underGroundCSV2Dict.pt"
dirPath = "data/exp_data"
outputFilePath = os.path.join(dirPath, "upperDepth_lonlat_400.pt")

if not os.path.isdir(dirPath):
    os.makedirs(dirPath)

dataDict = torch.load(dataName)

XPO = np.array(dataDict["XPO"])  
YPO = np.array(dataDict["YPO"]) 
JLON = np.array(dataDict["JLON"])  
JLAT = np.array(dataDict["JLAT"])  
Ts = np.array(dataDict["Ts"])  

max_x = np.max(XPO)
max_y = np.max(YPO)

depthImages = np.zeros((1, Ts.shape[0], max_y, max_x))

for i in range(Ts.shape[1]):
    x = XPO[i] - 1  
    y = max_y - YPO[i]  
    depthImages[0, :, y, x] = Ts[:, i]  

depthImages = zoom(depthImages, (1, 1, 518 / max_y, 508 / max_x), order=1)

depthImages = depthImages[:, :, 67:-51, 92:-16]

depthImages = torch.tensor(depthImages)
torch.save(depthImages, outputFilePath)
print("Data saved to", outputFilePath)

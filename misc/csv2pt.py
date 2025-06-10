import os
import csv
import torch


dataName = "data/raw_data/Nankai_JIVSM111110-50m_non1st_100km_130814.csv"
dirPath = "data/prep_data"  
ptFilePath = os.path.join(dirPath, "underGroundCSV2Dict.pt")

if not os.path.isdir(dirPath):
    os.makedirs(dirPath)

layerNum = 23  

csvData = csv.reader(open(dataName, "r"), delimiter=",", doublequote=True, lineterminator="\r\n")

for _ in range(42):
    _ = next(csvData)

XPO, YPO = [], []  
JLON, JLAT, WLON, WLAT = [], [], [], []  
Ts = [[] for _ in range(layerNum)]  

ite = 0
for row in csvData:
    ite += 1
    print("\r csv2pt : already read {} rows".format(ite), end="")
    if len(row) < 10 + 2 * layerNum:  
        print(f"\nSkipped row {ite} due to insufficient columns")
        continue  

    XPO.append(int(row[1]))
    YPO.append(int(row[0]))
    JLON.append(float(row[2]))
    JLAT.append(float(row[3]))
    WLON.append(float(row[4]))
    WLAT.append(float(row[5]))

    cumulative_depth = 0
    for i in range(layerNum):
        cumulative_depth += float(row[11 + 2 * i]) 
        Ts[i].append(cumulative_depth)

dataDict = {
    "XPO": XPO,
    "YPO": YPO,
    "JLON": JLON,
    "JLAT": JLAT,
    "WLON": WLON,
    "WLAT": WLAT,
    "Ts": Ts 
}

torch.save(dataDict, ptFilePath)
print("\nData saved to", ptFilePath)

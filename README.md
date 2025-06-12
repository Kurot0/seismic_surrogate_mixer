# Seismic Ground Motion Prediction using Deep Learning-based Surrogate Models

This is a temporary project description.

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Training and Evaluation](#training-and-evaluation)
    - [Example Results](#example-results)
---

## Requirements

- Python 3.9

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:Kurot0/seismic_surrogate_mixer.git
cd seismic_surrogate_mixer
```

### 2. Install required Python packages

```bash
pip install -r requirements.txt
```

### 3 Download the data

Please download and extract the following data sets, and place the extracted files under the `data/raw_data/`:

| Data set                              | Google Drive URL                                                                                                                                                                                                        |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Seismic ground‑motion simulation data | [https://drive.google.com/drive/u/2/folders/1GaLdsRwsOuoXumnlRe-drB-SfKlY9thV?dmr=1\&ec=wgc-drive-hero-goto](https://drive.google.com/drive/u/2/folders/1GaLdsRwsOuoXumnlRe-drB-SfKlY9thV?dmr=1&ec=wgc-drive-hero-goto) |
| Subsurface‑structure model data       | [https://drive.google.com/file/d/1DAiTbJEAKHoQi7zmTjyVBTaXFheSqVhs/view](https://drive.google.com/file/d/1DAiTbJEAKHoQi7zmTjyVBTaXFheSqVhs/view)   

---   

## Directory Structure

The overall directory structure of this project is as follows:
```text
seismic_surrogate_mixer/
├── config.yaml
│
├── data/
│   ├── raw_data/         # Raw seismic simulation & subsurface‑structure data
│   ├── prep_data/        # Pre‑processed data
│   ├── exp_data/         # Experimental data (created by scripts below)
│   └── result/           # Trained models & result files
│
├── misc/                 # Utility / preprocessing scripts
│   ├── csv2pt.py
│   ├── data2pkl.py
│   ├── make_label.py
│   ├── make_mask.py
│   ├── modelinfo.py
│   ├── pkl2pt.py
│   ├── pt2pt.py
│   └── visualize.py
│
└── src/                  # Core source code
    ├── crossValidation.py
    ├── evaluate.py
    ├── inference.py
    ├── ssimloss.py
    ├── train.py
    └── models/
```

---

## Usage

### Data Preparation

The following preprocessing scripts should be executed in the following order to prepare the training and evaluation data:

```bash
# 1️. Create ocean mask images
python misc/make_mask.py

# 2️. Create labels for data splitting (for cross-validation and zero-shot learning)
python misc/make_label.py

# 3️. Preprocess and split seismic ground motion simulation data
python misc/data2pkl.py
python misc/pkl2pt.py

# 4️. Preprocess subsurface structure model data
python misc/csv2pt.py
python misc/pt2pt.py
```

- Multi‑shot learning data:  `data/exp_data/cv_data/`
- Zero‑shot learning data: `data/exp_data/cv_data_zeroshot/`

### Training and Inference

Experiment settings such as model architecture and hyperparameters are defined in the `config.yaml` file.

To run training and inference with cross-validation:

```bash
python src/crossValidation.py
```

- Trained models and experiment results will be saved in: `data/result/`

### Example Results

- **Loss Curves (Training / Validation)**

![Loss Curves](./img/loss_curve_example.png)

- **Quantitative Evaluation**
```
Individual Results: (31.784717, 0.937626), (31.973283, 0.943903), (32.270648, 0.945620), (31.583992, 0.943013), (31.749764, 0.947818), (31.664950, 0.946627), (32.522364, 0.947405), (32.312085, 0.945339), (33.213400, 0.947383), (32.790078, 0.940169)
Combined PSNR: 32.18652820587158
Combined SSIM: 0.9444904372096061

Sv03:
  Individual Results: (30.636494, 0.923294), (30.706364, 0.928255), (31.106222, 0.930649), (30.682285, 0.933032), (30.859184, 0.936080), (30.725756, 0.935370), (31.746016, 0.935299), (31.323568, 0.931271), (32.170872, 0.933386), (31.484701, 0.923006)
  Combined PSNR: 31.144146156311034
  Combined SSIM: 0.9309643149375916

Sv05:
  Individual Results: (31.286421, 0.934916), (31.255421, 0.942220), (32.061081, 0.944678), (30.661140, 0.939027), (30.627058, 0.942774), (31.012190, 0.943326), (31.655066, 0.943736), (31.402273, 0.943426), (32.678673, 0.945170), (32.481041, 0.940300)
  Combined PSNR: 31.512036323547363
  Combined SSIM: 0.9419572591781616

Sv07:
  Individual Results: (31.868298, 0.942913), (32.264381, 0.950159), (32.111015, 0.949583), (31.620989, 0.945492), (31.891890, 0.951019), (32.030720, 0.950552), (32.684074, 0.951485), (32.480339, 0.950285), (33.588604, 0.953932), (33.036362, 0.947344)
  Combined PSNR: 32.35766716003418
  Combined SSIM: 0.949276226758957

Sv10:
  Individual Results: (33.347656, 0.949383), (33.666965, 0.954979), (33.804276, 0.957572), (33.371555, 0.954502), (33.620926, 0.961399), (32.891136, 0.957260), (34.004299, 0.959100), (34.042160, 0.956377), (34.415451, 0.957042), (34.158207, 0.950027)
  Combined PSNR: 33.73226318359375
  Combined SSIM: 0.9557639479637146

Individual Results: 49,76,62,65,85,81,71,71,66,56
Average Best Epoch: 68.2

Total Training time: 4186.97 seconds
Total Inference time: 0.8568 seconds
Avarage Inference time: 1.3644 milliseconds
```

### Visualization

You can also perform qualitative evaluation and visualization of the prediction results by running:
```bash
python misc/visualize.py
```

- **Seismic Ground Motion Images (Ground truth | Prediction)**

| Ground Truth | Prediction |
|:------------:|:----------:|
| <img src="./img/seismic_gt.png" width="300"/> | <img src="./img/seismic_pred.png" width="300"/> |

- **Scatter Plots (Ground truth | Prediction)**

| Ground Truth | Prediction |
|:------------:|:----------:|
| <img src="./img/scatter_gt.png" width="300"/> | <img src="./img/scatter_pred.png" width="300"/> |


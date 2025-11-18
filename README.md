# Waste Type Classification for Smart Recycling

## Project overview
Classifies waste images into five categories: Plastic, Glass, Paper, Metal, Organic.
This repository includes training scripts, a Streamlit demo app, and utilities.

## Dataset
This project uses the "Waste Classification Data" from Kaggle:
https://www.kaggle.com/datasets/techsash/waste-classification-data

Download and extract the dataset to the `dataset/` folder with the following structure:

dataset/
├── train/
│   ├── plastic/
│   ├── glass/
│   ├── paper/
│   ├── metal/
│   └── organic/
└── test/
    ├── plastic/
    ├── glass/
    ├── paper/
    ├── metal/
    └── organic/

If you cannot find the train/test split, create it by splitting images into train and validation folders.

## Quick start

1. Create a Python virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate       # Unix / Mac
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

2. Train the model (will save to models/waste_classifier.h5):
```bash
python train_model.py --dataset_dir dataset --epochs 20 --batch_size 32 --img_size 224
```

3. Run the Streamlit demo app:
```bash
streamlit run app.py
```

## What is included
- `train_model.py` : Train a MobileNetV2 transfer learning model or a simple CNN fallback.
- `app.py` : Streamlit web app for uploading an image and viewing predictions.
- `sample_predict.py` : CLI script to run prediction on a single image.
- `requirements.txt` : Python packages required.
- `README.md` : This file.

## Notes
- The repository does NOT include the dataset due to size and licensing.
- Expect training to take significant time if GPU is not available. Use Google Colab if needed.


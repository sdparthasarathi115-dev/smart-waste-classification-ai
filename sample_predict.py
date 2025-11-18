# Simple CLI to predict a single image using saved model
import sys
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json, os

MODEL_PATH = 'models/waste_classifier.h5'
LABELS_PATH = 'models/labels.json'

def load_labels():
    if os.path.exists(LABELS_PATH):
        return json.load(open(LABELS_PATH))
    # fallback
    return {0:'glass',1:'metal',2:'paper',3:'plastic',4:'organic'}

def predict(img_path):
    model = load_model(MODEL_PATH)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224,224))
    x = np.array(img)/255.0
    x = np.expand_dims(x,0)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    labels = load_labels()
    print('Predicted:', labels.get(str(idx), labels.get(idx, str(idx))), 'Confidence:', preds[idx])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python sample_predict.py path/to/image.jpg')
        sys.exit(1)
    predict(sys.argv[1])

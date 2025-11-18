# Small utilities to save and label mapping
import json, os

def save_labels(generator, out_path='models/labels.json'):
    # generator.class_indices maps label->index, we invert to index->label (string keys)
    idx2label = {str(v):k for k,v in generator.class_indices.items()}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,'w') as f:
        json.dump(idx2label, f)
    print('Saved label mapping to', out_path)

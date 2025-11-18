import argparse
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from utils import save_labels

def build_mobilenet(input_shape, num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)

    # Freeze most layers for faster CPU training
    for layer in base.layers:
        layer.trainable = False
    return model

def build_simple_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def plot_history(history, out_dir):
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

def main(args):
    dataset_dir = args.dataset_dir
    train_dir = os.path.join(dataset_dir, 'train')

    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size

    classes = sorted(os.listdir(train_dir))
    num_classes = len(classes)
    print('Detected classes:', classes)

    # Lighter augmentation for speed
    train_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    input_shape = (args.img_size, args.img_size, 3)

    # Use smaller model by default for faster training
    if args.model == 'mobilenet':
        model = build_mobilenet(input_shape, num_classes)
    else:
        model = build_simple_cnn(input_shape, num_classes)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint('models/waste_classifier.h5', save_best_only=True, monitor='val_accuracy', mode='max')

    # Reduce epochs and image size for faster run
    history = model.fit(
        train_flow,
        epochs=args.epochs,
        validation_data=val_flow,
        callbacks=[checkpoint],
        verbose=1
    )

    os.makedirs('outputs', exist_ok=True)
    plot_history(history, 'outputs')
    save_labels(train_flow, out_path='models/labels.json')
    print('✅ Training complete. Best model saved to models/waste_classifier.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, default=8)        # ↓ fewer epochs
    parser.add_argument('--batch_size', type=int, default=16)   # ↓ smaller batches
    parser.add_argument('--img_size', type=int, default=128)    # ↓ smaller images
    parser.add_argument('--model', type=str, default='simple', choices=['mobilenet','simple'], help='Model type')
    args = parser.parse_args()
    main(args)

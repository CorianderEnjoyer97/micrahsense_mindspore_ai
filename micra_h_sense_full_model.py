import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
import csv
import random

ms.set_context(device_target="CPU")

# Copy the dataset's path file
CSV_FILE_PATH = "/Users/taokiensam/.gemini/antigravity/scratch/micra_h_sense_dataset.csv"

# The 8-class mapping
EMOTION_MAP = {
    "Anger": 0,
    "Happiness": 1,
    "Depression": 2,
    "Sadness": 3,
    "Anxiety": 4,
    "Acute_Stress": 5,
    "Fear_Panic": 6,
    "Calm": 7
}

REVERSE_MAP = {v: k for k, v in EMOTION_MAP.items()}

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    X = []
    y = []
    
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Binary encode sex (M=0, F=1)
                sex_encoded = 1.0 if row['sex'] == 'F' else 0.0
                
                # Extract all 23 available numerical/encoded features
                features = [
                    float(row['age']),
                    sex_encoded,
                    float(row['bmi']),
                    float(row['cortisol_nmolL']),
                    float(row['epinephrine_pgmL']),
                    float(row['norepinephrine_pgmL']),
                    float(row['ne_epi_ratio']),
                    float(row['epi_ne_ratio']),
                    float(row['serotonin_ngmL']),
                    float(row['dopamine_pgmL']),
                    float(row['oxytocin_pgmL']),
                    float(row['bdnf_ngmL']),
                    float(row['acth_pgmL']),
                    float(row['il6_pgmL']),
                    float(row['gaba_nmolmL']),
                    float(row['prolactin_ngmL']),
                    float(row['insulin_uIUmL']),
                    float(row['leptin_ngmL']),
                    float(row['testosterone_ngdL']),
                    float(row['estradiol_pgmL']),
                    float(row['progesterone_ngmL']),
                    float(row['vasopressin_pgmL']),
                    float(row['melatonin_pgmL'])
                ]
                
                class_idx = EMOTION_MAP[row['mental_health_state']]
                
                X.append(features)
                y.append(class_idx)
            except Exception as e:
                pass # Skip malformed rows if any

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Shuffle dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Simple Min-Max normalization for features to keep training stable
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    range_diff = X_max - X_min
    range_diff[range_diff == 0] = 1.0 
    
    X_norm = (X - X_min) / range_diff

    # Split 80/20 train/test
    split_idx = int(len(X_norm) * 0.8)
    
    X_train = X_norm[:split_idx]
    y_train = y[:split_idx]
    
    X_test = X_norm[split_idx:]
    y_test = y[split_idx:]
    
    return Tensor(X_train), Tensor(y_train), Tensor(X_test), Tensor(y_test), X_min, range_diff

# --- Define Upgraded Model ---
class MicraSenseNetV2(nn.Cell):
    def __init__(self, input_dim=23, num_classes=8):
        super(MicraSenseNetV2, self).__init__()
        self.fc1 = nn.Dense(input_dim, 128, activation='relu')
        self.fc2 = nn.Dense(128, 64, activation='relu')
        self.fc3 = nn.Dense(64, 32, activation='relu')
        self.fc4 = nn.Dense(32, 16, activation='relu')
        self.output = nn.Dense(16, num_classes)

    def construct(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.output(x)
        return x

def train_model():
    X_train, y_train, X_test, y_test, X_min, range_diff = load_and_preprocess_data(CSV_FILE_PATH)

    net = MicraSenseNetV2()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=0.005)

    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    print(f"Training MicraSenseNetV2 on {len(X_train)} samples...")
    epochs = 400
    for epoch in range(epochs):
        loss = train_step(X_train, y_train)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} - Train Loss: {float(loss):.4f}")

    print("Training Complete!\n")
    return net, X_test, y_test, X_min, range_diff

def test_and_evaluate(net, X_test, y_test):
    print("==================================================")
    print("= MICRA H SENSE V2 - EVALUATION & SHOWCASE =")
    print("==================================================")
    
    logits = net(X_test)
    predictions = logits.argmax(axis=1).asnumpy()
    labels = y_test.asnumpy()
    
    accuracy = (predictions == labels).mean() * 100
    print(f"Overall Test Set Accuracy: {accuracy:.1f}%\n")
    
    print("--- Random Patient Showcase ---")
    
    # Pick 3 random patients from the test set
    indices = random.sample(range(len(labels)), min(3, len(labels)))
    for i in indices:
        pred_label = REVERSE_MAP[predictions[i]]
        true_label = REVERSE_MAP[labels[i]]
        
        confidence = float(np.max(nn.Softmax()(logits[i:i+1]).asnumpy())) * 100
        
        print(f"Random Test Sample {i}:")
        print(f" -> AI Predicted: {pred_label} ({confidence:.1f}% confidence)")
        print(f" -> Real Medical State: {true_label}")
        if pred_label == true_label:
            print(" -> Result: SUCCESS ✅\n")
        else:
            print(" -> Result: MISMATCH ❌\n")

if __name__ == "__main__":
    trained_net, x_test, y_test, x_min, r_diff = train_model()
    test_and_evaluate(trained_net, x_test, y_test)

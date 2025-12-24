import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from scipy.stats import pearsonr, spearmanr

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.arque_core import ARQUEModel

# --- CONFIGURATION ---
# UPDATE THIS PATH TO YOUR LOCAL DATA FOLDER
DATA_ROOT = "../data/LIVE/" 
MODELS_ROOT = "../models/"

JSON_FILE = os.path.join(MODELS_ROOT, "trained_models_LIVE.json")
PKL_CLF = os.path.join(MODELS_ROOT, "classifier_hybrid.pkl")
PKL_SVR = os.path.join(MODELS_ROOT, "svr_specialists_pro.pkl")
TYPES = ['gblur', 'wn', 'jpeg', 'jp2k', 'fastfading']

def load_live_dataset(root, types):
    try:
        dmos = sio.loadmat(os.path.join(root, 'dmos.mat'))['dmos'].flatten()
    except FileNotFoundError:
        print(f"Error: dmos.mat not found in {root}. Please download the LIVE dataset.")
        return []

    # Metadata specific to LIVE Release 2
    offs = {'jp2k':0, 'jpeg':227, 'wn':460, 'gblur':634, 'fastfading':808}
    lens = {'jp2k':227, 'jpeg':233, 'wn':174, 'gblur':174, 'fastfading':174}
    
    dataset = []
    for t in types:
        base = os.path.join(root, t)
        for i in range(lens[t]):
            fname = f"img{i+1}.bmp"
            if not os.path.exists(os.path.join(base, fname)): 
                fname = f"img{i+1}.jpg"
            
            full_path = os.path.join(base, fname)
            if os.path.exists(full_path):
                dataset.append({
                    'path': full_path, 
                    'type': t, 
                    'dmos': dmos[offs[t] + i]
                })
    return dataset

if __name__ == "__main__":
    print("--- REPRODUCING LIVE RESULTS (TABLE 1 & FIGURES) ---")
    
    # 1. Load Model
    model = ARQUEModel(JSON_FILE, PKL_CLF, PKL_SVR)
    
    # 2. Load Data
    dataset = load_live_dataset(DATA_ROOT, TYPES)
    if not dataset: exit()
    print(f"Loaded {len(dataset)} images.")

    # 3. Run Inference
    y_true_type, y_pred_type = [], []
    y_true_dmos, y_pred_dmos = [], []

    import cv2
    for i, item in enumerate(dataset):
        img = cv2.imread(item['path'], cv2.IMREAD_GRAYSCALE)
        pred_type, pred_dmos = model.predict(img)
        
        y_true_type.append(item['type'])
        y_pred_type.append(pred_type)
        y_true_dmos.append(item['dmos'])
        y_pred_dmos.append(pred_dmos)
        
        if i % 50 == 0: print(f"Processing {i}/{len(dataset)}...", end="\r")

    # 4. Calculate Metrics
    acc = accuracy_score(y_true_type, y_pred_type)
    rmse = np.sqrt(mean_squared_error(y_true_dmos, y_pred_dmos))
    plcc, _ = pearsonr(y_true_dmos, y_pred_dmos)
    srocc, _ = spearmanr(y_true_dmos, y_pred_dmos)

    print("\n" + "="*50)
    print(f"ARQUE PERFORMANCE ON LIVE")
    print("="*50)
    print(f"Classification Accuracy: {acc*100:.2f}%")
    print(f"PLCC (Linearity):      {plcc:.4f} (Paper: 0.954)")
    print(f"SROCC (Monotonicity):  {srocc:.4f}")
    print(f"RMSE (Error):          {rmse:.4f}")
    
    # 5. Generate Plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion Matrix
    labels = sorted(TYPES)
    cm = confusion_matrix(y_true_type, y_pred_type, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax[0], xticklabels=labels, yticklabels=labels)
    ax[0].set_title("Distortion Classification Accuracy")
    
    # Scatter Plot
    colors = {'gblur': 'red', 'wn': 'green', 'jpeg': 'blue', 'jp2k': 'orange', 'fastfading': 'purple'}
    c_list = [colors[t] for t in y_true_type]
    ax[1].scatter(y_true_dmos, y_pred_dmos, c=c_list, alpha=0.5, s=15)
    ax[1].plot([0, 100], [0, 100], 'k--', lw=1) # Identity line
    ax[1].set_xlabel("Ground Truth DMOS")
    ax[1].set_ylabel("Predicted DMOS")
    ax[1].set_title(f"Prediction Correlation (PLCC={plcc:.3f})")
    
    plt.tight_layout()
    plt.savefig("arque_results.png")
    print("\nPlot saved to 'arque_results.png'")
    plt.show()
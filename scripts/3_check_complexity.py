import os
import sys
import joblib
import numpy as np

MODELS_ROOT = "../models/"
PKL_SVR = os.path.join(MODELS_ROOT, "svr_specialists_pro.pkl")

if __name__ == "__main__":
    print("--- COMPLEXITY AUDIT (TABLE 3 REPRODUCTION) ---")
    
    if not os.path.exists(PKL_SVR):
        print("Model file not found.")
        exit()

    svrs = joblib.load(PKL_SVR)
    counts = []
    
    print(f"{'Expert':<15} | {'Active Support Vectors'}")
    print("-" * 40)
    
    for t, data in svrs.items():
        # Access the SVR inside the pipeline steps
        model = data['model'].steps[1][1] 
        n_sv = model.support_vectors_.shape[0]
        counts.append(n_sv)
        print(f"{t:<15} | {n_sv}")
        
    avg_active = np.mean(counts)
    
    # Baseline BRISQUE reference (from paper/experiments)
    brisque_sv = 632 
    
    print("-" * 40)
    print(f"ARQUE Avg Active SVs: {avg_active:.1f}")
    print(f"BRISQUE Active SVs:   {brisque_sv}")
    print(f"Reduction:            {100 * (1 - avg_active/brisque_sv):.1f}%")
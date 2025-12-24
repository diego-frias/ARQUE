import os
import sys
import numpy as np
import pandas as pd
import cv2
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Adiciona a pasta raiz ao path para importar o src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.arque_core import ARQUEModel

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
# Caminho relativo para a pasta de dados
DATA_ROOT = os.path.join(os.path.dirname(__file__), '../data/CSIQ/')

TYPES = ['wn', 'gblur', 'jpeg', 'jp2k'] # Tipos suportados no CSIQ

# Grid de busca para auto-calibração (Rápido e Eficiente)
GRID_ALPHAS = [0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
GRID_BETAS  = [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Instância "dummy" do ARQUE apenas para usar as funções matemáticas (DRY)
core = ARQUEModel.__new__(ARQUEModel) 
core.kh = np.array([[1, -2, 1]])
core.kv = np.array([[1], [-2], [1]])

# ==============================================================================
# 1. FUNÇÕES AUXILIARES (BRISQUE & OTIMIZAÇÃO)
# ==============================================================================
def compute_mscn(img):
    C = 1
    mu = cv2.GaussianBlur(img, (7, 7), 1.166)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(img*img, (7, 7), 1.166) - mu*mu))
    return (img - mu) / (sigma + C)

def extract_brisque_simple(img):
    """Implementação simplificada do BRISQUE para baseline rápido no CSIQ"""
    feats = []
    for s in [1, 0.5]:
        if s!=1: img = cv2.resize(img, (0,0), fx=s, fy=s)
        mscn = compute_mscn(img)
        feats.extend([0.5, np.mean(mscn**2)]) 
        
        # Pairwise simples (Mean, Std)
        for shift in [[0,1], [1,0], [1,1], [-1,1]]:
            if shift==[0,1]: vec=(mscn[:,:-1]*mscn[:,1:]).flatten()
            elif shift==[1,0]: vec=(mscn[:-1,:]*mscn[1:,:]).flatten()
            elif shift==[1,1]: vec=(mscn[:-1,:-1]*mscn[1:,1:]).flatten()
            else: vec=(mscn[1:,:-1]*mscn[:-1,1:]).flatten()
            feats.extend([np.mean(vec), np.std(vec)**2, 0.5, 0])
    return np.array(feats)

def optimize_parameters(data_list):
    """
    Recebe lista de dicts {'ch', 'cv', 'dmos'} e encontra o melhor Alpha/Beta
    Maximiza Spearman (SROCC) no conjunto de TREINO.
    """
    if len(data_list) < 5: return 1.0, 1.0 # Fallback
    
    y_true = [d['dmos'] for d in data_list]
    best_rho = -1
    best_cfg = (1.0, 1.0)
    
    # Grid Search
    for a in GRID_ALPHAS:
        for b in GRID_BETAS:
            # Usa a matemática do Core para calcular o ATR
            y_pred = [core._calc_atr(d['ch'], d['cv'], a, b) for d in data_list]
            
            if np.std(y_pred) < 1e-9: rho = 0
            else: 
                r = spearmanr(y_true, y_pred)[0]
                rho = abs(r) if not np.isnan(r) else 0
                
            if rho > best_rho:
                best_rho = rho
                best_cfg = (a, b)
                
    return best_cfg

def load_csiq(root):
    print(f">>> Reading CSIQ from: {root}")
    csv_path = os.path.join(root, 'csiq_scores_by_image.csv')
    try: 
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: 'csiq_scores_by_image.csv' not found.")
        return []
        
    df.columns = [c.strip() for c in df.columns]
    
    target_map = {'noise': 'wn', 'awgn': 'wn', 'blur': 'gblur', 'jpeg': 'jpeg', 
                  'jpeg2000': 'jp2k', 'jpeg 2000': 'jp2k'}
    folder_map = {'noise': 'awgn', 'awgn': 'awgn', 'blur': 'blur', 'jpeg': 'jpeg', 
                  'jpeg2000': 'jpeg2000', 'jpeg 2000': 'jpeg2000'}
    
    data = []
    count = 0
    
    # Pre-check folders
    img_root = os.path.join(root, 'dst_imgs')
    if not os.path.exists(img_root):
        print(f"Error: 'dst_imgs' folder not found in {root}")
        return []

    for idx, row in df.iterrows():
        dtype = str(row.get('dst_type', '')).strip().lower()
        if dtype in target_map:
            my_type = target_map[dtype]
            img_b = str(row['image']).strip()
            lev = str(row['dst_lev']).strip()
            
            ftag = dtype if 'noise' not in dtype and 'awgn' not in dtype else 'AWGN'
            if 'jpeg' in dtype and '2000' in dtype: ftag = 'jpeg2000'
            
            fname = f"{img_b}.{ftag}.{lev}.png"
            fdir = os.path.join(img_root, folder_map.get(dtype, dtype))
            
            # Tenta variações de nome de arquivo (comum no CSIQ)
            tries = [fname, fname.lower(), fname.upper(), f"{img_b}.{ftag.lower()}.{lev}.png"]
            
            found = None
            for fn in tries:
                path_try = os.path.join(fdir, fn)
                if os.path.exists(path_try):
                    found = path_try; break
            
            if found:
                img = cv2.imread(found, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Pré-calcula mapas para agilizar
                    ch, cv = core._calc_maps(img)
                    data.append({
                        'ch': ch, 'cv': cv, 'raw_img': img,
                        'type': my_type, 'dmos': float(row['dmos'])
                    })
                    count += 1
    print(f"Total Images Loaded: {count}")
    return data

def sanitize_features(X):
    """Substitui NaNs e Infs por 0."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

# ==============================================================================
# 2. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("--- REPRODUCING CSIQ BENCHMARK (TABLE 2) ---")
    
    # A. Carregar
    dataset = load_csiq(DATA_ROOT)
    if not dataset: 
        print("Please ensure the CSIQ dataset is in the 'data/CSIQ' folder.")
        exit()
    
    # B. Split 80/20 (Estratificado)
    y_types = [d['type'] for d in dataset]
    indices = np.arange(len(dataset))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, stratify=y_types, random_state=42)
    
    print(f"\nSplit: {len(idx_train)} Train / {len(idx_test)} Test")
    
    # C. AUTO-CALIBRAÇÃO
    print("\n[ARQUE] Auto-calibrating parameters on TRAINING set...")
    OPTIMIZED_PARAMS = {}
    
    for t in TYPES:
        train_subset = [dataset[i] for i in idx_train if dataset[i]['type'] == t]
        
        if train_subset:
            print(f"  Optimizing {t.upper()} ({len(train_subset)} imgs)... ", end="")
            a, b = optimize_parameters(train_subset)
            OPTIMIZED_PARAMS[t] = {'alpha': a, 'beta': b}
            print(f"Alpha: {a}, Beta: {b}")
        else:
            OPTIMIZED_PARAMS[t] = {'alpha': 1.0, 'beta': 1.0}

    # D. EXTRAÇÃO DE FEATURES
    print("\n[GENERAL] Extracting feature vectors...")
    X_arque_clf = []
    y_all = []
    
    # Containers para o BRISQUE
    ALL_X_BRISQUE = []
    
    for i in range(len(dataset)):
        d = dataset[i]
        ch, cv = d['ch'], d['cv']
        
        # 1. BRISQUE
        ALL_X_BRISQUE.append(extract_brisque_simple(d['raw_img']))
        
        # 2. ARQUE - Classificador (Usa todos os ATRs otimizados)
        atrs = []
        for t in TYPES:
            p = OPTIMIZED_PARAMS[t]
            atrs.append(core._calc_atr(ch, cv, p['alpha'], p['beta']))
        
        # Features NSS para classificação
        nss = core._calc_nss(ch, cv, mode='clf')
        X_arque_clf.append(atrs + nss)
        
        y_all.append(d['dmos'])
        
        if i%50==0: print(f"  {i}/{len(dataset)}", end="\r")

    ALL_X_BRISQUE = np.array(ALL_X_BRISQUE)
    X_arque_clf = np.array(X_arque_clf)
    y_all = np.array(y_all)
    
    # Sanitização
    ALL_X_BRISQUE = sanitize_features(ALL_X_BRISQUE)
    X_arque_clf = sanitize_features(X_arque_clf)
    
    # E. TREINAMENTO E AVALIAÇÃO
    
    # --- 1. BRISQUE Baseline ---
    print("\n\n>>> 1. BRISQUE Baseline (Training SVR...)")
    pipe_b = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
    grid_b = GridSearchCV(pipe_b, {'svr__C': [100, 1000, 5000], 'svr__gamma': [0.01, 0.1]}, cv=3, n_jobs=-1)
    grid_b.fit(ALL_X_BRISQUE[idx_train], y_all[idx_train])
    y_pred_b = grid_b.predict(ALL_X_BRISQUE[idx_test])
    
    pb = pearsonr(y_all[idx_test], y_pred_b)[0]
    rb = np.sqrt(mean_squared_error(y_all[idx_test], y_pred_b))
    print(f"  PLCC: {pb:.4f} | RMSE: {rb:.2f}")
    
    # --- 2. ARQUE (Auto-Calibrated) ---
    print("\n>>> 2. ARQUE (Training Hybrid System...)")
    
    # 2.1 Treinar Classificador (Gate)
    le = LabelEncoder()
    y_enc = le.fit_transform([dataset[i]['type'] for i in range(len(dataset))])
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_arque_clf[idx_train], y_enc[idx_train])
    
    # 2.2 Treinar SVRs Especialistas
    svrs = {}
    for t in TYPES:
        idxs_t = [i for i in idx_train if dataset[i]['type'] == t]
        
        if idxs_t:
            X_t = []
            p = OPTIMIZED_PARAMS[t]
            for i in idxs_t:
                d = dataset[i]
                atr = core._calc_atr(d['ch'], d['cv'], p['alpha'], p['beta'])
                nss = core._calc_nss(d['ch'], d['cv'], mode='svr')
                X_t.append([atr] + nss)
            
            regr = make_pipeline(StandardScaler(), SVR(C=100, gamma='scale'))
            regr.fit(X_t, y_all[idxs_t])
            svrs[t] = regr
            
    # 2.3 Inferência no Teste (Soft Voting)
    y_pred_a = []
    for i in idx_test:
        d = dataset[i]
        
        # Probabilidades do Classificador
        probs = clf.predict_proba(X_arque_clf[i].reshape(1,-1))[0]
        val = 0.0
        
        for c_idx, prob in enumerate(probs):
            if prob < 0.01: continue
            c_name = le.inverse_transform([c_idx])[0]
            
            if c_name in svrs:
                p = OPTIMIZED_PARAMS[c_name]
                atr = core._calc_atr(d['ch'], d['cv'], p['alpha'], p['beta'])
                nss = core._calc_nss(d['ch'], d['cv'], mode='svr')
                
                pred = svrs[c_name].predict(np.array([[atr] + nss]))[0]
                val += prob * pred
            else:
                val += prob * np.mean(y_all[idx_train])
        y_pred_a.append(val)
        
    pa = pearsonr(y_all[idx_test], y_pred_a)[0]
    ra = np.sqrt(mean_squared_error(y_all[idx_test], y_pred_a))
    
    print("\n" + "="*60)
    print(f"RESULTS ON CSIQ (INTRA-DATASET TEST)")
    print("="*60)
    print(f"{'METHOD':<15} | {'PLCC':<10} | {'RMSE':<10}")
    print("-" * 45)
    print(f"{'BRISQUE':<15} | {pb:.4f}      | {rb:.2f}")
    print(f"{'ARQUE':<15} | {pa:.4f}      | {ra:.2f}")
    print("="*60)
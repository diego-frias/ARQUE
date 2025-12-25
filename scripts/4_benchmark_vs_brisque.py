import cv2
import numpy as np
import scipy.io as sio
from scipy import ndimage
from scipy.special import gamma
from scipy.stats import kurtosis, skew, pearsonr, spearmanr
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import os
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. NÚCLEO MATEMÁTICO DO BRISQUE (Completo - 36 Features)
# ==============================================================================
def estimate_ggd_param(vec):
    gam = np.arange(0.2, 10.0, 0.001)
    r_gam = (gamma(1/gam) * gamma(3/gam)) / (gamma(2/gam)**2)
    sigma_sq = np.mean(vec**2)
    E = np.mean(np.abs(vec))
    rho = sigma_sq / (E**2)
    array_position = np.argmin(np.abs(rho - r_gam))
    return gam[array_position], sigma_sq

def estimate_aggd_param(vec):
    gam = np.arange(0.2, 10.0, 0.001)
    r_gam = ((gamma(2/gam))**2) / (gamma(1/gam) * gamma(3/gam))
    left, right = vec[vec < 0], vec[vec >= 0]
    sigma_l_sq = np.mean(left**2) if len(left) > 0 else 0
    sigma_r_sq = np.mean(right**2) if len(right) > 0 else 0
    sigma_l, sigma_r = np.sqrt(sigma_l_sq), np.sqrt(sigma_r_sq)
    gamma_hat = sigma_l / sigma_r if sigma_r > 1e-5 else 0
    E_abs = np.mean(np.abs(vec))
    E_sq = np.mean(vec**2)
    rho = (E_abs**2) / E_sq if E_sq > 1e-9 else 0
    array_position = np.argmin(np.abs(rho - r_gam))
    alpha = gam[array_position]
    const = np.sqrt(gamma(1/alpha) / gamma(3/alpha))
    mean = (sigma_r - sigma_l) * (gamma(2/alpha) / gamma(1/alpha)) * const
    return alpha, mean, sigma_l_sq, sigma_r_sq

def compute_mscn(img):
    C = 1
    mu = cv2.GaussianBlur(img, (7, 7), 1.166)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(img*img, (7, 7), 1.166) - mu*mu))
    return (img - mu) / (sigma + C)

def extract_brisque_features_full(img_gray):
    img = img_gray.astype(np.float32) / 255.0
    features = []
    for scale in [1, 0.5]:
        if scale != 1:
            img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mscn = compute_mscn(img)
        alpha, sigma_sq = estimate_ggd_param(mscn.flatten())
        features.extend([alpha, sigma_sq])
        shifts = [[0,1], [1,0], [1,1], [-1,1]]
        for shift in shifts:
            if shift == [0,1]: vec = (mscn[:, :-1] * mscn[:, 1:]).flatten()
            elif shift == [1,0]: vec = (mscn[:-1, :] * mscn[1:, :]).flatten()
            elif shift == [1,1]: vec = (mscn[:-1, :-1] * mscn[1:, 1:]).flatten()
            elif shift == [-1,1]: vec = (mscn[1:, :-1] * mscn[:-1, 1:]).flatten()
            alpha_aggd, mean, sigma_l, sigma_r = estimate_aggd_param(vec)
            features.extend([alpha_aggd, mean, sigma_l, sigma_r])
    return np.array(features)

# ==============================================================================
# 2. SISTEMA ARQUE (Atualizado com SOFT VOTING)
# ==============================================================================
class ARQUE_SoftVoting_System:
    def __init__(self, json_path, clf_pkl_path, svr_pkl_path):
        with open(json_path, 'r') as f: self.reg_data = json.load(f)
        clf_pack = joblib.load(clf_pkl_path)
        self.clf = clf_pack['classifier']
        self.le = clf_pack['label_encoder']
        self.spec_order = clf_pack['specialist_order']
        self.svrs = joblib.load(svr_pkl_path)
        self.kh = np.array([[1,-2,1]])
        self.kv = np.array([[1],[-2],[1]])

    def _calc_maps(self, img):
        img_int = img.astype(np.int16)
        ch = np.log1p(np.abs(ndimage.convolve(img_int, self.kh, mode='reflect')))
        cv = np.log1p(np.abs(ndimage.convolve(img_int, self.kv, mode='reflect')))
        return ch, cv

    def _calc_atr(self, ch, cv, alpha, beta):
        lstd_h, lstd_v = np.std(ch), np.std(cv)
        if lstd_h<1e-9: lstd_h=1e-9
        if lstd_v<1e-9: lstd_v=1e-9
        mh = (cv >= beta*lstd_v) & (ch < alpha*lstd_h)
        mv = (ch >= beta*lstd_h) & (cv < alpha*lstd_v)
        return (np.sum(mh)+np.sum(mv))/2.0/ch.size

    def _calc_nss_clf(self, ch, cv):
        f = []
        for m in [ch, cv]: f.extend([kurtosis(m.flatten()), skew(m.flatten()), np.std(m.flatten())])
        for m in [ch, cv]:
            hp, vp = m[:, :-1]*m[:, 1:], m[:-1, :]*m[1:, :]
            for p in [hp, vp]: f.extend([np.mean(p), np.std(p)])
        return f

    def _calc_nss_svr(self, ch, cv):
        f = []
        for m in [ch, cv]:
            flat = m.flatten()
            f.extend([kurtosis(flat), skew(flat), np.std(flat), np.mean(flat)])
        for m in [ch, cv]:
            hp = m[:, :-1]*m[:, 1:]
            f.extend([np.mean(hp), np.std(hp)])
        return f

    def predict_score(self, img):
        """Retorna apenas o DMOS (float) usando Soft Voting"""
        ch, cv = self._calc_maps(img)
        lstd_h, lstd_v = np.std(ch), np.std(cv)
        if lstd_h<1e-9: lstd_h=1e-9
        if lstd_v<1e-9: lstd_v=1e-9

        # 1. Features para Classificação
        atr_vec_clf = []
        for spec in self.spec_order:
            p = self.reg_data['models'][spec]
            atr = self._calc_atr(ch, cv, p['alpha'], p['beta'])
            atr_vec_clf.append(atr)
            
        feat_clf = np.array(atr_vec_clf + self._calc_nss_clf(ch, cv)).reshape(1, -1)
        
        # 2. Soft Voting (Otimizado)
        probs = self.clf.predict_proba(feat_clf)[0]
        weighted_dmos = 0.0
        
        for i, prob in enumerate(probs):
            if prob < 0.01: continue # Otimização para ignorar classes irrelevantes
            
            c_name = self.le.inverse_transform([i])[0]
            if c_name in self.svrs:
                model_data = self.svrs[c_name]
                # Features específicas do SVR
                atr = self._calc_atr(ch, cv, model_data['alpha'], model_data['beta'])
                nss = self._calc_nss_svr(ch, cv)
                feat_svr = np.array([atr] + nss).reshape(1, -1)
                
                pred = model_data['model'].predict(feat_svr)[0]
                weighted_dmos += prob * pred
            else:
                # Fallback seguro (média global aproximada se modelo não existir)
                weighted_dmos += prob * 50.0 
                
        return weighted_dmos

# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    
	# --- CONFIGURAÇÃO AUTOMÁTICA DE CAMINHOS ---
    # Detecta onde este script está rodando
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define a raiz do projeto (sobe um nível: de 'scripts/' para 'root/')
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Adiciona 'src' ao Python Path para importar o arque_core
    sys.path.append(project_root)
    from src.arque_core import ARQUEModel # Importa agora que o path está certo

    # Define caminhos relativos para Dados e Modelos
    # Estrutura esperada:
    # root/
    #   ├── data/
    #   │   └── LIVE/
    #   ├── models/
    #   │   ├── trained_models_LIVE.json
    #   │   └── ...
    #   └── scripts/ (onde este arquivo está)
    
    PATH_LIVE = os.path.join(project_root, 'data', 'LIVE')
    JSON = os.path.join(project_root, 'models', 'trained_models_LIVE.json')
    PKL_CLF = os.path.join(project_root, 'models', 'classifier_hybrid.pkl')
    PKL_SVR = os.path.join(project_root, 'models', 'svr_specialists_pro.pkl')
    
    TYPES = ['gblur', 'wn', 'jpeg', 'jp2k', 'fastfading']
    
    # Verifica se os dados existem antes de começar
    if not os.path.exists(PATH_LIVE):
        print(f"ERRO CRÍTICO: Pasta de dados não encontrada em:\n{PATH_LIVE}")
        print("Por favor, baixe o LIVE Dataset e coloque na pasta 'data/LIVE'.")
        exit()
        
    if not os.path.exists(PKL_CLF):
        print(f"ERRO: Modelos não encontrados em 'models/'. Verifique o download do repositório.")
        exit()
		
    # 1. Carregar LIVE Dataset
    print(">>> Carregando LIVE Dataset...")
    try:
        dmos = sio.loadmat(os.path.join(PATH_LIVE, 'dmos.mat'))['dmos'].flatten()
    except FileNotFoundError:
        print("ERRO: dmos.mat não encontrado. Verifique o caminho.")
        exit()

    offs = {'jp2k':0, 'jpeg':227, 'wn':460, 'gblur':634, 'fastfading':808}
    lens = {'jp2k':227, 'jpeg':233, 'wn':174, 'gblur':174, 'fastfading':174}
    
    paths = []
    y_true = []
    y_types = [] # Para fazer o Stratified Split (Sorting Correto)

    for t in TYPES:
        base = os.path.join(PATH_LIVE, t)
        for i in range(lens[t]):
            fname = f"img{i+1}.bmp"
            if not os.path.exists(os.path.join(base, fname)): fname = f"img{i+1}.jpg"
            
            full_path = os.path.join(base, fname)
            if os.path.exists(full_path):
                paths.append(full_path)
                y_true.append(dmos[offs[t] + i])
                y_types.append(t)
                
    y_true = np.array(y_true)
    y_types = np.array(y_types)
    print(f"Total de imagens carregadas: {len(y_true)}")

    # 2. Extrair Features BRISQUE (Pesado - 36 Dimensões)
    print("\n>>> Extraindo Features BRISQUE (36D)...")
    X_brisque = []
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        X_brisque.append(extract_brisque_features_full(img))
        if i % 50 == 0: print(f"  {i}/{len(paths)}...", end="\r")
    X_brisque = np.array(X_brisque)
    
    # 3. SPLIT (SORTEIO CORRETO / STRATIFIED)
    # Garante que todos os tipos de distorção estejam representados igualmente
    print("\n\n>>> Realizando Split Estratificado (80/20)...")
    indices = np.arange(len(y_true))
    
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y_true, 
        test_size=0.2, 
        stratify=y_types,  # <--- AQUI ESTÁ O "SORTING CORRETO" DAS CLASSES
        random_state=42
    )
    
    # 4. AVALIAÇÃO BRISQUE (SVR Otimizado)
    print("\n>>> 1. Otimizando e Treinando BRISQUE...")
    param_grid = {
        'svr__C': [80, 100, 120],        # Focando ao redor de 100
        'svr__gamma': [0.8, 1.0, 1.2],   # Focando ao redor de 1.0
        'svr__epsilon': [0.5, 1.0, 1.5]  # Focando ao redor de 1.0
    }
    pipe_base = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
    grid = GridSearchCV(pipe_base, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_brisque[idx_train], y_train)
    
    print(f"  Melhores params BRISQUE: {grid.best_params_}")
    best_brisque = grid.best_estimator_
    y_pred_b = best_brisque.predict(X_brisque[idx_test])

    # 5. AVALIAÇÃO ARQUE (Soft Voting)
    print("\n>>> 2. Rodando ARQUE (Soft Voting)...")
    model_arque = ARQUE_SoftVoting_System(JSON, PKL_CLF, PKL_SVR)
    
    y_pred_a = []
    for idx in idx_test:
        img = cv2.imread(paths[idx], cv2.IMREAD_GRAYSCALE)
        # Usa o novo método com Soft Voting
        score = model_arque.predict_score(img)
        y_pred_a.append(score)
    y_pred_a = np.array(y_pred_a)

    # 6. RELATÓRIO FINAL
    def get_metrics(yr, yp):
        p, _ = pearsonr(yr, yp)
        s, _ = spearmanr(yr, yp)
        r = np.sqrt(mean_squared_error(yr, yp))
        return p, s, r

    pb, sb, rb = get_metrics(y_test, y_pred_b)
    pa, sa, ra = get_metrics(y_test, y_pred_a)

    print("\n" + "="*75)
    print(f"{'MÉTODO':<20} | {'PLCC':<10} | {'SROCC':<10} | {'RMSE':<10}")
    print("-" * 75)
    print(f"{'BRISQUE (Optimized)':<20} | {pb:.4f}     | {sb:.4f}     | {rb:.4f}")
    print(f"{'ARQUE (Soft Vote)':<20} | {pa:.4f}     | {sa:.4f}     | {ra:.4f}")
    print("="*75)
    
    # Destaque de Vantagem
    gain = 100 * (pa - pb) / pb
    print(f"\n>> Ganho de Performance do ARQUE: +{gain:.1f}% em Correlação Linear")
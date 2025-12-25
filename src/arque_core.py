import numpy as np
import cv2
import joblib
import json
from scipy import ndimage
from scipy.stats import kurtosis, skew

class ARQUEModel:
    def __init__(self, json_path, clf_pkl_path, svr_pkl_path):
        """
        Inicializa o modelo ARQUE carregando as configurações e os pesos treinados.
        """
        # Carrega dicionário com alphas/betas ótimos
        with open(json_path, 'r') as f: 
            self.reg_data = json.load(f)
        
        # Carrega o Classificador (Random Forest) e Label Encoder
        clf_pack = joblib.load(clf_pkl_path)
        self.clf = clf_pack['classifier']
        self.le = clf_pack['label_encoder']
        self.spec_order = clf_pack['specialist_order']
        
        # Carrega os SVRs Especialistas
        self.svrs = joblib.load(svr_pkl_path)
        
        # Kernels Laplacianos para extração de curvatura
        self.kh = np.array([[1, -2, 1]])
        self.kv = np.array([[1], [-2], [1]])

    def _calc_maps(self, img):
        """Calcula os mapas de curvatura logarítmica (Horizontal e Vertical)."""
        img_int = img.astype(np.int16)
        ch = np.log1p(np.abs(ndimage.convolve(img_int, self.kh, mode='reflect')))
        cv = np.log1p(np.abs(ndimage.convolve(img_int, self.kv, mode='reflect')))
        return ch, cv

    def _calc_atr(self, ch, cv, alpha, beta):
        """Calcula a métrica ATR baseada nos limiares paramétricos."""
        lstd_h, lstd_v = np.std(ch), np.std(cv)
        # Evita divisão por zero
        if lstd_h < 1e-9: lstd_h = 1e-9
        if lstd_v < 1e-9: lstd_v = 1e-9
        
        mask_h = (cv >= beta * lstd_v) & (ch < alpha * lstd_h)
        mask_v = (ch >= beta * lstd_h) & (cv < alpha * lstd_v)
        
        return (np.sum(mask_h) + np.sum(mask_v)) / 2.0 / ch.size

    def _calc_nss(self, ch, cv, mode='svr'):
        """
        Calcula estatísticas de cena natural (NSS).
        mode='clf': Features usadas pelo Classificador (compatível com treino).
        mode='svr': Features usadas pelos Especialistas (inclui Média dos mapas).
        """
        f = []
        # Estatísticas de primeira ordem (Histograma)
        for m in [ch, cv]:
            flat = m.flatten()
            f.extend([kurtosis(flat), skew(flat), np.std(flat)])
            if mode == 'svr':
                f.append(np.mean(flat)) # SVR usa a média, Classificador não
        
        # Estatísticas de segunda ordem (Pairwise Products)
        for m in [ch, cv]:
            hp = m[:, :-1] * m[:, 1:] # Horizontal Pair
            
            if mode == 'clf':
                vp = m[:-1, :] * m[1:, :] # Vertical Pair (apenas clf usa)
                for p in [hp, vp]: f.extend([np.mean(p), np.std(p)])
            else:
                f.extend([np.mean(hp), np.std(hp)]) # SVR usa apenas horizontal pair
        return f

    def predict(self, img):
        """
        Pipeline de Predição Completo (Soft Voting):
        1. Extrai mapas base.
        2. Classifica (obtém probabilidades).
        3. Pondera a saída dos especialistas.
        
        Retorna: (tipo_detectado_str, nota_dmos_float)
        """
        ch, cv = self._calc_maps(img)
        lstd_h, lstd_v = np.std(ch), np.std(cv)
        if lstd_h < 1e-9: lstd_h = 1e-9
        if lstd_v < 1e-9: lstd_v = 1e-9

        # 1. Monta vetor de features para o Classificador
        atr_vec_clf = []
        for spec in self.spec_order:
            p = self.reg_data['models'][spec]
            atr = self._calc_atr(ch, cv, p['alpha'], p['beta'])
            atr_vec_clf.append(atr)
        
        feat_clf = np.array(atr_vec_clf + self._calc_nss(ch, cv, mode='clf')).reshape(1, -1)
        
        # 2. Soft Voting: Obtém probabilidades
        probs = self.clf.predict_proba(feat_clf)[0]
        
        # Identifica a classe dominante (apenas para reportar o tipo)
        dominant_idx = np.argmax(probs)
        detected_type = self.le.inverse_transform([dominant_idx])[0]
        
        # 3. Regressão Ponderada (Weighted Fusion)
        weighted_dmos = 0.0
        
        for i, prob in enumerate(probs):
            if prob < 0.01: continue # Otimização: ignora classes com chance residual
            
            c_name = self.le.inverse_transform([i])[0]
            
            if c_name in self.svrs:
                model_data = self.svrs[c_name]
                
                # Extrai feature específica para este especialista (ATR otimizado + NSS)
                atr = self._calc_atr(ch, cv, model_data['alpha'], model_data['beta'])
                nss = self._calc_nss(ch, cv, mode='svr')
                
                feat_svr = np.array([atr] + nss).reshape(1, -1)
                
                # Predição individual
                dmos_part = model_data['model'].predict(feat_svr)[0]
                
                # Acumula ponderado pela confiança do classificador
                weighted_dmos += prob * dmos_part
            else:
                # Fallback de segurança (média geral) caso falte um modelo
                weighted_dmos += prob * 50.0 
            
        return detected_type, weighted_dmos

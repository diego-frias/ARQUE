import numpy as np
import cv2
import joblib
import json
from scipy import ndimage
from scipy.stats import kurtosis, skew

class ARQUEModel:
    def __init__(self, json_path, clf_pkl_path, svr_pkl_path):
        # Carrega configurações e modelos
        with open(json_path, 'r') as f: 
            self.reg_data = json.load(f)
        
        clf_pack = joblib.load(clf_pkl_path)
        self.clf = clf_pack['classifier']
        self.le = clf_pack['label_encoder']
        self.spec_order = clf_pack['specialist_order']
        
        self.svrs = joblib.load(svr_pkl_path)
        
        # Kernel Laplaciano
        self.kh = np.array([[1, -2, 1]])
        self.kv = np.array([[1], [-2], [1]])

    def _calc_maps(self, img):
        img_int = img.astype(np.int16)
        ch = np.log1p(np.abs(ndimage.convolve(img_int, self.kh, mode='reflect')))
        cv = np.log1p(np.abs(ndimage.convolve(img_int, self.kv, mode='reflect')))
        return ch, cv

    def _calc_atr(self, ch, cv, alpha, beta):
        lstd_h, lstd_v = np.std(ch), np.std(cv)
        if lstd_h < 1e-9: lstd_h = 1e-9
        if lstd_v < 1e-9: lstd_v = 1e-9
        
        mask_h = (cv >= beta * lstd_v) & (ch < alpha * lstd_h)
        mask_v = (ch >= beta * lstd_h) & (cv < alpha * lstd_v)
        
        return (np.sum(mask_h) + np.sum(mask_v)) / 2.0 / ch.size

    def _calc_nss(self, ch, cv, mode='svr'):
        # mode='svr' retorna features mais ricas (Kurtosis, Skew, Std, Mean)
        # mode='clf' retorna features de pareamento (baseado no script original)
        f = []
        # Estatísticas de primeira ordem
        for m in [ch, cv]:
            flat = m.flatten()
            f.extend([kurtosis(flat), skew(flat), np.std(flat)])
            if mode == 'svr':
                f.append(np.mean(flat))
        
        # Estatísticas de segunda ordem (pairwise)
        for m in [ch, cv]:
            hp = m[:, :-1] * m[:, 1:]
            if mode == 'clf':
                vp = m[:-1, :] * m[1:, :]
                for p in [hp, vp]: f.extend([np.mean(p), np.std(p)])
            else:
                f.extend([np.mean(hp), np.std(hp)])
        return f

    def predict(self, img):
        """
        Prediction pipeline: 
        1. Extract Features -> 2. Classify (Soft Voting) -> 3. Regress -> 4. Weighted Average
        """
        ch, cv = self._calc_maps(img)
        lstd_h, lstd_v = np.std(ch), np.std(cv)
        if lstd_h < 1e-9: lstd_h = 1e-9
        if lstd_v < 1e-9: lstd_v = 1e-9

        # 1. Feature Vector for Classification
        atr_vec_clf = []
        for spec in self.spec_order:
            p = self.reg_data['models'][spec]
            atr = self._calc_atr(ch, cv, p['alpha'], p['beta'])
            atr_vec_clf.append(atr)
        
        feat_clf = np.array(atr_vec_clf + self._calc_nss(ch, cv, mode='clf')).reshape(1, -1)
        
        # 2. Get Probabilities (Soft Voting)
        probs = self.clf.predict_proba(feat_clf)[0]
        
        # Get dominant class for Confusion Matrix analysis
        dominant_idx = np.argmax(probs)
        detected_type = self.le.inverse_transform([dominant_idx])[0]
        
        # 3. Weighted Regression
        weighted_dmos = 0.0
        for i, prob in enumerate(probs):
            if prob < 0.01: continue # Optimization threshold
            
            c_name = self.le.inverse_transform([i])[0]
            model_data = self.svrs[c_name]
            
            # Specific features for this expert
            atr = self._calc_atr(ch, cv, model_data['alpha'], model_data['beta'])
            nss = self._calc_nss(ch, cv, mode='svr')
            
            feat_svr = np.array([atr] + nss).reshape(1, -1)
            dmos_part = model_data['model'].predict(feat_svr)[0]
            
            weighted_dmos += prob * dmos_part
            
        return detected_type, weighted_dmos
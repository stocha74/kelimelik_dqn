#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import product
from fractions import Fraction
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import kelimelik_engine1 as ke
from copy import deepcopy
import os
from datetime import datetime

# --- 84 eylem: toplamı 1 olan 4'lü ağırlık vektörleri ---
def generate_simplex_actions(levels=6):
    step = Fraction(1, levels)
    grid = [step * i for i in range(levels + 1)]
    actions = []
    for w in product(grid, repeat=4):
        if sum(w) == 1:
            actions.append(tuple(float(x) for x in w))
    return actions

actions_84 = generate_simplex_actions(levels=6)

# --- Harfler ve ID eşlemeleri ---
HARFLER = ['A','B','C','Ç','D','E','F','G','Ğ','H','I','İ','J',
           'K','L','M','N','O','Ö','P','R','S','Ş','T','U','Ü','V','Y','Z','*']
HARF2ID = {h: i for i, h in enumerate(HARFLER)}

# --- Encode yardımcıları ---
def encode_board(board):
    tensor = np.zeros((15, 15, len(HARFLER)), dtype=np.float32)
    for y in range(15):
        for x in range(15):
            cell = str(board[y][x]).upper() if board[y][x] != "" else ""
            if cell in HARF2ID:
                tensor[y, x, HARF2ID[cell]] = 1.0
    return tensor

def encode_raf(raf):
    vec = np.zeros(len(HARFLER), dtype=np.float32)
    for h in raf:
        if h in HARF2ID:
            vec[HARF2ID[h]] += 1.0
    return vec

def encode_stok(stok):
    vec = np.zeros(len(HARFLER), dtype=np.float32)
    for h, count in stok.items():
        if h in HARF2ID:
            vec[HARF2ID[h]] = float(count)
    return vec

def encode_bonus_matrix(bonus_mat):
    return bonus_mat.reshape((15, 15, 1)).astype(np.float32)


# --- Ortam tanımı ---
class KelimelikDQNEnv(gym.Env):
    def __init__(self, sozluk, tahta_puanlari2, harf_stogu, debug=False, log_dir="logs", auto_save_excel=True):
        super().__init__()
        self.sozluk = sozluk
        self.tahta_puanlari2 = deepcopy(tahta_puanlari2)
        self.init_stok = deepcopy(harf_stogu)
        self.debug = debug
        self.log_dir = log_dir
        self.auto_save_excel = auto_save_excel
        self.episode_no = 0
        self.turn_history = []
        self.last_log_path = None

        # Gözlem alanı
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 1, shape=(15, 15, len(HARFLER)), dtype=np.float32),
            "raf": spaces.Box(0, 7, shape=(len(HARFLER),), dtype=np.float32),
            "bonus": spaces.Box(0, 25, shape=(15, 15, 1), dtype=np.float32),
            "stok": spaces.Box(0, 12, shape=(len(HARFLER),), dtype=np.float32),
            "skor_farki": spaces.Box(low=-300.0, high=300.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(len(actions_84))  # 84 aksiyon
        self.reset()

    def _rastgele_altay_pozisyonu(self):
        """
        ALTAY kelimesini her episode'da merkez hücreden (7,7) geçecek şekilde
        yatay veya dikey rastgele konumlar.
        """
        olasi = []
        # Yatay: satır 7 sabit, başlangıç x = 3..7 -> x=7 kesin içeride
        for x0 in range(3, 8):
            olasi.append((x0, 7, "h"))
        # Dikey: sütun 7 sabit, başlangıç y = 3..7 -> y=7 kesin içeride
        for y0 in range(3, 8):
            olasi.append((7, y0, "v"))

        idx = int(self.np_random.integers(0, len(olasi)))
        return olasi[idx]

    def _altay_yerlestir(self):
        x0, y0, orient = self._rastgele_altay_pozisyonu()
        if orient == "h":
            for i, ch in enumerate("ALTAY"):
                self.board[y0][x0 + i] = ch
        else:
            for i, ch in enumerate("ALTAY"):
                self.board[y0 + i][x0] = ch

    def _rastgele_25_bonusu_koy(self):
        """
        25 puan hücresini her episode'da yeni bir yere taşır.
        Sadece bonusu 0 olan ve tahta üzerinde harf olmayan bir hücre seçilir.
        """
        self.bonus[self.bonus == 25] = 0

        uygun_hucreler = []
        for y in range(15):
            for x in range(15):
                if self.bonus[y, x] == 0 and self.board[y][x] == "":
                    uygun_hucreler.append((y, x))

        if not uygun_hucreler:
            return

        idx = int(self.np_random.integers(0, len(uygun_hucreler)))
        y, x = uygun_hucreler[idx]
        self.bonus[y, x] = 25


    def _log_turn(self, action_id, action_source, weights, puan_biz, puan_rakip, rl_no_move, rakip_no_move):
        self.turn_history.append({
            "episode": int(self.episode_no),
            "turn": int(self.turn),
            "action_id": int(action_id),
            "action_source": str(action_source),
            "agent_type": "policy" if str(action_source).lower() == "policy" else "random",
            "w_puan": float(weights[0]),
            "w_harf": float(weights[1]),
            "w_dez": float(weights[2]),
            "w_oran": float(weights[3]),
            "puan_biz": float(puan_biz),
            "puan_rakip": float(puan_rakip),
            "kumulatif_biz": float(self.own_score),
            "kumulatif_rakip": float(self.opp_score),
            "forced_pass_biz": bool(rl_no_move),
            "forced_pass_rakip": bool(rakip_no_move)
        })

    def _save_turn_history_excel(self):
        if not self.turn_history:
            return None

        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_path = os.path.join(self.log_dir, f"episode_{self.episode_no:05d}_{ts}.xlsx")

        try:
            import pandas as pd
            pd.DataFrame(self.turn_history).to_excel(xlsx_path, index=False)
            self.last_log_path = xlsx_path
            return xlsx_path
        except Exception:
            csv_path = xlsx_path.replace('.xlsx', '.csv')
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.turn_history[0].keys()))
                writer.writeheader()
                writer.writerows(self.turn_history)
            self.last_log_path = csv_path
            return csv_path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_no += 1
        self.turn_history = []
        self.last_log_path = None

        # Boş tahta
        self.board = np.array([["" for _ in range(15)] for _ in range(15)])
        self.bonus = deepcopy(self.tahta_puanlari2)
        self._altay_yerlestir()
        self._rastgele_25_bonusu_koy()
        self.stok = deepcopy(self.init_stok)

        # Her iki oyuncuya 7'şer harf dağıt
        self.elde, self.stok = ke.harf_dagit(self.stok, 7)
        self.elde_rakip, self.stok = ke.harf_dagit(self.stok, 7)

        self.own_score = 0
        self.opp_score = 0
        self.turn = 0
        self.consecutive_passes = 0

        return self._get_obs(), {}

    def step_emektar(self, action_id, action_source="policy"):

        """
        2 ajanlı ortam:
        - RL ajan (biz) action_id ile oynar
        - Deterministik ajan (rakip) max puanla oynar
        - Reward = fark (biz - rakip)
        - Eğer kelime bulunamazsa DEBUG bilgisi basar
        """
        import numpy as np
    
        # --- 1️⃣ RL ajan oynuyor ---
        weights = actions_84[action_id]
        elde_str = ''.join(self.elde)
    
        #self.board, eksilen_biz, puan_biz, ana_dizin_biz = ke.hamle_cok_kriterli(
         #   self.board, self.bonus, elde_str, self.sozluk,
          #  w_puan=1, w_harf=0,
          #  w_dez=0, w_oran=0
        #)

        self.board, eksilen_biz, puan_biz, ana_dizin_biz = ke.hamle_cok_kriterli(
             self.board, 
             self.bonus, 
             elde_str, 
             self.sozluk,
             w_puan=weights[0], 
             w_harf=weights[1],
             w_dez=weights[2], 
             w_oran=weights[3]
        )





        
    
        # --- DEBUG: kelime bulunamadı mı? ---
        if not ana_dizin_biz or puan_biz == 0:
            if self.debug:
                print("⚠️  [DEBUG] RL ajan kelime bulamadı!")
                print(f"     Ağırlıklar: w_puan={weights[0]:.2f}, w_harf={weights[1]:.2f}, w_dez={weights[2]:.2f}, w_oran={weights[3]:.2f}")
        else:
            if self.debug:
                print(f"✅  [DEBUG] Kelime bulundu | puan={puan_biz:.1f} | w={tuple(round(w,2) for w in weights)}")
                print("---------------------------------------------")
    
        # --- Raf güncellemesi ---
        if eksilen_biz:
            self.elde = ke.raftan_cikar(self.elde, eksilen_biz)
            yeniler_biz, self.stok = ke.harf_dagit(self.stok, len(eksilen_biz))
            self.elde.extend(yeniler_biz)
    
        self.own_score += puan_biz
        #print("Bizim opsiyon sayısı"+str(len(ana_dizin_biz)))
    
        # --- 2️⃣ Deterministik rakip oynuyor ---
        elde_rakip_str = ''.join(self.elde_rakip)
        self.board, eksilen_rakip, puan_rakip, ana_dizin_rakip = ke.hamle_cok_kriterli(
            self.board, self.bonus, elde_rakip_str, self.sozluk,
            w_puan=1.0, w_harf=0.0, w_dez=0.0, w_oran=0.0
        )
        #print("Rakip opsiyon sayısı"+str(len(ana_dizin_rakip)))
        
        if self.debug:
            print(f"🔸  [DEBUG] Rakip Kelime bulundu | puan={puan_rakip:.1f} ")
            print("---------------------------------------------")
    
        if eksilen_rakip:
            self.elde_rakip = ke.raftan_cikar(self.elde_rakip, eksilen_rakip)
            yeniler_rakip, self.stok = ke.harf_dagit(self.stok, len(eksilen_rakip))
            self.elde_rakip.extend(yeniler_rakip)
        self.opp_score += puan_rakip
    
        # --- 3️⃣ Reward ve done ---
        reward_raw = puan_biz - puan_rakip
        reward = reward_raw   # clipping kaldırıldı
    
        self.turn += 1
        done = (self.turn >= 30) or (sum(self.stok.values()) == 0)

        self._log_turn(action_id, action_source, weights, puan_biz, puan_rakip, False, False)
        log_path = self._save_turn_history_excel() if (done and self.auto_save_excel) else None
    
        # --- 4️⃣ Bilgi döndür ---
        info = {
            "kelime_biz": ana_dizin_biz,
            "kelime_rakip": ana_dizin_rakip,
            "puan_biz": puan_biz,
            "puan_rakip": puan_rakip,
            "agirliklar": weights,
            "reward_raw": reward_raw,
            "action_source": action_source,
            "log_path": log_path
        }
    
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        return {
            "board": encode_board(self.board),
            "raf": encode_raf(self.elde),
            "bonus": encode_bonus_matrix(self.bonus),
            "stok": encode_stok(self.stok),
            "skor_farki": np.array([self.own_score - self.opp_score], dtype=np.float32)
        }
    
    def render(self):
        ke.print_board(self.board)
        print(f"Biz: {self.own_score} | Rakip: {self.opp_score} | Turn: {self.turn}")
        print("Raf:", self.elde)
        print("Stok:", sum(self.stok.values()))

    def step(self, action_id, action_source="policy"):

        import numpy as np
    
        weights = actions_84[action_id]
        elde_str = ''.join(self.elde)
    
        # --- 1️⃣ RL ajan oynuyor ---
        board_before_biz = self.board.copy()
    
        self.board, eksilen_biz, puan_biz, ana_dizin_biz = ke.hamle_cok_kriterli(
            self.board,
            self.bonus,
            elde_str,
            self.sozluk,
            w_puan=weights[0],
            w_harf=weights[1],
            w_dez=weights[2],
            w_oran=weights[3]
        )
    
        rl_no_move = (not ana_dizin_biz) or (eksilen_biz is None) or (len(eksilen_biz) == 0)
    
        if rl_no_move:
            # FORCED PASS
            self.board = board_before_biz
            puan_biz = 0
            eksilen_biz = []
            ana_dizin_biz = []
    
            self.consecutive_passes = getattr(self, "consecutive_passes", 0) + 1
            if self.debug:
                print("⚠️  [DEBUG] RL FORCED PASS (hamle yok)")
                print(f"     Ağırlıklar: w_puan={weights[0]:.2f}, w_harf={weights[1]:.2f}, w_dez={weights[2]:.2f}, w_oran={weights[3]:.2f}")
        else:
            self.consecutive_passes = 0
            #print(f"✅  [AJAN 1 DEBUG] Kelime bulundu | puan={puan_biz:.1f} | w={tuple(round(w,2) for w in weights)}")
            #print("---------------------------------------------")
    
        # --- Raf güncellemesi (hamle yoksa eksilen_biz zaten []) ---
        if eksilen_biz:
            self.elde = ke.raftan_cikar(self.elde, eksilen_biz)
            yeniler_biz, self.stok = ke.harf_dagit(self.stok, len(eksilen_biz))
            self.elde.extend(yeniler_biz)
    
        self.own_score += puan_biz
        #print("Bizim opsiyon sayısı" + str(len(ana_dizin_biz)))
        if self.debug:
            print("---------------------------------------------")
    
        # --- 2️⃣ Deterministik rakip oynuyor ---
        elde_rakip_str = ''.join(self.elde_rakip)
        board_before_rakip = self.board.copy()
    
        self.board, eksilen_rakip, puan_rakip, ana_dizin_rakip = ke.hamle_cok_kriterli(
            self.board, self.bonus, elde_rakip_str, self.sozluk,
            w_puan=1.0, w_harf=0.0, w_dez=0.0, w_oran=0.0
        )
    
        rakip_no_move = (not ana_dizin_rakip) or (eksilen_rakip is None) or (len(eksilen_rakip) == 0)
    
        if rakip_no_move:
            self.board = board_before_rakip
            puan_rakip = 0
            eksilen_rakip = []
            ana_dizin_rakip = []
    
            self.consecutive_passes = getattr(self, "consecutive_passes", 0) + 1
            if self.debug:
                print("⚠️  [DEBUG] RAKIP FORCED PASS (hamle yok)")
                print("---------------------------------------------")
        else:
            self.consecutive_passes = 0
            #print(f"🔸  [AJAN 2 DEBUG] Rakip Kelime bulundu | puan={puan_rakip:.1f} ")
            #print("---------------------------------------------")
    
        if eksilen_rakip:
            self.elde_rakip = ke.raftan_cikar(self.elde_rakip, eksilen_rakip)
            yeniler_rakip, self.stok = ke.harf_dagit(self.stok, len(eksilen_rakip))
            self.elde_rakip.extend(yeniler_rakip)
    
        self.opp_score += puan_rakip
    
        # --- 3️⃣ Reward ve done ---
        reward_raw = puan_biz - puan_rakip
        reward = reward_raw
    
        self.turn += 1
        done = (self.turn >= 30) or (sum(self.stok.values()) == 0) or (getattr(self, "consecutive_passes", 0) >= 2)

        self._log_turn(action_id, action_source, weights, puan_biz, puan_rakip, rl_no_move, rakip_no_move)
        log_path = self._save_turn_history_excel() if (done and self.auto_save_excel) else None
    
        info = {
            "kelime_biz": ana_dizin_biz,
            "kelime_rakip": ana_dizin_rakip,
            "puan_biz": puan_biz,
            "puan_rakip": puan_rakip,
            "agirliklar": weights,
            "reward_raw": reward_raw,
            "forced_pass_biz": rl_no_move,
            "forced_pass_rakip": rakip_no_move,
            "action_source": action_source,
            "log_path": log_path
        }
    
        return self._get_obs(), reward, done, False, info

'''

    def step(self, action_id, action_source="policy"):

        """
        2 ajanlı ortam:
        - RL ajan (biz) action_id ile oynar
        - Deterministik ajan (rakip) max puanla oynar
        - Reward = fark (biz - rakip)
        """
        import numpy as np

        # --- 1️⃣ RL ajan oynuyor ---
        weights = actions_84[action_id]
        elde_str = ''.join(self.elde)

        self.board, eksilen_biz, puan_biz, ana_dizin_biz = ke.hamle_cok_kriterli(
            self.board, self.bonus, elde_str, self.sozluk,
            w_puan=weights[0], w_harf=weights[1],
            w_dez=weights[2], w_oran=weights[3]
        )
        ops_biz = len(ana_dizin_biz) if ana_dizin_biz else 0

        # --- DEBUG BLOĞU ---
        if not ana_dizin_biz or puan_biz == 0:
            print("⚠️  [DEBUG] Kelime bulunamadı!")
            print(f"     Ağırlıklar: w_puan={weights[0]:.2f}, w_harf={weights[1]:.2f}, w_dez={weights[2]:.2f}, w_oran={weights[3]:.2f}")
        else:
            print(f"✅  [DEBUG] Kelime bulundu | puan={puan_biz} | w={weights}")

        if eksilen_biz:
            self.elde = ke.raftan_cikar(self.elde, eksilen_biz)
            yeniler_biz, self.stok = ke.harf_dagit(self.stok, len(eksilen_biz))
            self.elde.extend(yeniler_biz)
        self.own_score += puan_biz

        # --- 2️⃣ Deterministik rakip oynuyor ---
        elde_rakip_str = ''.join(self.elde_rakip)
        self.board, eksilen_rakip, puan_rakip, ana_dizin_rakip = ke.hamle_cok_kriterli(
            self.board, self.bonus, elde_rakip_str, self.sozluk,
            w_puan=1.0, w_harf=0.0, w_dez=0.0, w_oran=0.0
        )
        ops_rakip = len(ana_dizin_rakip) if ana_dizin_rakip else 0

        if eksilen_rakip:
            self.elde_rakip = ke.raftan_cikar(self.elde_rakip, eksilen_rakip)
            yeniler_rakip, self.stok = ke.harf_dagit(self.stok, len(eksilen_rakip))
            self.elde_rakip.extend(yeniler_rakip)
        self.opp_score += puan_rakip

        # --- 3️⃣ Reward ve done ---
        reward_raw = puan_biz - puan_rakip
        reward = np.clip(reward_raw, -30, 30)

        self.turn += 1
        done = (self.turn >= 30) or (sum(self.stok.values()) == 0)

        # --- 4️⃣ Bilgi döndür ---
        info = {
            "kelime_biz": ana_dizin_biz,
            "kelime_rakip": ana_dizin_rakip,
            "puan_biz": puan_biz,
            "puan_rakip": puan_rakip,
            "opsiyon_biz": ops_biz,
            "opsiyon_rakip": ops_rakip,
            "agirliklar": weights,
            "reward_raw": reward_raw
        }

        return self._get_obs(), reward, done, False, info
'''

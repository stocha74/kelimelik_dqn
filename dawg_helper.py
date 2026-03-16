#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter

class DawgNode:
    __slots__ = ("children", "is_word")
    def __init__(self):
        self.children = {}
        self.is_word = False

class Dawg:
    def __init__(self):
        self.root = DawgNode()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            node = node.children.get(ch) or node.children.setdefault(ch, DawgNode())
        node.is_word = True


def build_dawg_from_dictionary(sozluk):
    """
    sozluk: iterable[str] (tercihen UPPERCASE Türkçe kelimeler)
    """
    d = Dawg()
    for w in sozluk:
        w = w.strip()
        if w:
            d.insert(w)
    return d


# In[2]:


def extract_word_parts(board):
    """
    board: 15x15, hücreler '' veya 'A' gibi tek harf.
    Dönen: list[(parca_str, (y,x), 'H'/'V')]
    """
    H, W = len(board), len(board[0])
    parts = []

    # --- Horizontal (H) ---
    for y in range(H):
        x = 0
        while x < W:
            if board[y][x] != '':
                x0 = x
                s = []
                while x < W and board[y][x] != '':
                    s.append(board[y][x])
                    x += 1
                parca = ''.join(s)
                parts.append((parca, (y, x0), 'H'))
            else:
                x += 1

    # --- Vertical (V) ---
    for x in range(W):
        y = 0
        while y < H:
            if board[y][x] != '':
                y0 = y
                s = []
                while y < H and board[y][x] != '':
                    s.append(board[y][x])
                    y += 1
                parca = ''.join(s)
                parts.append((parca, (y0, x), 'V'))
            else:
                y += 1

    return parts


# In[3]:


def compute_left_right_limits(board, part, yx, yon):
    """
    part: ör. 'KÖL'
    yx: (y,x) part'ın başladığı yer
    yon: 'H' / 'V'
    Dönen: (max_left, max_right)
    """
    H, W = len(board), len(board[0])
    y, x = yx
    L = len(part)

    if yon.upper() == 'H':
        # sol boşluklar
        left = 0
        xx = x - 1
        while xx >= 0 and board[y][xx] == '':
            left += 1
            xx -= 1
        # sağ boşluklar
        right = 0
        xx = x + L
        while xx < W and board[y][xx] == '':
            right += 1
            xx += 1
        return left, right

    else:  # 'V'
        # üst boşluklar = left
        left = 0
        yy = y - 1
        while yy >= 0 and board[yy][x] == '':
            left += 1
            yy -= 1
        # alt boşluklar = right
        right = 0
        yy = y + L
        while yy < H and board[yy][x] == '':
            right += 1
            yy += 1
        return left, right


# In[4]:


def _walk_fixed(node, fixed_str):
    """node'dan fixed_str boyunca ilerler; yoksa None döner."""
    cur = node
    for ch in fixed_str:
        nxt = cur.children.get(ch)
        if nxt is None:
            return None
        cur = nxt
    return cur

def _dfs_suffix(node, rack_counter, remaining_slots, prefix_word, out):
    """
    node: DAWG node (şu ana kadar prefix_word yazıldı)
    remaining_slots: en fazla kaç harf daha eklenebilir
    """
    if node.is_word:
        out.add(prefix_word)

    if remaining_slots == 0:
        return

    for ch, nxt in node.children.items():
        if rack_counter.get(ch, 0) > 0:
            rack_counter[ch] -= 1
            _dfs_suffix(nxt, rack_counter, remaining_slots - 1, prefix_word + ch, out)
            rack_counter[ch] += 1

def _dfs_prefix_then_fixed(dawg, fixed, rack_counter, max_left, max_right):
    """
    fixed: tahtadaki sabit parça (ör. 'KÖL')
    max_left/max_right: boşluk limitleri
    Dönen: set[str]  -> (prefix + fixed + suffix) aday kelimeler
    """
    out = set()

    def rec_prefix(node, prefix_str, depth_left):
        # prefix_str oluşturuldu; şimdi fixed'i yürüt
        after_fixed = _walk_fixed(node, fixed)
        if after_fixed is not None:
            # fixed'ten sonra suffix üret
            _dfs_suffix(after_fixed, rack_counter, max_right, prefix_str + fixed, out)

        # daha fazla prefix harfi ekleyebilir miyiz?
        if depth_left == 0:
            return

        for ch, nxt in node.children.items():
            if rack_counter.get(ch, 0) > 0:
                rack_counter[ch] -= 1
                rec_prefix(nxt, prefix_str + ch, depth_left - 1)
                rack_counter[ch] += 1

    rec_prefix(dawg.root, "", max_left)
    return out

def generate_candidates_for_word_part(dawg, word_part, rack_letters):
    """
    word_part: (part, (y,x), 'H'/'V')
    rack_letters: eldeki harfler -> ör. ['A','K','L',... ] veya 'AKL...'
    Dönen: list[(kelime, x0, y0, yon)]  -> engine'e direkt uyumlu
    """
    part, (y, x), yon = word_part
    yon = yon.upper()
    rack_counter = Counter(rack_letters if isinstance(rack_letters, list) else list(rack_letters))

    max_left, max_right = compute_left_right_limits(board=None, part=part, yx=(y, x), yon=yon)  # placeholder
    raise RuntimeError("Bu fonksiyonu aşağıdaki wrapper ile çağıracağız.")


# In[5]:


def generate_candidates_for_word_part_with_board(dawg, board, word_part, rack_letters):
    part, (y, x), yon = word_part
    yon = yon.upper()

    rack_counter = Counter(rack_letters if isinstance(rack_letters, list) else list(rack_letters))
    max_left, max_right = compute_left_right_limits(board, part, (y, x), yon)

    words = _dfs_prefix_then_fixed(dawg, part, rack_counter, max_left, max_right)

    # Start coord: prefix uzunluğuna göre kaydır
    out = []
    for w in words:
        prefix_len = w.find(part)  # part ilk kez geçtiği yer
        if prefix_len < 0:
            continue

        if yon == 'H':
            x0, y0 = x - prefix_len, y
        else:  # 'V'
            x0, y0 = x, y - prefix_len

        out.append((w, x0, y0, yon))

    return out


# In[ ]:





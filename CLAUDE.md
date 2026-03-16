# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Q-Network (DQN) reinforcement learning project for training an AI to play **Kelimelik**, a Turkish Scrabble-like word game. The RL agent learns to select optimal multi-criteria weight vectors to guide move selection rather than directly selecting individual moves.

## Dependencies

Install manually (no requirements.txt):
```bash
pip install tensorflow gymnasium numpy jupyter
```

## Running the Project

The main entry point is the Jupyter notebook:
```bash
jupyter notebook "Kelimelik Dqn Trainer2.ipynb"
```

There is no formal test suite, build system, or linter configured.

## Architecture

### Key Design Pattern: Indirect Action Space

The DQN does not select specific word placements directly. Instead, it selects from **84 discrete weight vectors** (generated via `generate_simplex_actions(levels=6)`) that parameterize the scoring function `hamle_cok_kriterli()`. The 4 weights are:
- `w_puan` — raw score
- `w_harf` — letters per point (letter efficiency)
- `w_dez` — disadvantage penalty (blocks opponent)
- `w_oran` — board coverage ratio

### Module Roles

| File | Role |
|------|------|
| `kelimelik_engine1.py` | Game engine (4,226 lines): word validation, scoring, board multipliers, move generation |
| `kelimelik_parametric_env.py` | Gymnasium `Env` wrapper; defines state/action spaces and two-player episode logic |
| `dqn_model_keras.py` | Keras Q-network: flattened input (7,036 dims) → Dense(512→256→128) → 84 Q-values |
| `dawg_helper.py` | DAWG structure for efficient Turkish dictionary lookups and candidate generation |
| `turkce_kelime_listesi.txt` | 76,131-word Turkish dictionary |

### State Space (observation dict)
- `board`: `(15, 15, 30)` one-hot tensor — 29 Turkish letters + joker wildcard
- `raf`: `(30,)` letter count vector — agent's current rack
- `bonus`: `(15, 15, 1)` — board multiplier map (0, 2×, 3×, 4×, 6×)
- `stok`: `(30,)` — remaining letters in the shared stock
- `skor_farki`: scalar — current score difference (agent − opponent)

`encode_obs_dict()` in `dqn_model_keras.py` flattens this dict to a 7,036-dim float32 vector for the network.

### Game Flow per Episode
1. Board initialized with "ALTAY" placed at center
2. RL agent selects action (weight vector index) via ε-greedy
3. `hamle_cok_kriterli()` uses weights to score and select the best valid word
4. Deterministic opponent responds using pure score maximization (`w_puan=1.0`)
5. Reward = `our_score − opponent_score` for the turn
6. Episode ends after 30 turns, stock depletion, or 2 consecutive passes

### Training Hyperparameters (notebook)
- γ = 0.99, LR = 1e-3, batch_size = 64, replay buffer = 50k
- ε-greedy decay over ~30k steps
- Best model saved when episode reward improves

## Important Engine Functions

- `kelime_kontrol()` — validates word placement and computes raw score
- `hamle_cok_kriterli()` — **core move selector**: scores all candidates on 4 criteria, returns best weighted move
- `kelime_yerlestir_ve_puanla5()` — places word on board and finalizes score
- `generate_candidates_for_word_part_with_board()` in `dawg_helper.py` — generates valid word extensions given board constraints

## Language Notes

Source code comments and variable names are in Turkish. Key vocabulary:
- `tahta` = board, `harf` = letter, `kelime` = word, `puan` = score
- `raf` = rack (player's letters), `stok` = tile stock, `hamle` = move
- `sozluk` = dictionary, `yatay`/`dikey` = horizontal/vertical

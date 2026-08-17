# Configuration index

Gin configs are organized by purpose. Reviewers should start with `cfg/paper/`,
which contains exactly the configs that produced the runs reported in the paper.

## Layout

| Folder | Purpose |
|--------|---------|
| `cfg/paper/`        | Configs that produced the runs reported in the paper, grouped by table. |
| `cfg/debug/`        | Local-debug configs (CPU, dummy data, small model variants). |
| `cfg/downstream/`   | Downstream-task configs (probing, zero-shot, retrieval). |
| `cfg/freesound/`    | Freesound-only training variants. |
| `cfg/nvidia_grant/` | Configs from the NVIDIA grant work; not part of the paper. |
| `cfg/*.gin`         | Baselines, embedding-extraction configs, and non-paper experiments (E1, E3 variants, E4–E6, E8/E9 ablations). Several of these are `include` targets for the curated paper configs and must stay at this path. |

## Paper → config mapping

Each row in the paper tables corresponds to one training run; each run was launched
from one gin config. Run IDs are anonymised labels (R01–R14) for review.

### Table 1 (`tab:data`) — data ablation, section 5.1

| Run ID | Config                                              |
|--------|-----------------------------------------------------|
| R01    | `cfg/paper/tab_data/config_E3_8_music_sound.gin`           |
| R02    | `cfg/paper/tab_data/config_E3_12_quotes.gin`               |
| R03    | `cfg/paper/tab_data/config_E3_11_struct.gin`               |
| R04    | `cfg/paper/tab_data/config_E3_5_dt_msd_m4rag_fs_pse.gin`   |
| R05    | `cfg/paper/tab_data/config_E3_10_music_sound_struct.gin`   |

### Table 4 (`tab:layers`) — layer selection, section 5.2

| Run ID | Config                                              |
|--------|-----------------------------------------------------|
| R04    | `cfg/paper/tab_data/config_E3_5_dt_msd_m4rag_fs_pse.gin` (layer 12, shared with Table 1) |
| R06    | `cfg/paper/tab_layers/config_E7_1_layer6.gin`              |
| R07    | `cfg/paper/tab_layers/config_E7_2_all_layers.gin`          |

### Table 5 (`tab:objective`) — training objective, section 5.3

| Run ID           | Config                                                    | Objective |
|------------------|-----------------------------------------------------------|-----------|
| R07              | `cfg/paper/tab_layers/config_E7_2_all_layers.gin`                | InfoNCE |
| R08              | `cfg/paper/tab_objective/config_E7_4_all_layers_sigmoid.gin`     | Sigmoid |
| (E8_6 in flight) | `cfg/paper/tab_objective/config_E8_6_lejepa.gin`                 | LeJEPA (cosine) |
| (E9_9 in flight) | `cfg/paper/tab_objective/config_E9_9_lejepa_infonce.gin`         | LeJEPA (InfoNCE invariance) |

### Table 6 (`tab:text_encoder`) — text-encoder training, section 5.4

| Run ID | Config                                                              | Notes |
|--------|---------------------------------------------------------------------|-------|
| R09    | `cfg/config_clap_mpnet_base_v2_ssl_mp_10s_small_clap_dt_msd_fs_pse_lr_5e-6.gin` | MPNet baseline (this file is also the include target shared by all paper configs, hence kept at `cfg/`). |
| R10    | `cfg/paper/tab_text_encoder/config_E2_1_roberta.gin`                       | RoBERTa |
| R11    | `cfg/paper/tab_text_encoder/config_E2_2_xlmr.gin`                          | XLM-R |
| R12    | `cfg/paper/tab_text_encoder/config_E2_3_modernbert.gin`                    | ModernBERT |
| R07    | `cfg/paper/tab_layers/config_E7_2_all_layers.gin`                          | All-layers, frozen TE (baseline row) |
| R13    | `cfg/paper/tab_text_encoder/config_E7_5_all_layers_train_text.gin`         | + trainable TE |
| R14    | `cfg/paper/tab_objective/config_E9_8_lejepa_infonce.gin`                   | + trainable TE + LeJEPA-InfoNCE. Both R13 (frozen-objective trainable TE) and R14 (LeJEPA-InfoNCE trainable TE) are kept as paper rows. |

## Conventions

* Paths in `include` statements are relative to the repo root, so all configs
  must be invoked as `python src/train.py cfg/paper/tab_*/config_*.gin` from
  the project root. This mirrors how the cluster jobs launch them.
* Files named `config_E<num>_<num>_*.gin` are experiment-series identifiers
  used internally; the paper tables refer to anonymised run IDs (R01–R14),
  see the mapping above.

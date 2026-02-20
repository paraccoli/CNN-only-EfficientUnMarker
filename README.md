# CNN-only EfficientUnMarker

**[日本語解説 / Japanese Explanation]** | [English](#english)

---

## 日本語解説

### このリポジトリは何をするもの？

このリポジトリは、**画像に埋め込まれた可視ウォーターマーク（透かし）を自動的に除去する**パイプラインです。  
CNN（畳み込みニューラルネットワーク）ベースの検出器（MobileNetV3）だけを使った、高速・軽量・実用的な実装です。

> **例:** 「Getty Images」「SAMPLE」などの透かしが入った画像から、その透かしを取り除くことを目的としています。

---

### 処理の全体像（3ステージパイプライン）

```
入力画像（透かし入り）
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1: 周波数解析                │
│  torch.fft で周波数スペクトルを分析  │
│  → 攻撃すべき周波数帯域マスクを生成  │
└──────────────┬──────────────────────┘
               │ 周波数帯域マスク
               ▼
┌─────────────────────────────────────┐
│  Stage 2: マルチスケール攻撃         │
│  256×256 で粗最適化                 │
│  → 元の解像度にアップスケール        │
│  → 高解像度で精密最適化             │
│  ※ 早期停止・PSNR ガード付き        │
└──────────────┬──────────────────────┘
               │ 攻撃済み画像
               ▼
┌─────────────────────────────────────┐
│  Stage 3: 品質改善（任意・デフォルト無効）│
│  アダプティブローパスフィルタで       │
│  ノイズを平滑化                     │
└──────────────┬──────────────────────┘
               │
               ▼
        出力画像（透かし除去済み）
```

---

### 各コンポーネントの詳細説明

#### 🔍 ウォーターマーク検出器（`core/detection.py`）

透かしが「まだ残っているか」を判定するCNNモデルです。攻撃ループの終了条件として使います。

| クラス名 | モデル | 特徴 |
|---|---|---|
| `QuickDetector` | MobileNetV3-Small | 軽量・高速。シグモイド出力で信頼度スコアを返す |
| `DeepDetector` | DenseNet-121 | 高精度。不確実な場合にのみ呼ばれる |
| `DualBranchDetector` | MobileNetV3 + DenseNet | 信頼度が曖昧な場合のみ DenseNet を併用する「適応モード」 |
| `WatermarkDetector` | 上記のラッパー | `mode='quick'` / `'deep'` / `'adaptive'` を切り替え可能 |

**動作フロー（adaptive モード）：**
```
MobileNetV3 でスコア算出
    ├─ スコアが明確（< 0.3 または > 0.7）→ そのまま使用
    └─ 不確実（0.3 〜 0.7）→ DenseNet も使って重み付き平均
        quick_weight=0.6, deep_weight=0.4
```

---

#### 📊 Stage 1: 周波数解析（`core/stage1_frequency.py`）

画像のどの周波数帯域に透かしが多く含まれているかを予測し、攻撃の優先帯域を決定します。

**処理の流れ：**
1. `torch.fft.rfft2` で画像を周波数領域に変換
2. 8つの周波数帯域ごとにエネルギー・分散を計算
3. MobileNetV3 の画像特徴量と周波数特徴量を結合
4. 全結合層で各帯域の「攻撃優先度マスク」（0〜1）を出力

```
画像 → FFT → 周波数統計量（帯域エネルギー × 8）
             ＋                              → 融合 → 攻撃優先マスク
画像 → MobileNetV3 → 画像特徴量
```

---

#### ⚡ Stage 2: マルチスケール攻撃（`core/efficient_unmarker.py`）

画像を最適化して透かし検出スコアを下げる、このプロジェクトの中核部分です。

**最適化ループ：**
```python
# Adam オプティマイザで以下の損失を最小化
loss = w_det × 検出損失          # 透かし検出スコアを下げる
     + w_freq × 周波数損失        # 特定周波数成分を除去
     + w_smooth × 平滑化損失      # ノイズを抑制
     + w_rec × 領域再構成損失     # マスク内の画質を保つ
     + w_bg × 背景保持損失        # マスク外の画質を保つ
     + PSNR ペナルティ           # 画質劣化を防ぐソフト制約
```

**マルチスケール処理：**
| フェーズ | 解像度 | 反復回数 | 目的 |
|---|---|---|---|
| 低解像度最適化 | 256×256 | 最大40回 | 粗く大域的な透かし除去 |
| アップスケール | → 元サイズ | — | bicubic 補間 |
| 高解像度最適化 | 元サイズ | 最大60回 | 精密な仕上げ |

---

#### 🛑 早期停止（`core/early_stopping.py`）

無駄な計算を省くため、以下のいずれかの条件で最適化を早期終了します：

| 停止理由 | 条件 |
|---|---|
| `stopped_by_detection` | 検出スコアが閾値（デフォルト 0.5）を下回った |
| `stopped_by_convergence` | 直近5回の損失分散が 1e-4 未満（収束した） |
| `stopped_by_quality` | PSNR が下限（デフォルト 18.0 dB）を4回連続下回った |

---

#### 🖼️ インペインティングモデル（`core/inpainting_cnn.py`）

Partial Convolution ベースの U-Net で、マスク領域を自然に補完します（オプション）。

```
入力: RGB画像(3ch) + マスク(1ch) = 4ch
      ↓ エンコーダ（Partial Conv × 5段: 64→128→256→512→512）
      ↓ デコーダ（Upsample + Skip Connection × 5段）
      ↓ 最終Conv + Tanh → [0,1] にスケール
出力: 補完された RGB 画像(3ch)
```

**パラメータ数:** 約 14.3M（256×256 解像度時）

---

#### 📏 評価指標（`utils/metrics.py`）

| 指標 | 説明 |
|---|---|
| **PSNR** (Peak Signal-to-Noise Ratio) | 画質の客観評価。値が高いほど原画に近い（目標: 23 dB 以上） |
| **SSIM** (Structural Similarity) | 人間の視覚特性に基づく類似度（0〜1、1が完全一致） |
| **Masked PSNR/SSIM** | マスク領域内だけの評価指標 |

---

### 設定ファイル（`configs/fast_cnn_only.yaml`）

```yaml
device: cuda                # 使用デバイス（cuda / cpu）
target_detection: 0.5       # この値を下回れば「除去成功」と判定
max_iterations: 120         # 最大反復回数

detector:
  mode: quick               # 検出器モード（quick=QuickDetector (MobileNetV3-Small) のみ）

stage1:
  enabled: true             # 周波数解析を使う

stage2:
  low_res_size: 256         # 低解像度フェーズのサイズ
  low_iters: 40             # 低解像度での最大反復数
  high_iters: 60            # 高解像度での最大反復数

stage3:
  enabled: false            # 品質改善（デフォルト無効・速度重視）

stopping:
  psnr_min: 18.0            # PSNR の下限（dB）
  psnr_patience: 4          # 下限を何回連続で下回ったら停止するか

loss:
  detector_weight: 1.0      # 検出損失の重み
  psnr_penalty: 0.1         # PSNR ペナルティの強さ
  psnr_penalty_floor: 18.0  # PSNR ペナルティを発動する下限
```

---

### ディレクトリ構成

```
CNN-only-EfficientUnMarker/
├── core/                          # メインパイプライン
│   ├── efficient_unmarker.py      # EfficientUnMarker クラス（全体制御）
│   ├── detection.py               # ウォーターマーク検出器（MobileNetV3, DenseNet）
│   ├── stage1_frequency.py        # 周波数解析・帯域マスク予測
│   ├── inpainting_cnn.py          # Partial Conv U-Net（インペインティング）
│   └── early_stopping.py          # 早期停止・進捗トラッカー
├── utils/                         # ユーティリティ
│   ├── image_processing.py        # FFT 変換・フィルタ・リサイズ
│   ├── metrics.py                 # PSNR / SSIM 計算
│   ├── losses.py                  # 損失関数
│   └── masks.py                   # マスク生成・処理
├── configs/
│   └── fast_cnn_only.yaml         # 設定ファイル（ハイパーパラメータ）
├── experiments/
│   └── benchmark_cnn_only.py      # ベンチマーク実行スクリプト
├── datasets/                      # データセット置き場（要ダウンロード）
├── results/                       # ベンチマーク結果 CSV の出力先
├── requirements.txt               # 依存パッケージ一覧
└── install.sh                     # インストールスクリプト
```

---

### セットアップと使い方

#### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

または:

```bash
bash install.sh
```

RTX 5070 Ti など sm_120 系 GPU の場合は CUDA 12.8 対応 PyTorch が必要です:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### 2. データセットの準備（任意）

[Kaggle - Large-scale Common Watermark Dataset](https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset) からダウンロードして `datasets/` に配置してください。

#### 3. ベンチマークの実行

```bash
python experiments/benchmark_cnn_only.py \
  --config configs/fast_cnn_only.yaml \
  --input_dir <透かし入り画像フォルダ> \
  --clean_dir <クリーン画像フォルダ> \
  --wm_suffix v2 \
  --output results/benchmarks/phaseA_cnn_only.csv
```

**主なオプション：**

| オプション | 説明 |
|---|---|
| `--config` | YAML 設定ファイルのパス |
| `--input_dir` | 透かし入り画像のディレクトリ |
| `--clean_dir` | 比較用クリーン画像のディレクトリ（PSNR計算に使用） |
| `--wm_suffix` | 透かし入りファイルの末尾識別子（例: `v2` → `xxxv2.jpg` を処理） |
| `--output` | 結果 CSV の出力パス |
| `--det_threshold` | 検出成功閾値の上書き |

結果は `results/benchmarks/` 以下に CSV 形式で保存されます。

---

### 参考性能（44 枚の実画像で測定）

| 指標 | 値 | 目標 |
|---|---|---|
| 成功率 | **95.5%** | > 90% |
| 平均処理時間 | **1.5 秒/枚** | < 10 秒 |
| ピーク VRAM | **0.26 GB** | < 8 GB |
| 平均 PSNR | **23.5 dB** | > 20 dB |

> ※ ウォーターマークの種類・データセット・GPU によって結果は異なります。

---

### 参考文献・クレジット

- **元の UnMarker 論文・コード:**  
  UnMarker: A Universal Attack on Defensive Image Watermarking  
  <https://github.com/andrekassis/ai-watermark>

- **使用データセット:**  
  Large-scale Common Watermark Dataset  
  <https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset>

本リポジトリは研究・ベンチマーク目的で公開されています。  
商用利用や大規模使用の際は、上記オリジナル UnMarker の研究を参照・クレジットしてください。

---

## English

<a name="english"></a>

A fast, lightweight, and practical watermark removal pipeline using only a CNN-based detector (MobileNetV3).  
This repository provides a minimal, reproducible baseline for universal watermark removal with a focus on speed, memory efficiency, and simplicity.

### Overview

- **Purpose:** Remove visible watermarks using a 3-stage pipeline (frequency analysis → multi-scale attack → quality safeguard) guided only by a CNN-based detector.
- **Target hardware:** RTX 5070 Ti–class GPU or similar.
- **Design goals:** <10s per image, <8GB VRAM, >90% removal success.
- **Relation to EfficientUnMarker:** Distilled version of the full EfficientUnMarker project, containing only the CNN-based (MobileNetV3) detection and attack components.

### Pipeline Architecture

```
Watermarked Image
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1: Frequency Analysis        │
│  torch.fft spectrum analysis        │
│  → Generate frequency band mask     │
└──────────────┬──────────────────────┘
               │ band mask
               ▼
┌─────────────────────────────────────┐
│  Stage 2: Multi-scale Attack        │
│  Optimize at 256×256 (coarse)       │
│  → Upsample to original resolution  │
│  → Fine-tune at full resolution     │
│  (with early stopping & PSNR guard) │
└──────────────┬──────────────────────┘
               │ attacked image
               ▼
┌─────────────────────────────────────┐
│  Stage 3: Quality Refinement        │
│  (optional, disabled by default)    │
│  Adaptive low-pass filter           │
└──────────────┬──────────────────────┘
               │
               ▼
      Output Image (watermark removed)
```

### Features

- **Stage 1 – Frequency analysis:** `torch.fft`-based spectrum analysis + MobileNetV3 features to generate a frequency band mask.
- **Stage 2 – Multi-scale attack:** 256→original resolution optimization with early stopping and a soft PSNR penalty to avoid excessive degradation.
- **Stage 3 – Optional refinement:** Additional quality refinement stage (disabled by default in this baseline).
- **Dual-branch detector:** MobileNetV3 (quick) + DenseNet-121 (deep), with adaptive switching based on detection confidence.
- **Metrics:** Success flag, detection score, PSNR, SSIM, runtime, and peak VRAM usage.
- **Configurable:** YAML-based configuration for attack hyperparameters and stopping criteria.

### Dataset

Evaluation uses the **Large-scale Common Watermark Dataset**:

- Source: [Kaggle - Large-scale Common Watermark Dataset](https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset)
- Contains realistic visible watermarks for benchmarking removal algorithms.

### Quick Start

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Download dataset (optional)

Download the Large-scale Common Watermark Dataset from the link above and place it in `datasets/`.

#### Run benchmark (example)

```bash
python experiments/benchmark_cnn_only.py \
  --config configs/fast_cnn_only.yaml \
  --input_dir <watermarked_dir> \
  --clean_dir <clean_dir> \
  --wm_suffix v2 \
  --output results/benchmarks/phaseA_cnn_only.csv
```

- `--wm_suffix v2` matches files like `xxxv2.jpg` (watermarked) to `xxx.jpg` (clean).

#### Results

- Per-image metrics and aggregates are saved under `results/benchmarks/`.

### Directory Structure

```
core/
├── efficient_unmarker.py   # Main EfficientUnMarker class (pipeline controller)
├── detection.py            # Watermark detectors (MobileNetV3, DenseNet-121)
├── stage1_frequency.py     # Frequency analysis & band mask prediction
├── inpainting_cnn.py       # Partial Conv U-Net for inpainting (optional)
└── early_stopping.py       # Early stopping & progress tracking
utils/
├── image_processing.py     # FFT transforms, filters, multi-scale resize
├── metrics.py              # PSNR / SSIM computation
├── losses.py               # Loss functions
└── masks.py                # Mask generation and processing
configs/
└── fast_cnn_only.yaml      # YAML config (hyperparameters)
experiments/
└── benchmark_cnn_only.py   # Benchmark & evaluation script
datasets/                   # Place downloaded datasets here
requirements.txt
```

### Reference Performance

Measured on 44 real images from the Large-scale Common Watermark Dataset:

| Metric | Value | Target |
|---|---|---|
| Success rate | **95.5%** | > 90% |
| Avg runtime | **1.5 s / image** | < 10 s |
| Peak VRAM | **0.26 GB** | < 8 GB |
| Avg PSNR | **23.5 dB** | > 20 dB |

(Results may vary depending on watermark scheme, dataset, and GPU.)

### License / Attribution

This repository is intended for research and benchmarking.  
For commercial or large-scale use, please refer to and credit the original UnMarker work:

- UnMarker: A Universal Attack on Defensive Image Watermarking  
  <https://github.com/andrekassis/ai-watermark>

Dataset credit:
- Large-scale Common Watermark Dataset
  <https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset>

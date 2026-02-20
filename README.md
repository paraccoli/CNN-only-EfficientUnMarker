# CNN-only EfficientUnMarker

A fast, lightweight, and practical watermark removal pipeline using only a CNN-based detector (MobileNetV3).  
This repository provides a minimal, reproducible baseline for universal watermark removal with a focus on speed, memory efficiency, and simplicity.

---

## 📖 日本語解説 / Japanese Explanation

### このリポジトリは何をするのか？

**CNN-only EfficientUnMarker** は、画像に埋め込まれた **可視ウォーターマーク（透かし）を自動で除去する** パイプラインです。

ウォーターマーク除去を「攻撃問題」として捉え、CNNベースの検出器（MobileNetV3）を使ってウォーターマークが残っているかどうかを判定しながら、画像を反復的に最適化することで透かしを消去します。

#### 目標仕様

| 項目 | 目標値 |
|---|---|
| 処理時間 | 1枚あたり 10秒以下 |
| GPU メモリ | 8GB 以下 |
| 除去成功率 | 90% 以上 |
| 対象ハードウェア | RTX 5070 Ti 相当以上の GPU |

---

### 仕組みの全体像（3ステージパイプライン）

```
入力画像（ウォーターマークあり）
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: 周波数解析         │  ← torch.fft + MobileNetV3
│  どの周波数帯に透かしがあるか分析 │
└──────────────┬──────────────┘
               │ 周波数バンドマスク
               ▼
┌─────────────────────────────┐
│  Stage 2: マルチスケール攻撃  │  ← 主要ステージ
│  低解像度→高解像度で最適化     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 3: 品質改善（任意）   │  ← デフォルト無効
│  ローパスフィルタで後処理     │
└──────────────┬──────────────┘
               │
               ▼
        出力画像（透かし除去済み）
```

---

### 各ステージの詳細解説

#### Stage 1 — 周波数解析（`core/stage1_frequency.py`）

**目的：** 画像のどの周波数帯域にウォーターマークが含まれているかを推定し、攻撃対象の帯域を絞り込む。

**処理内容：**

1. `torch.fft.rfft2` で画像をフーリエ変換し、周波数スペクトルを取得
2. 周波数空間を8つの同心円状の帯域（バンド）に分割
3. 各バンドのエネルギー・分散を計算
4. MobileNetV3の画像エンコーダと周波数特徴を組み合わせ、融合ネットワークで「どのバンドを攻撃すべきか」を示すマスクを出力

```
MobileNetV3特徴量（画像の見た目）
        +
周波数統計量（エネルギー・分散）
        ↓
   融合ネットワーク（全結合層）
        ↓
  8次元の帯域マスク（0〜1）
```

処理時間は約 5〜10ms と非常に高速です。

---

#### Stage 2 — マルチスケール攻撃（`core/efficient_unmarker.py`）

**目的：** 検出スコアを下げながら（＝ウォーターマークを消しながら）、画像品質をできるだけ維持する。

**処理内容：**

まず **低解像度（256×256）** で粗い最適化を行い、次にオリジナル解像度に戻して **精細な最適化** を行います。

```
低解像度最適化（40回）  → アップスケール →  高解像度最適化（60回）
(256×256)                                 (元の解像度)
```

最適化では以下の **損失関数** を最小化します：

| 損失項 | 役割 |
|---|---|
| `detector_weight × det_loss` | ウォーターマーク検出スコアを下げる（メイン） |
| `psnr_penalty` | PSNR が下限を割ると追加ペナルティ（品質保護） |
| `freq_weight × freq_loss` | 周波数的な変化を促進（オプション） |
| `smooth_weight × smooth_loss` | 高周波ノイズを抑制（オプション） |

**早期終了（Early Stopping）** の条件：
- 検出スコアがしきい値（デフォルト 0.5）を下回ったとき → **成功で終了**
- 損失の変動が収束したとき → **収束で終了**
- PSNRが最低値（デフォルト 18 dB）を下回り続けたとき → **品質ガードで終了**

---

#### Stage 3 — 品質改善（`core/efficient_unmarker.py`）

**目的：** 攻撃後の画像のノイズを滑らかにして、見た目の品質を向上させる。

**処理内容：** バターワースローパスフィルタ（`adaptive_low_pass_filter`）で高周波ノイズを除去し、元画像と 9:1 でブレンドします。

デフォルトでは **無効**（`configs/fast_cnn_only.yaml` の `stage3.enabled: false`）です。速度を優先したい場合はこのまま使用してください。

---

### 検出器（Detector）の仕組み（`core/detection.py`）

ウォーターマークが「まだ残っているか」を判定する心臓部です。

#### 3つのモード

| モード | 使用モデル | 特徴 |
|---|---|---|
| `quick` | MobileNetV3 のみ | 高速・軽量（本リポジトリのデフォルト） |
| `deep` | DenseNet121 のみ | 精度重視・低速 |
| `adaptive` | MobileNetV3 + DenseNet121 | 不確かな場合のみ深いモデルを追加使用 |

```
入力画像
   │
   ▼
MobileNetV3 → 検出スコア（0〜1）
   │
   ├─ スコアが 0.3〜0.7 の場合（不確かな領域）
   │    ↓
   │  DenseNet121 も使って重み付き平均
   │    （QuickWeight: 0.6, DeepWeight: 0.4）
   │
   └─ スコアが確定的な場合 → MobileNetV3の結果をそのまま使用
```

`adaptive` モードは精度と速度のバランスを自動で調整します。

---

### 設定ファイル（`configs/fast_cnn_only.yaml`）

YAMLで全パラメータを管理しています。主な設定項目：

```yaml
device: cuda               # GPUを使用
target_detection: 0.5      # この値以下なら「除去成功」
max_iterations: 120        # 最大最適化イテレーション数
learning_rate: 0.01        # Adam最適化の学習率

detector:
  mode: quick              # MobileNetV3のみ使用（高速）

stage1:
  enabled: true            # 周波数解析を有効化

stage2:
  low_res_size: 256        # 低解像度フェーズのサイズ
  low_iters: 40            # 低解像度での反復回数
  high_iters: 60           # 高解像度での反復回数

stage3:
  enabled: false           # 品質改善は無効（速度優先）

stopping:
  psnr_min: 18.0           # PSNRがこの値を下回ったら早期終了
  psnr_patience: 4         # 何回連続で下回ったら終了するか

loss:
  detector_weight: 1.0     # 検出スコア損失の重み
  psnr_penalty: 0.1        # PSNRペナルティの強さ
  psnr_penalty_floor: 18.0 # PSNRペナルティが発動する閾値
```

---

### ファイル構成と役割

```
CNN-only-EfficientUnMarker/
│
├── core/                            # メインロジック
│   ├── efficient_unmarker.py        # EfficientUnMarker クラス（全体制御）
│   ├── detection.py                 # ウォーターマーク検出器
│   │                                  （MobileNetV3 / DenseNet121）
│   ├── stage1_frequency.py          # Stage1: 周波数解析モジュール
│   ├── early_stopping.py            # 早期終了・収束判定ロジック
│   └── inpainting_cnn.py            # （追加の修復モジュール）
│
├── utils/                           # ユーティリティ
│   ├── image_processing.py          # 周波数変換・フィルタ・リサイズ
│   ├── metrics.py                   # PSNR・SSIM 計算
│   ├── losses.py                    # 損失関数
│   └── masks.py                     # マスク生成
│
├── configs/
│   └── fast_cnn_only.yaml           # 高速動作用設定（デフォルト）
│
├── experiments/
│   └── benchmark_cnn_only.py        # ベンチマーク実行スクリプト
│
├── datasets/                        # データセット配置場所（手動ダウンロード）
├── requirements.txt                 # 依存ライブラリ一覧
└── README.md                        # このファイル
```

---

### クイックスタート

#### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

> RTX 5070 Ti など最新世代の GPU (sm_120+) を使う場合は CUDA 12.8 対応 PyTorch が必要です：
> ```bash
> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```

#### 2. データセットのダウンロード（任意）

[Kaggle - Large-scale Common Watermark Dataset](https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset) からダウンロードして `datasets/` に配置します。

#### 3. ベンチマーク実行

```bash
python experiments/benchmark_cnn_only.py \
  --config configs/fast_cnn_only.yaml \
  --input_dir <ウォーターマーク画像のディレクトリ> \
  --clean_dir <クリーン画像のディレクトリ> \
  --wm_suffix v2 \
  --output results/benchmarks/phaseA_cnn_only.csv
```

- `--wm_suffix v2`：`xxxv2.jpg`（ウォーターマークあり）と `xxx.jpg`（クリーン）を自動対応付け
- 結果 CSV は `results/benchmarks/` 以下に保存されます

#### 4. 動作確認（ダミー画像で単体テスト）

```bash
python core/efficient_unmarker.py
```

512×512 のランダム画像で全パイプラインが動作することを確認できます。

---

### 参考性能（実測値）

Large-scale Common Watermark Dataset の実画像 44 枚で計測：

| 指標 | 実測値 | 目標値 |
|---|---|---|
| 除去成功率 | **95.5%** | 90% 以上 |
| 平均処理時間 | **1.5 秒 / 枚** | 10 秒以下 |
| ピーク VRAM | **0.26 GB** | 8 GB 以下 |
| 平均 PSNR | **23.5 dB** | — |

※ 使用する GPU・ウォーターマークの種類・データセットによって結果は変わります。

---

### 依存ライブラリ

| ライブラリ | 役割 |
|---|---|
| `torch`, `torchvision` | ディープラーニング基盤 |
| `timm` | MobileNetV3 / DenseNet121 の事前学習済みモデル |
| `numpy`, `Pillow` | 画像読み込み・数値処理 |
| `opencv-python` | 画像処理補助 |
| `pyyaml` | YAML設定ファイル読み込み |
| `lpips` | 知覚的画像品質評価 |
| `wandb` | 実験ログ（任意） |

---

### ライセンス / 参考文献

本リポジトリは研究・ベンチマーク目的での使用を想定しています。  
商用利用や大規模利用の際は、以下のオリジナル研究を参照・引用してください：

- **UnMarker: A Universal Attack on Defensive Image Watermarking**  
  <https://github.com/andrekassis/ai-watermark>

データセット：
- **Large-scale Common Watermark Dataset**  
  <https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset>

---

## Overview (English)

- **Purpose:** Remove visible watermarks using a 3-stage pipeline (frequency analysis → multi-scale attack → quality safeguard) guided only by a CNN-based detector.
- **Target hardware:** RTX 5070 Ti–class GPU or similar.
- **Design goals:** <10s per image, <8GB VRAM, >90% removal success.
- **Relation to EfficientUnMarker:** Distilled version of the full EfficientUnMarker project, containing only the CNN-based (MobileNetV3) detection and attack components.

## Features

- **Stage 1 – Frequency analysis:** `torch.fft`-based spectrum analysis + MobileNetV3 features to generate a frequency band mask.
- **Stage 2 – Multi-scale attack:** 256→original resolution optimization with early stopping and a soft PSNR penalty to avoid excessive degradation.
- **Stage 3 – Optional refinement:** Additional quality refinement stage (disabled by default in this baseline).
- **Metrics:** Success flag, detection score, PSNR, runtime, and peak VRAM usage.
- **Configurable:** YAML-based configuration for attack hyperparameters and stopping criteria.

## Dataset

Evaluation uses the **Large-scale Common Watermark Dataset**:

- Source: [Kaggle - Large-scale Common Watermark Dataset](https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset)
- Contains realistic visible watermarks for benchmarking removal algorithms.

## Quick Start

### Install dependencies

```txt
pip install -r requirements.txt
```

### Download dataset (optional)

Download the Large-scale Common Watermark Dataset from the link above and place it in `datasets/`.

### Run benchmark (example)

```txt
python experiments/benchmark_cnn_only.py \
  --config configs/fast_cnn_only.yaml \
  --input_dir <watermarked_dir> \
  --clean_dir <clean_dir> \
  --wm_suffix v2 \
  --output results/benchmarks/phaseA_cnn_only.csv
```

- `--wm_suffix v2` matches files like `xxxv2.jpg` (watermarked) to `xxx.jpg` (clean).

### Results

- Per-image metrics and aggregates are saved under `results/benchmarks/`.

## Directory Structure

```txt
core/          # Main attack pipeline and CNN detector
utils/         # Image I/O, transforms, metrics
configs/       # YAML configs (e.g., fast_cnn_only.yaml)
experiments/   # Benchmark and evaluation scripts
datasets/      # (Place downloaded datasets here)
requirements.txt
README.md
```

## Reference Performance

Measured on 44 real images from the Large-scale Common Watermark Dataset:

- Success rate: **95.5%**
- Avg runtime: **1.5 s / image**
- Peak VRAM: **0.26 GB**
- Avg PSNR: **23.5 dB**

(Results may vary depending on watermark scheme, dataset, and GPU.)

## License / Attribution

This repository is intended for research and benchmarking.  
For commercial or large-scale use, please refer to and credit the original UnMarker work:

- UnMarker: A Universal Attack on Defensive Image Watermarking  
  <https://github.com/andrekassis/ai-watermark>

Dataset credit:
- Large-scale Common Watermark Dataset
  <https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset>

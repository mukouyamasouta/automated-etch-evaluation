# 野島氏の学習済み重みによるEnd-to-End実行

## 使用する重み

- Detection: `Detection/20250208 Faster R-CNN/TrainingV163_pthfiles/train_30.pth`
- Segmentation: `Segmentation/20250108 U-Net/TrainingV204_pthfiles/train_87.pth`

重み、Dataset、生成出力はGit管理対象外である。GitHubでは実行コードと条件のみを管理する。

## 実行コード

`End-To-End/20260925 U-Net--Faster R-CNN/20260925EndToEnd_Seg204_Det163_nojima_weights.py`

元の`20250526EndToEnd_Seg204_Det163_1.py`は変更せず保持している。追加版は以下の実行環境の差だけを調整する。

- CUDAの有無で`NUM_WORKERS`と`PIN_MEMORY`を自動切替する。
- CPUとCUDAのどちらでもDet163 / Seg204重みを読み込む。
- Det163を読み込む前の不要なCOCO重みダウンロードを行わない。
- 重みとDetection test CSVの存在を推論開始前に確認する。

## 事前確認

リポジトリ直下で次を実行する。推論は開始されない。

```bash
bash scripts/run_end_to_end_nojima.sh --check
```

## 実行

```bash
bash scripts/run_end_to_end_nojima.sh
```

Macでは`caffeinate`を自動的に使用し、実行中のスリープを防止する。実行ログは`/tmp/end_to_end_nojima_YYYYMMDD_HHMMSS.log`に保存する。

## 出力

`End-To-End/20260925 U-Net--Faster R-CNN/InferenceSeg204_Det163_1_outputs/`

処理はDetection test CSVの9枚を対象に、Det163検出、構造切出し、Seg204セグメンテーション、寸法・形状計測の順で実行する。

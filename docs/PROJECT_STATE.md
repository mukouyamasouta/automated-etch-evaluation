# プロジェクト状態

最終更新日: 2026-09-25

## 目的

野島氏の手法Bを、Dataset確認、Faster R-CNNの重み作成、U-Netの重み作成、End-To-End推論、寸法・形状評価まで追試する。

## リポジトリ

- ローカル: `/Volumes/met-info/Research Progress/Mukoyama/Gan系トレース手法B`
- GitHub: `https://github.com/mukouyamasouta/automated-etch-evaluation`
- 現在の作業ブランチ: `experiment/seg204-local-mac`

## 完了済み

- GaN主実験用Datasetを`Dataset/`へ配置した。
- 6個のCSVに含まれる画像パスを`20260925 Image_Dataset`へ更新した。
- CSVの全1,104パス参照が実在することを確認した。
- Detectionの基準コードとしてDet163を選定した。
- Segmentationの基準コードとしてSeg204を選定した。
- End-To-Endの基準コードとしてSeg204 / Det163を1本に絞った。
- 旧`.7z`履歴は追試側から除去し、原本を`Model_backup`に残した。
- コード内のCSV・Dataset参照を、スクリプト位置基準で解決するよう更新した。
- Gitリポジトリを作成し、GitHubの`main`へpushした。
- Det163の基準コードを保持したまま、Mac用ローカル動作確認版を作成した。
- Git初心者向けの継続的な説明方針と操作ガイドを追加した。
- Mac内蔵ストレージにPython 3.12仮想環境を作り、PyTorchとTorchVisionのimportを確認した。重み学習はまだ実行していない。
- `experiment/det163-local-mac`ブランチで、バックアップの完全無変更Det163とMac版の対応関係を整理した。
- Mac版の変更へ`MAC-*`、`ENV-*`、`TRACE-*`番号と変更理由を記載した。
- 元コードとMac版の変更理由表および保存済みdiffを追加した。
- Det163 Mac版のepochを5へ変更し、コミット`a6ff6cd`でGitHubへ反映した。重み学習はまだ実行していない。
- Seg204の完全無変更版、Mac用5 epochパイロット版、変更理由表、保存済みdiffを作成した。

主要コミット:

- `8ba9a4b` — 手法Bの学習準備
- `b465281` — End-To-Endコードの選定
- `2d4fe6e` — Mac用Det163動作確認版の初回追加
- `9d138f7` — Mac環境と実行待ち状態の記録

## 採用コード

### Detection

`Detection/20250208 Faster R-CNN/20250525FasterRCNN_TrainingV163.py`

- モデル: Faster R-CNN ResNet50-FPN
- 本設定: 30 epoch、batch size 8、learning rate 0.01
- 入力: `Original/Images`
- 教師: Detection CSVの`boxes`と`labels`
- End-To-Endが使用する重み: `TrainingV163_pthfiles/train_30.pth`

完全無変更の参照元:

`Detection/reference/20250525FasterRCNN_TrainingV163_original.py`

Mac用動作確認版:

`Detection/20250208 Faster R-CNN/20260924FasterRCNN_TrainingV163_local_mac.py`

変更理由表:

`docs/DETECTION_MAC_CHANGES.md`

### Segmentation

`Segmentation/20250108 U-Net/20250526UNet_TrainingV204.py`

- モデル: U-Net
- 本設定: 100 epoch、batch size 16、256×256
- 入力: `Structure_padded/Images`
- 教師: `Structure_padded/Labels`
- End-To-Endが使用する重み: `TrainingV204_pthfiles/train_87.pth`

完全無変更の参照元:

`Segmentation/reference/20250526UNet_TrainingV204_original.py`

Mac用パイロット版:

`Segmentation/20250108 U-Net/20260925UNet_TrainingV204_local_mac.py`

- 5 epoch、batch size 1
- 学習8構造（左右反転込み16サンプル）、検証2構造、テスト2構造
- 256×256、前処理、モデル、損失関数は元コードを維持
- CPU固定・4スレッド
- 出力重み: `TrainingV204_local_mac_pthfiles/train_1.pth`～`train_5.pth`
- 精度再現用ではなく、Mac上で学習・評価・保存の流れを確認するための設定

変更理由表:

`docs/SEGMENTATION_MAC_CHANGES.md`

### End-To-End

`End-To-End/20260925 U-Net--Faster R-CNN/20250526EndToEnd_Seg204_Det163_1.py`

- Detection test.csvの9枚を対象とする。
- Det163 epoch 30 → 構造切出し → Seg204 epoch 87 → 寸法計測・正解比較の順で処理する。

## Dataset

```text
Dataset/
├── 20260925 CSV_Data/
│   ├── Detection/Original/
│   │   ├── train.csv    75枚
│   │   ├── val.csv       8枚
│   │   └── test.csv      9枚
│   └── Segmentation/Detected_padded/
│       ├── train.csv    375構造
│       ├── val.csv       40構造
│       └── test.csv      45構造
└── 20260925 Image_Dataset/
    ├── Original/
    │   ├── Images/      全体SEM
    │   └── Labels/      全体マスク
    └── Structure_padded/
        ├── Images/      構造単位SEM
        └── Labels/      構造単位マスク
```

Datasetは外付けボリュームにのみ置き、Git管理しない。

## 実行環境

- MacBook Air `MacBookAir10,1`
- Apple M1、8コア
- メモリ16GB
- 元コードはCUDAがなければCPUを使用するため、このMacではCPU実行になる。
- Detectionの元画像は1280×806である。
- Mac用仮想環境: `/Users/mu-sota/.venvs/gan-method-b`
- Python 3.12.13、PyTorch 2.14.0、TorchVision 0.29.0
- Seg204に必要な`segmentation_models_pytorch`は未導入。実行前に`Segmentation/requirements-segmentation-mac.txt`から追加する。

外付けネットワークボリューム内の`.venv`は、小ファイルの配置不良により正常に構築できなかった。学習時は上記のMac内蔵ストレージ側のPythonを明示して使う。

本設定のローカル実行は、長時間のCPU高負荷、メモリスワップ、発熱による速度低下、プロセス強制終了、端末の応答低下の可能性がある。本実験には研究室のNVIDIA GPU搭載PCまたは計算サーバーを推奨する。

## 次に行うこと

別ターミナルでSeg204用依存パッケージを追加し、import確認後にMac用5 epochパイロットを実行する。Codex側からはまだ学習を実行しない。

対象コード:

`Segmentation/20250108 U-Net/20260925UNet_TrainingV204_local_mac.py`

パイロット条件:

```text
version: 204_local_mac
train: 8構造（左右反転込み16サンプル）
val: 2構造
test: 2構造
epoch: 5
batch size: 1
num workers: 0
pin memory: False
drop last: False
image size: 256×256
device: CPU
CPU threads: 4
```

目的はデータ読込み、forward/backward、損失・IoU記録、評価画像、`train_1.pth`～`train_5.pth`保存までの確認である。精度比較には使用しない。

実行コマンド:

```bash
cd "/Volumes/met-info/Research Progress/Mukoyama/Gan系トレース手法B"
/Users/mu-sota/.venvs/gan-method-b/bin/python -m pip install \
  -r "Segmentation/requirements-segmentation-mac.txt"

/usr/bin/time -p /Users/mu-sota/.venvs/gan-method-b/bin/python -u \
  "Segmentation/20250108 U-Net/20260925UNet_TrainingV204_local_mac.py" \
  2>&1 | tee "/tmp/seg204_local_mac_$(date +%Y%m%d_%H%M%S).log"
```

## 未実施

- Det163のローカル5 epoch実行
- 研究室GPU環境でのDet163本学習
- Seg204依存パッケージのMac環境への追加
- Seg204のローカル5 epoch実行
- 研究室GPU環境でのSeg204本学習
- End-To-End実行
- 論文値との比較

## 作業ツリーに関する注意

2026-09-24時点で、ユーザーによる次の未コミット変更が存在する。

- ルートの`README.md`が削除状態
- ルートの`requirements.txt`が削除状態
- `docs/README.md`が未追跡
- `docs/requirements.txt`が未追跡
- `branch`と`--show-current`が未追跡

ファイルを`docs/`へ移動した操作と見られるが、ユーザーの変更なので無断で戻したり、別のコミットへ混ぜたりしない。整理方針を確認してから扱う。

## 更新ルール

作業が進んだら、この文書の「完了済み」「次に行うこと」「未実施」を更新する。実験値やエラーの詳細は`docs/EXPERIMENT_LOG.md`へ記録し、この文書には現在の結論だけを残す。

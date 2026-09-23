# GaNエッチング形状自動評価 — 手法Bトレース

野島氏の実験における**手法B（物体検出 → 構造ごとのセグメンテーション）**を、学習済み重みの作成から追試するための作業リポジトリです。

## 手法Bの流れ

```text
全体SEM画像
  │
  ├─ Detection V163（Faster R-CNN）
  │      └─ 構造ごとの矩形を検出
  │
  └─ 検出矩形で構造を1個ずつ切り出す
         └─ Segmentation V204（U-Net）
                └─ 基板／背景の二値マスク
```

## このリポジトリで管理するもの

```text
Gan系トレース手法B/
├── README.md
├── requirements.txt
├── .gitignore
├── Detection/
│   └── 20250208 Faster R-CNN/
│       └── 20250525FasterRCNN_TrainingV163.py
├── Segmentation/
│   └── 20250108 U-Net/
│       └── 20250526UNet_TrainingV204.py
├── Dataset/                         # 実データ。Git管理外
└── docs/
    ├── Dataset_ファイル構成ガイド.html
    └── FILE_INVENTORY.md
```

バックアップにある多数の旧版`.py.7z`、過去の出力、全epochの`.pth`はコピーしていません。手法Bの基準となる組合せは、既存のEnd-To-End実験名にも使われている **Seg204 / Det163** です。2本の学習スクリプトはローカル補助モジュールに依存せず、必要なモデル定義とデータ読込み処理を内部に持っています。

## Dataset

データ本体は容量と研究データ管理の都合からGitHubへ登録しません。作業環境では次の配置を前提とします。

```text
Dataset/
├── 20260925 CSV_Data/
│   ├── Detection/Original/
│   │   ├── train.csv                 # 75枚
│   │   ├── val.csv                   # 8枚
│   │   └── test.csv                  # 9枚
│   └── Segmentation/Detected_padded/
│       ├── train.csv                 # 375構造
│       ├── val.csv                   # 40構造
│       └── test.csv                  # 45構造
└── 20260925 Image_Dataset/
    ├── Original/
    │   ├── Images/                    # Detection入力
    │   └── Labels/                    # 全体正解マスク
    └── Structure_padded/
        ├── Images/                    # U-Net入力
        └── Labels/                    # U-Net教師マスク
```

CSV内の画像参照は`20260925 Image_Dataset`へ更新済みです。学習スクリプトも実行時のカレントディレクトリに依存せず、スクリプト自身の位置から上記Datasetを解決します。

さらに詳しい役割、データ生成関係、手法Aとの違いは[Dataset構成ガイド](docs/Dataset_ファイル構成ガイド.html)を参照してください。

## 環境構築

Python仮想環境を作り、依存パッケージを導入します。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDAを使う場合、`torch`と`torchvision`は実行環境のCUDAバージョンに合う方法で先に導入してください。Detectionの初回実行時は、TorchVisionがCOCO事前学習済みFaster R-CNN重みを取得するため、インターネット接続が必要です。

## 実行順序

### 1. Faster R-CNNの重みを作る

```bash
python "Detection/20250208 Faster R-CNN/20250525FasterRCNN_TrainingV163.py"
```

`Detection/20250208 Faster R-CNN/TrainingV163_pthfiles/`へ各epochの重み、`TrainingV163_outputs/`へ評価結果が生成されます。

### 2. U-Netの重みを作る

```bash
python "Segmentation/20250108 U-Net/20250526UNet_TrainingV204.py"
```

`Segmentation/20250108 U-Net/TrainingV204_pthfiles/`へ各epochの重み、`TrainingV204_outputs/`へ評価結果が生成されます。

DetectionとSegmentationの学習データは既に作成済みなので、重みを作るだけならこの2本の順序は入れ替えられます。新しいSEMからEnd-To-End推論する段階では、Det163とSeg204の両方の採用重みが必要です。

## Git/GitHub運用

Gitにはコード、設定、説明文書だけを登録します。Dataset、`.pth`、学習出力、`.7z`履歴は`.gitignore`で除外しています。実験条件を変更するときは旧ファイルを複製して連番を増やす代わりに、変更をコミットし、条件はコミットメッセージまたは実験記録へ残します。

リモート: <https://github.com/mukouyamasouta/automated-etch-evaluation>

## 原本からの変更点

アルゴリズムとハイパーパラメータは変更していません。次の運用上の変更のみ加えています。

1. CSVフォルダ名を`20250515 CSV_Data`から`20260925 CSV_Data`へ更新。
2. CSVと出力先をスクリプト位置基準の絶対パスとして解決。
3. `.7z`から実行可能な`.py`を展開。

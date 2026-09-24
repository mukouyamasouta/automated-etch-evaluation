# Seg204元コードからMac用パイロット版への変更

## ファイルの関係

```text
完全無変更の元コード
Segmentation/reference/20250526UNet_TrainingV204_original.py
        │
        │ 複製して、下表の変更だけを追加
        ▼
Mac用パイロットコード
Segmentation/20250108 U-Net/
└── 20260925UNet_TrainingV204_local_mac.py
```

完全無変更版は`Model_backup`の`20250526UNet_TrainingV204.py.7z`から展開したものである。元コードには説明コメントを追加せず、比較用として保存する。

元コードのSHA-256は次である。

```text
e274905f599b05899f76e164a9b47b1be122a4f8083955abc473a592042302a9
```

この値は`.py.7z`から一時展開したファイルと、リポジトリの完全無変更版で一致している。

## Mac向け計算量調整

| ID | 設定項目 | 元コード | Mac用 | 変更理由 | 結果への影響 |
|---|---|---:|---:|---|---|
| `SEG-MAC-01` | バージョン名 | `204` | `204_local_mac` | 本実験の重み・出力を上書きしない | 精度への影響なし |
| `SEG-MAC-02` | epoch | 100 | 5 | 重み保存と損失推移を短時間で確認する | 大。本実験用の重みとして使用不可 |
| `SEG-MAC-03` | batch size | 16 | 1 | 16GB MacでU-Netの中間特徴によるメモリ使用量を抑える | あり。勾配更新条件が変わる |
| `SEG-MAC-04` | workers | 4 | 0 | macOSの別プロセス読込みトラブルを避ける | 精度への影響なし。読込み速度に影響 |
| `SEG-MAC-05` | pin memory | `True` | `False` | CUDAを使用しないため固定メモリが不要 | 精度への影響なし |
| `SEG-MAC-06` | drop last | `True` | `False` | 少数データの最後の端数バッチを捨てない | 本パイロットの学習内容に影響 |
| `SEG-MAC-07` | 学習元データ | 375構造 | 8構造 | 動作確認時間を短縮する | 大。精度比較不可 |
| `SEG-MAC-07` | 学習時サンプル | 750（左右反転込み） | 16（左右反転込み） | 元コードの左右反転は維持する | 大。精度比較不可 |
| `SEG-MAC-07` | 検証データ | 40構造 | 2構造 | 評価処理を短縮する | 大。精度比較不可 |
| `SEG-MAC-07` | テストデータ | 45構造 | 2構造 | 評価・画像保存を短縮する | 大。精度比較不可 |
| `SEG-MAC-08` | device | CUDAまたはCPU | CPU固定 | M1 Macで使用する経路を明示する | 原則なし。速度に影響 |
| `SEG-MAC-09` | CPUスレッド | 制限なし | 4 | 発熱とMac全体の応答低下を抑える | 精度への影響なし |

## 変更しなかった学習条件

次の項目はSeg204の処理内容を保つため変更していない。

| 項目 | 維持した設定 | 理由 |
|---|---|---|
| 入力サイズ | 256×256 | セグメンテーション対象の画素スケールを変えないため |
| データ増強 | 元画像と左右反転の2通り | 元実験と同じ前処理を通すため |
| 輝度処理 | Yチャンネルのヒストグラム均等化 | 元コードのコントラスト処理を保つため |
| エッジ処理 | Laplacianを0.5倍して加算 | 元コードの形状境界強調を保つため |
| モデル | 独自実装U-Net | ネットワーク構造を変えないため |
| 損失関数 | Tversky Loss（alpha=1、beta=1） | 元コードのIoU系損失を保つため |
| 学習率 | 0.001 | batch size以外の最適化条件を不用意に変えないため |
| 学習用augmentation | `True` | 少数データでも元コードと同じ左右反転を適用するため |

`STEP_SIZE`は元コードと同じ式`4000 // BATCH_SIZE * 100`を残している。batch sizeの変更に伴い数値は変わるが、元設定・Mac設定とも実際の総更新回数より大きいため、この実行中に学習率減衰は発生しない。

## Mac性能以外の変更

| ID | 内容 | 理由 | Datasetへの影響 |
|---|---|---|---|
| `ENV-01` | スクリプト位置から`20260925 CSV_Data`を解決 | 実行時のカレントディレクトリに依存させない | CSV内容は変更しない |
| `TRACE-01` | `run_config.json`を保存 | 実験条件を後から確認できるようにする | なし |
| `TRACE-02` | NumPyとPyTorchのseedを42に固定 | 同じ条件で再実行しやすくする | なし |

## CSVの扱い

Mac用コードはCSVファイルを書き換えない。375/40/45行を読み込んだ後、Pythonのメモリ上のDataFrameだけを先頭8/2/2件へ絞る。

```python
train_df = train_df.iloc[:TRAIN_SAMPLE_COUNT].copy()
val_df = val_df.iloc[:VAL_SAMPLE_COUNT].copy()
test_df = test_df.iloc[:TEST_SAMPLE_COUNT].copy()
```

したがって、Mac用実行後もCSVの行数、画像パス、マスクパス、train/val/testの所属は変化しない。

## 出力

```text
Segmentation/20250108 U-Net/
├── TrainingV204_local_mac_pthfiles/
│   ├── train_1.pth
│   └── ... train_5.pth
└── TrainingV204_local_mac_outputs/
    ├── run_config.json
    ├── 損失・IoU記録
    └── test_epoch1～5の確認画像
```

これらは`.gitignore`で除外され、GitHubには登録しない。

## コード内で変更理由を探す

```bash
rg -n '\[(SEG-MAC|ENV|TRACE)-' \
  "Segmentation/20250108 U-Net/20260925UNet_TrainingV204_local_mac.py"
```

## 差分を見る

GitHubでは`docs/diffs/Seg204_original_to_local_mac.diff`を開く。ローカルでは次を実行する。

```bash
git diff --no-index -- \
  "Segmentation/reference/20250526UNet_TrainingV204_original.py" \
  "Segmentation/20250108 U-Net/20260925UNet_TrainingV204_local_mac.py"
```

この比較コマンドはファイルを変更しない。

## 実行前の依存パッケージ

この仮想環境は作成時にpipが入っていなかったため、最初にPython同梱の`ensurepip`で復旧する。その後、別ターミナルからSeg204用パッケージを追加する。

```bash
/Users/mu-sota/.venvs/gan-method-b/bin/python -m ensurepip --upgrade

/Users/mu-sota/.venvs/gan-method-b/bin/python -m pip install \
  -r "Segmentation/requirements-segmentation-mac.txt"
```

2026-09-25に`segmentation_models_pytorch 0.5.0`と`tqdm 4.70.1`の導入およびimport確認を完了した。既存のPyTorch 2.14.0とTorchVision 0.29.0は変更されていない。

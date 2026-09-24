# Det163元コードからMac用動作確認版への変更

## ファイルの関係

```text
完全無変更の元コード
Detection/reference/20250525FasterRCNN_TrainingV163_original.py
        │
        │ 複製して、下表の変更だけを追加
        ▼
Mac用動作確認コード
Detection/20250208 Faster R-CNN/
└── 20260924FasterRCNN_TrainingV163_local_mac.py
```

元コードのSHA-256は次である。

```text
4d2090e2b8ab7daede3f6a00b254d9aa8b3de16da67b634ef5b43e46f91f062b
```

これは`Model_backup`の`.py.7z`から直接読み出した内容のSHA-256と一致する。元コードには説明コメントを追加せず、完全無変更で保存する。

## Mac向け計算量調整

| ID | 設定項目 | 元コード | Mac用 | 変更理由 | 結果への影響 |
|---|---|---:|---:|---|---|
| `MAC-01` | バージョン名 | `163` | `163_local_mac_threshold_sweep` | 元実験と最初のMac実験の重み・出力を上書きしない | 精度への影響なし |
| `MAC-02` | epoch | 30 | 5 | 重み保存と検出傾向を短時間で確認する | 大。本実験の重みとして使用不可 |
| `MAC-03` | batch size | 8 | 1 | 16GB Macのメモリ使用量を抑える | あり。勾配更新条件が変わる |
| `MAC-04` | workers | 4 | 0 | macOSの別プロセス読込みトラブルを避ける | 精度への影響なし |
| `MAC-05` | pin memory | `True` | `False` | CUDAを使用しないため不要 | 精度への影響なし |
| `MAC-06` | drop last | `True` | `False` | 少数データを捨てない | 本確認版の学習内容に影響 |
| `MAC-07` | 学習画像 | 75枚 | 8枚 | 動作確認時間を短縮する | 大。精度比較不可 |
| `MAC-07` | 検証画像 | 8枚 | 2枚 | 評価処理を短縮する | 大。精度比較不可 |
| `MAC-07` | テスト画像 | 9枚 | 2枚 | 評価処理を短縮する | 大。精度比較不可 |
| `MAC-08` | モデル入力 | 標準 | min 400 / max 640 | 特徴マップの計算量を減らす | あり。検出精度が変わり得る |
| `MAC-09` | device | CUDAまたはCPU | CPU固定 | M1上で対応が安定する経路を明示する | 原則なし。速度に影響 |
| `MAC-10` | CPUスレッド | 制限なし | 4 | 発熱とMac全体の応答低下を抑える | 精度への影響なし |

## Mac性能以外の変更

| ID | 内容 | 理由 | Datasetへの影響 |
|---|---|---|---|
| `ENV-01` | スクリプト位置から`20260925 CSV_Data`を解決 | 実行するカレントディレクトリに依存させない | CSV内容は変更しない |
| `TRACE-01` | `run_config.json`を保存 | 実験条件を後から確認できるようにする | なし |
| `TRACE-02` | 乱数seedを42に固定 | 同じ条件で再実行しやすくする | なし |

## 検証データによる信頼度閾値調査

最初のMac 5 epoch実験では、固定閾値0.9を超えるBBoxが検証・テストとも0件だった。今回の診断版では、各epochの検証データを一度だけモデル推論し、同一の生予測へ次の閾値を適用する。

```text
0.1 / 0.3 / 0.5 / 0.7 / 0.9
```

| ID | 内容 | 理由 | 学習・重みへの影響 |
|---|---|---|---|
| `EVAL-01` | 検証データだけで5閾値を比較 | テストデータを閾値選定へ使う情報漏洩を避ける | なし |
| `EVAL-01` | 閾値ごとのBBox画像を保存 | 低い閾値で候補BBoxが出ているか目視確認する | なし |
| `EVAL-01` | BBox数、件数一致率、IoU、Precision、Recall、F1をCSV/JSONへ保存 | 「BBoxが出た」だけでなく位置と過検出・見逃しを比較する | なし |
| `EVAL-01` | BBox数・F1・対応IoUの閾値比較グラフを保存 | 5閾値の傾向と選択値を一目で確認する | なし |
| `EVAL-01` | 検証F1を優先して候補閾値を選択 | テスト結果を見ずにテスト用閾値を決める | なし |
| `EVAL-02` | IoU 0.5以上を正解対応とする | TP/FP/FNを一貫した基準で数える | なし |

候補閾値は、検証F1、BBox数完全一致率、BBox数誤差、対応BBoxの平均IoU、閾値の順で比較する。ただし検証画像は2枚だけなので、選ばれた値はローカル診断候補であり、本実験の最終閾値ではない。

各epochで選んだ閾値だけをテストデータへ適用する。テストデータを見て閾値を選び直す処理は行わない。

## 閾値調査版の出力

最初のMac実験結果を上書きしないよう、次の別フォルダへ保存する。

```text
Detection/20250208 Faster R-CNN/
├── TrainingV163_local_mac_threshold_sweep_pthfiles/
│   └── train_1.pth ～ train_5.pth
└── TrainingV163_local_mac_threshold_sweep_outputs/
    ├── run_config.json
    ├── validation_raw_predictions_epoch1.json ～ epoch5.json
    ├── validation_threshold_sweep_epoch1.csv ～ epoch5.csv
    ├── validation_threshold_sweep_epoch1.json ～ epoch5.json
    ├── validation_threshold_sweep_epoch1.jpg ～ epoch5.jpg
    ├── validation_threshold_sweep/
    │   └── epoch_1 ～ epoch_5/
    │       ├── threshold_0_1/
    │       ├── threshold_0_3/
    │       ├── threshold_0_5/
    │       ├── threshold_0_7/
    │       └── threshold_0_9/
    └── test_epoch*_threshold_*/
```

`validation_raw_predictions_epoch*.json`には、閾値適用前のBBox・スコアと正解BBoxを保存する。すべての閾値で0件だった場合でも、モデルが出した最大スコアを確認できる。

## CSVの扱い

Mac用コードはCSVファイルを書き換えない。全75/8/9行を読み込んだ後、Pythonのメモリ上で次のように先頭8/2/2件へ絞る。

```python
train_df = train_df[:TRAIN_SAMPLE_COUNT]
val_df = val_df[:VAL_SAMPLE_COUNT]
test_df = test_df[:TEST_SAMPLE_COUNT]
```

したがって、Mac用実行後もCSVの行数、boxes、labels、train/val/testの所属は変化しない。

## コード内で変更理由を探す

例えばepoch変更を探す場合は、Mac用コードで`MAC-02`を検索する。

```bash
rg -n 'MAC-02' \
  "Detection/20250208 Faster R-CNN/20260924FasterRCNN_TrainingV163_local_mac.py"
```

すべての変更理由を一覧表示する場合:

```bash
rg -n '\[(MAC|ENV|TRACE|EVAL)-' \
  "Detection/20250208 Faster R-CNN/20260924FasterRCNN_TrainingV163_local_mac.py"
```

## 差分を見る

GitHubでは`docs/diffs/Det163_original_to_local_mac.diff`を開く。ローカルでは次を実行する。

```bash
git diff --no-index -- \
  "Detection/reference/20250525FasterRCNN_TrainingV163_original.py" \
  "Detection/20250208 Faster R-CNN/20260924FasterRCNN_TrainingV163_local_mac.py"
```

この比較コマンドはファイルを変更しない。

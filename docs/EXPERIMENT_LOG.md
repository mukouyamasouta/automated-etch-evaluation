# 実験ログ

コード変更と実験結果を、再現できる情報とともに時系列で記録する。Dataset、重み、生成画像そのものはGitへ登録しない。

## 2026-09-23 — 追試用リポジトリの準備

- Det163学習コードを選定した。
- Seg204学習コードを選定した。
- DatasetのCSV参照を`20260925 Image_Dataset`へ更新した。
- GitHubリポジトリを初期化した。
- コミット: `8ba9a4b`

## 2026-09-23 — End-To-Endコードの整理

- GaN主Datasetに対応するSeg204 / Det163コードを1本だけ残した。
- 指定フォルダにあった301個の`.7z`と不要な`count_classification.py`を削除した。
- フォルダ容量を約2.4GBから約37KBへ縮小した。
- 削除した履歴は`Model_backup`から復元できる。
- コミット: `b465281`

## 2026-09-24 — Det163 Mac用動作確認の準備

- 基準のDet163を変更せず、Mac用ローカル動作確認コードを複製した。
- 条件を学習8枚、検証2枚、テスト2枚、1 epoch、batch size 1へ縮小した。
- CPUを4スレッドに制限し、Faster R-CNN内部の画像サイズをmin 400 / max 640とした。
- 実験前の基準コミット: `2d4fe6e`
- 仮想環境: `/Users/mu-sota/.venvs/gan-method-b`
- Python 3.12.13、PyTorch 2.14.0、TorchVision 0.29.0
- 外付けボリューム内の`.venv`ではパッケージメタデータ欠損が発生したため、内蔵ストレージへ切り替えた。
- 重み学習は未実行。別ターミナルからユーザーが実行する。

## 2026-09-24 — Det163元コードとMac版の追跡方法を改善

- 既存履歴を書き換えない案Aを採用した。
- `Model_backup`から完全無変更のDet163を展開してGit管理対象へ追加した。
- 元コードのSHA-256: `4d2090e2b8ab7daede3f6a00b254d9aa8b3de16da67b634ef5b43e46f91f062b`
- Mac版の各変更へ`MAC-01`〜`MAC-10`、`ENV-01`、`TRACE-01/02`を付けた。
- 変更理由表とUnified Diffを追加した。
- 作業ブランチ: `experiment/det163-local-mac`
- DatasetとCSVは変更していない。
- 重み学習は未実行。

## 2026-09-25 — Seg204 Mac用パイロットの準備

- Seg204の本設定コードを上書きせず、Mac用コードを複製した。
- `Model_backup`の`.py.7z`から完全無変更のSeg204を比較用に展開した。
- 元コードのSHA-256: `e274905f599b05899f76e164a9b47b1be122a4f8083955abc473a592042302a9`
- 条件を学習8構造（左右反転込み16サンプル）、検証2構造、テスト2構造、5 epoch、batch size 1へ縮小した。
- workers 0、pin memory False、drop last False、CPU固定・4スレッド、seed 42とした。
- 256×256、左右反転、輝度均等化、エッジ強調、U-Net構造、Tversky Loss、学習率は変更していない。
- Mac版の各変更へ`SEG-MAC-*`、`ENV-*`、`TRACE-*`番号を付け、変更理由表と保存済みdiffを追加した。
- 作業ブランチ: `experiment/seg204-local-mac`
- DatasetとCSVは変更していない。
- Python構文検査は成功した。
- `segmentation_models_pytorch`はMac用仮想環境へ未導入であり、学習前に追加が必要である。
- Seg204の重み学習は未実行。

## 2026-09-25 — Seg204初回起動失敗と環境修復

- 実行ファイル: `Segmentation/20250108 U-Net/20260925UNet_TrainingV204_local_mac.py`
- 初回実行はimport段階で`ModuleNotFoundError: No module named 'segmentation_models_pytorch'`となり終了した。
- 終了までの実測: real 10.72秒、user 2.05秒、sys 0.67秒。
- モデル作成・学習開始前の失敗であり、重みと出力フォルダは作られていない。
- 原因: Mac用仮想環境にpipと`segmentation_models_pytorch`が入っていなかった。
- `python -m ensurepip --upgrade`でpip 25.0.1を復旧した。
- 依存解決のdry-runで既存のPyTorch/TorchVisionが変更されないことを確認してから、`segmentation_models_pytorch 0.5.0`と`tqdm 4.70.1`および依存パッケージを導入した。
- import確認成功: Python 3.12.13、PyTorch 2.14.0、TorchVision 0.29.0、segmentation-models-pytorch 0.5.0、tqdm 4.70.1。
- 次の対応: 同じ5 epochコマンドを別ターミナルから再実行する。

## 実験記録テンプレート

以下を複製して使用する。

```markdown
## YYYY-MM-DD — 実験名

- 目的:
- Gitコミット:
- 実行ファイル:
- Dataset:
- 実行環境:
- device:
- train / val / test件数:
- epoch:
- batch size:
- image size / min_size / max_size:
- その他の変更条件:
- 開始時刻:
- 終了時刻:
- 所要時間:
- メモリ・発熱状況:
- 終了コード:
- 生成された重み:
- 評価結果:
- エラー・警告:
- 結論:
- 次の対応:
```

縮小条件で実行した場合は、冒頭に「ローカル動作確認であり、論文との精度比較には使用しない」と明記する。

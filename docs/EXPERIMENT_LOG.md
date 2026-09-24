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

## 2026-09-25 — Det163 Mac 5 epoch初回実験

- ローカル動作確認であり、論文との精度比較には使用しない。
- 実行ファイル: `Detection/20250208 Faster R-CNN/20260924FasterRCNN_TrainingV163_local_mac.py`
- 条件: 学習8枚、検証2枚、テスト2枚、5 epoch、batch size 1、CPU 4スレッド、min 400 / max 640。
- `train_1.pth`～`train_5.pth`と評価出力の作成に成功した。
- 固定信頼度閾値0.9では、全epochの検証・テスト画像で予測BBoxが0件だった。
- 検証IoU・テストIoUは全epochで0.0だった。
- 結論: 学習・保存経路は動作したが、閾値0.9ではEnd-To-End用の切出しを生成できない。

## 2026-09-25 — Det163検証閾値調査版の準備

- 作業ブランチ: `experiment/det163-threshold-sweep`
- 出力バージョンを`163_local_mac_threshold_sweep`とし、初回Mac実験を上書きしない。
- 学習条件は5 epoch、8/2/2枚、batch size 1、CPU 4スレッド、min 400 / max 640のまま変更していない。
- 各epochの検証生予測に0.1 / 0.3 / 0.5 / 0.7 / 0.9を適用する処理を追加した。
- 閾値ごとにBBox数、件数一致率、IoU、Precision、Recall、F1、比較グラフとBBox描画画像を保存する。
- 検証F1を第一基準として候補閾値を選び、その値だけをテストへ適用する。
- IoU 0.5以上をTPとする1対1greedy matchingを追加した。
- 合成BBoxによる評価・閾値選択・CSV/JSON保存テストは成功した。
- 重み学習の再実行はまだ行っていない。

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

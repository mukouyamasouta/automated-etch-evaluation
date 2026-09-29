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

# 2026-09-29〜30 独立3試行の実行・評価基盤

- 区分: 本実験準備（推論は未実施）
- 3試行: Seg20811 epoch 46 / Det1631 epoch 24、Seg20822 epoch 90 / Det1632 epoch 18、Seg20833 epoch 74 / Det1633 epoch 18
- 追加コード: `End-To-End/20260929_PipelineB_ThreeTrials/run_three_trials.py`
- 追加評価: `End-To-End/20260929_PipelineB_ThreeTrials/evaluate_three_trials.py`
- 評価項目: 誤検出、未検出、形状評価不能、形状評価対象、深さ、アスペクト比、w1〜w9、3試行平均、標本標準偏差
- データ管理: 6個の選択重みは新規フォルダへコピーするがGit管理外。Model_backup原本は変更しない。
- 検証結果: 2本の構文検査、合成マスクの9点測定、box IoU対応付け、標本標準偏差、3試行用生成コードの構文を確認した。
- 重み検証: 6個すべてがアーカイブ記載サイズと一致し、PyTorchのstate_dictとして読み込めた。Detectionは各295キー、Segmentationは各176キーである。
- 事前確認: `run_three_trials.py --check`はPASSした。評価側`--check`は長時間推論前のため、3試行出力が未生成であることを正しく報告した。
- 未実施: 長時間のEnd-to-End 3試行と、実出力を用いた最終評価表生成。ユーザーが別ターミナルで実施する。

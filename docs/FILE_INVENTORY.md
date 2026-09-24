# ファイル選定記録

## 採用したバックアップファイル

| 配置先 | 元ファイル | 用途 | 採用理由 |
|---|---|---|---|
| `Detection/20250208 Faster R-CNN/20250525FasterRCNN_TrainingV163.py` | 同名の`.py.7z` | 全体SEMから5個の構造を検出するFaster R-CNNの学習・検証・テスト | 手法BのEnd-To-End実験で`Det163`として参照される基準版 |
| `Segmentation/20250108 U-Net/20250526UNet_TrainingV204.py` | 同名の`.py.7z` | `Structure_padded`を用いた構造単位U-Netの学習・検証・テスト | 手法BのEnd-To-End実験で`Seg204`として参照される基準版 |
| `End-To-End/20260925 U-Net--Faster R-CNN/20250526EndToEnd_Seg204_Det163_1.py` | 同名の`.py.7z` | Det163による検出、構造切出し、Seg204による二値化、寸法計測、正解比較 | 現在のGaN主DatasetとDet163/Seg204を組み合わせた基準版 |

## コピーしなかったもの

| 種類 | 理由 |
|---|---|
| V01〜V162などの旧`.py.7z` | 試行履歴であり、基準実験の実行に不要 |
| `TrainingV*_outputs.7z` | 過去の生成結果。再実行で生成可能 |
| `TrainingV*_pthfiles.7z` | 学習済み重み一式で容量が大きい。今回は重み作成から追試するため不要 |
| `InferenceFromV*_outputs.7z` | 過去の推論結果。学習には不要 |
| `Others.7z` | 開発途中・補助ファイルの集合。基準学習コードのimport先ではない |

## `.py`だけでよいか

バックアップから移す対象は、2本の学習用`.py`と1本のEnd-To-End用`.py`で足ります。ただし実験全体としては、別途Dataset、Python依存パッケージ、計算環境、学習によって作成するDet163/Seg204の重みが必要です。

## End-To-Endフォルダの整理

`20260925 U-Net--Faster R-CNN`にあった301個の`.7z`と、別モデル出力専用の`count_classification.py`は削除対象としました。後続のSeg204/Det163コードは`20250701`以降の別Dataset向けであり、現在のGaN主Datasetをトレースする基準コードには採用していません。削除した履歴の原本は`Model_backup`に残っています。

## 元データの保持方針

`Model_backup`は変更も削除もせず、読み取り元として保持します。追試側には`.7z`から展開した実行コードだけを置きます。以後の変更履歴はファイル名の連番ではなくGitコミットで管理します。

## Seg204のMac用ファイル

| ファイル | 用途 | 変更可否 |
|---|---|---|
| `Segmentation/reference/20250526UNet_TrainingV204_original.py` | `.py.7z`から展開した完全無変更の比較基準 | 変更しない |
| `Segmentation/20250108 U-Net/20250526UNet_TrainingV204.py` | 現行Datasetのパスへ適合した本設定コード | Macパイロットでは変更しない |
| `Segmentation/20250108 U-Net/20260925UNet_TrainingV204_local_mac.py` | 5 epoch・8/2/2件・batch 1・CPU固定のMac用パイロット | Gitで変更履歴を管理する |
| `docs/SEGMENTATION_MAC_CHANGES.md` | 元コードとMac版の変更理由・影響の説明 | 条件変更時に更新する |
| `docs/diffs/Seg204_original_to_local_mac.diff` | 完全無変更版とMac版の保存済み差分 | Mac版変更時に再生成する |

Mac版はCSVを変更せず、読込み後のDataFrameだけを先頭8/2/2件へ絞ります。学習8構造は左右反転によって16サンプルになります。

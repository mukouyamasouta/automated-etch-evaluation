# 保存済みコード差分

このフォルダには、バックアップから展開した完全無変更コードとMac用コードを比較したUnified Diffを置く。

- `Det163_original_to_local_mac.diff`: Faster R-CNN Det163
- `Seg204_original_to_local_mac.diff`: U-Net Seg204

- `---`と赤い`-`行: 元コード側
- `+++`と緑の`+`行: Mac用で追加・変更した側
- `@@`: 変更箇所のおおよその行番号

このファイルは説明用であり、Pythonから読み込まれず、学習結果にも影響しない。

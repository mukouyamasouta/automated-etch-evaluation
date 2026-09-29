# パイプラインB：独立3試行の実行・評価ガイド

このガイドは、ターミナル操作に慣れていない人が、野島氏の独立学習済み重み3組でEnd-to-End推論を行い、評価表を作るための手順書である。

## 何を行うか

処理の順序は次のとおりである。

```text
独立学習したDetection・Segmentation重み3組
    ↓
同じテストデータでEnd-to-End推論を3回実行
    ↓
各試行の検出結果と教師ボックスを対応付け
    ↓
検出が妥当なマスクの形状を測定
    ↓
各試行の評価値を算出
    ↓
3試行平均と標本標準偏差を算出
```

## 使用する3組

| 試行 | Detection | Segmentation |
|---|---|---|
| 1 | V1631・epoch 24 | V20811・epoch 46 |
| 2 | V1632・epoch 18 | V20822・epoch 90 |
| 3 | V1633・epoch 18 | V20833・epoch 74 |

これらはModel_backup内の3試行End-to-Endコードと実行結果に対応している。重みは次のフォルダにコピー済みである。

```text
End-To-End/20260929_PipelineB_ThreeTrials/weights/
```

重みは大容量なのでGitHubには登録されない。

## 新しく追加した2本のプログラム

### `run_three_trials.py`

物体検出、検出領域の切出し、U-Net推論を実行する。既存コード後半の旧相対誤差計算には正解と予測の役割が逆になる問題があるため、このプログラムでは推論画像を保存した時点で止める。正しい比較は次の評価プログラムが担当する。

### `evaluate_three_trials.py`

3試行の出力を教師データと比較し、次を算出する。

- 誤検出数
- 未検出数
- セグメンテーション不良による形状評価不能マスク数
- 形状評価対象マスク数
- 深さの相対誤差平均
- アスペクト比の相対誤差平均
- `w1`～`w9`の相対誤差平均
- 上記項目の3試行平均
- 上記項目の3試行間の標本標準偏差

相対誤差は、必ず教師データを分母として計算する。

```text
相対誤差 = |予測値 - 教師値| ÷ 教師値 × 100
```

## ターミナルを開いた直後に行うこと

次をコピーし、ターミナルへ貼り付けてEnterを押す。

```bash
cd "/Volumes/met-info/Research Progress/Mukoyama/Gan系トレース手法B"
```

このコマンドは、ターミナルの作業場所を本プロジェクトへ移す。推論や評価はまだ始まらない。

以下では、プロジェクト用Pythonを明示して実行する。

```text
/Users/mu-sota/.venvs/gan-method-b/bin/python
```

## 手順1：推論前の確認

```bash
/Users/mu-sota/.venvs/gan-method-b/bin/python \
  "End-To-End/20260929_PipelineB_ThreeTrials/run_three_trials.py" --check
```

このコマンドは次だけを確認する。

- 6個の重みがあるか
- DetectionのテストCSVがあるか
- 必要なPythonパッケージがあるか
- 元にするEnd-to-Endコードがあるか

`事前確認: PASS` と表示されれば成功である。`--check`では推論を行わない。

## 手順2：End-to-Endを3回実行

### 推奨：1試行ずつ実行する

1回目は次である。

```bash
/usr/bin/time -p /Users/mu-sota/.venvs/gan-method-b/bin/python -u \
  "End-To-End/20260929_PipelineB_ThreeTrials/run_three_trials.py" --trial 1
```

終了後、2回目を実行する。

```bash
/usr/bin/time -p /Users/mu-sota/.venvs/gan-method-b/bin/python -u \
  "End-To-End/20260929_PipelineB_ThreeTrials/run_three_trials.py" --trial 2
```

終了後、3回目を実行する。

```bash
/usr/bin/time -p /Users/mu-sota/.venvs/gan-method-b/bin/python -u \
  "End-To-End/20260929_PipelineB_ThreeTrials/run_three_trials.py" --trial 3
```

各コマンドは、指定した重み1組で同じテストデータを処理する。`time -p`は最後に処理時間を表示する。`-u`は途中経過をすぐ画面に表示する。

3回を連続で実行したい場合は次でもよい。

```bash
/usr/bin/time -p /Users/mu-sota/.venvs/gan-method-b/bin/python -u \
  "End-To-End/20260929_PipelineB_ThreeTrials/run_three_trials.py" --all
```

Macでは実行中のスリープを防ぐため、プログラムが`caffeinate`を自動的に使用する。CPUでは長時間かかる可能性があるため、研究室のNVIDIA GPU搭載PCを推奨する。

### 推論の出力先

```text
End-To-End/20260929_PipelineB_ThreeTrials/
├── InferenceSeg20811_Det1631_11_outputs/
├── InferenceSeg20822_Det1632_22_outputs/
└── InferenceSeg20833_Det1633_33_outputs/
```

過去結果との混在を防ぐため、同じ出力フォルダが既にある場合、プログラムは停止する。

## 手順3：評価前の確認

3試行がすべて終わったら、次を実行する。

```bash
/Users/mu-sota/.venvs/gan-method-b/bin/python \
  "End-To-End/20260929_PipelineB_ThreeTrials/evaluate_three_trials.py" --check
```

`評価入力の事前確認: PASS` と表示されれば、3試行分の必要ファイルがそろっている。評価計算はまだ行わない。

## 手順4：評価表を作る

```bash
/Users/mu-sota/.venvs/gan-method-b/bin/python \
  "End-To-End/20260929_PipelineB_ThreeTrials/evaluate_three_trials.py"
```

検出ボックスはIoU 0.5以上を正しい対応とする。どの予測とも対応しない予測ボックスを誤検出、どの予測とも対応しない教師ボックスを未検出とする。

評価結果は次へ保存される。

```text
End-To-End/20260929_PipelineB_ThreeTrials/evaluation_outputs/
├── trial_1_summary.json
├── trial_2_summary.json
├── trial_3_summary.json
├── three_trial_mean_sample_std.json
└── evaluation_table.md
```

最初に確認するファイルは `evaluation_table.md` である。3回それぞれの値、3試行平均、標本標準偏差が1表にまとまる。各マスクの判定理由を確認するときは `trial_N_summary.json` を見る。

## 判定の意味

### 誤検出

IoU 0.5以上でどの教師ボックスとも対応しなかった予測ボックスである。

### 未検出

IoU 0.5以上でどの予測ボックスとも対応しなかった教師ボックスである。

### 形状評価不能マスク

検出は教師ボックスと対応したが、予測二値マスクが次の条件を満たさず、深さ・幅を測定できなかったものである。

- 背景と基板がそれぞれ1つの連結領域である
- 最上段が背景、最下段が基板である
- 深さ方向の11点を測定できる

11点の最上部と最下部を除いた9点を、開口側から底側へ `w1`～`w9` とする。

## 再実行したい場合

既存出力を自動削除する機能は付けていない。誤削除を防ぐためである。再実行前に、残したい結果を別の場所へコピーしてから、対象となる `Inference..._outputs` フォルダだけをFinderで確認して移動する。重みやDatasetを削除してはいけない。

## Gitに含まれるもの・含まれないもの

GitHubに反映するものは、2本のPythonコードとこの説明書などの小さな文書だけである。

GitHubに含めないものは次である。

- `.pth`重み
- Dataset
- 推論画像
- 評価結果
- 実行ログ

これらは大容量データまたは再生成可能な実験生成物だからである。

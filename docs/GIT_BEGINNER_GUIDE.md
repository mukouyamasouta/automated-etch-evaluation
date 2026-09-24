# この実験で使うGit入門

この文書は、GaN手法Bの追試で実際に使う操作だけをまとめる。Dataset、重み、生成結果はGit管理外であり、コード、設定、説明、実験ログをGitHubで管理する。

## まず状態を見る

リポジトリへ移動して実行する。

```bash
cd "/Volumes/met-info/Research Progress/Mukoyama/Gan系トレース手法B"
git status --short --branch
```

記号の意味:

- `??`: Gitがまだ管理していない新規ファイル
- `M`: 管理中ファイルの変更
- `D`: 管理中ファイルの削除
- 何も表示されない: 記録対象の変更なし

## 変更内容を見る

まだコミットしていない変更を見る。

```bash
git diff
```

新規ファイルは`git diff`だけでは表示されない。先に対象をステージしてから確認する。

```bash
git add "対象ファイル"
git diff --cached
```

`git add`はGitHubへ送信する操作ではない。「次のコミットへ含める候補を選ぶ」操作である。

## コミットする

```bash
git commit -m "変更目的を短く書く"
```

コミットは、その時点の選択した変更へ名前を付けて保存する操作である。Datasetや重みは選ばない。

## GitHubへ送る

```bash
git push
```

GitHubではリポジトリを開き、`Commits`から対象コミットを選ぶ。緑色が追加行、赤色が削除行である。複製したMac用コードは、基準コードとの比較URLやPull Requestの差分でも確認できる。

## 基準コードとMac版を比較する

```bash
git diff --no-index \
  "Detection/20250208 Faster R-CNN/20250525FasterRCNN_TrainingV163.py" \
  "Detection/20250208 Faster R-CNN/20260924FasterRCNN_TrainingV163_local_mac.py"
```

`--no-index`は、名前が異なる2ファイルを直接比較する指定である。このコマンドはファイルを変更しない。

## 元へ戻す

### コミット前の変更を戻す

```bash
git restore "対象ファイル"
```

未保存の編集内容が消えるため、必ず先に`git diff`で対象を確認する。新規ファイルには通常この方法を使わない。

### GitHubへ送ったコミットを取り消す

```bash
git log --oneline
git revert コミットID
git push
```

`git revert`は過去を消さず、変更を打ち消す新しいコミットを作る。共同利用や初心者の操作では、履歴を書き換える`reset --hard`より安全である。

## この実験で避ける操作

意味と影響範囲を確認せず、次を実行しない。

```text
git reset --hard
git clean -fd
git push --force
```

これらは未保存ファイルや共有履歴を失う可能性がある。必要になった場合は、先に対象と復旧方法を確認する。

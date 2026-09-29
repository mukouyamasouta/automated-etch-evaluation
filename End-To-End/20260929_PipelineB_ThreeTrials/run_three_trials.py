#!/usr/bin/env python3
"""野島氏の独立学習済み重み3組で手法Bを順番に実行する。

既存のMac/CUDA対応End-to-Endコードを実行時に読み込み、試行ごとの設定だけを
置換した一時スクリプトを作る。元コードとModel_backupは変更しない。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TEMPLATE = (
    REPO_ROOT
    / "End-To-End"
    / "20260925 U-Net--Faster R-CNN"
    / "20260925EndToEnd_Seg204_Det163_nojima_weights.py"
)
TEST_CSV = REPO_ROOT / "Dataset" / "20260925 CSV_Data" / "Detection" / "Original" / "test.csv"


@dataclass(frozen=True)
class Trial:
    number: int
    version: int
    segmentation_number: int
    segmentation_epoch: int
    detection_number: int
    detection_epoch: int

    @property
    def seg_weight(self) -> Path:
        return SCRIPT_DIR / "weights" / f"TrainingV{self.segmentation_number}_pthfiles" / f"val_{self.segmentation_epoch}.pth"

    @property
    def det_weight(self) -> Path:
        return SCRIPT_DIR / "weights" / f"TrainingV{self.detection_number}_pthfiles" / f"train_{self.detection_epoch}.pth"

    @property
    def output_dir(self) -> Path:
        return SCRIPT_DIR / f"InferenceSeg{self.segmentation_number}_Det{self.detection_number}_{self.version}_outputs"


TRIALS = {
    1: Trial(1, 11, 20811, 46, 1631, 24),
    2: Trial(2, 22, 20822, 90, 1632, 18),
    3: Trial(3, 33, 20833, 74, 1633, 18),
}


def replace_once(source: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"テンプレート内の設定を一意に置換できませんでした: {pattern}")
    return updated


def build_trial_source(trial: Trial) -> str:
    """既存コードから、推論終了時点までの試行別ソースを生成する。"""
    source = TEMPLATE.read_text(encoding="utf-8")
    source = replace_once(source, r"^version_number\s*=.*$", f"version_number = {trial.version}")
    source = replace_once(source, r"^IMAGE_SIZE\s*=.*$", "IMAGE_SIZE = (224, 464)")
    source = replace_once(source, r"^segmentation_number\s*=.*$", f"segmentation_number = {trial.segmentation_number}")
    source = replace_once(source, r"^SEG_EPOCH_NUMBER\s*=.*$", f"SEG_EPOCH_NUMBER = {trial.segmentation_epoch}")
    source = replace_once(source, r"^seg_model_path\s*=.*$", f"seg_model_path = Path({str(trial.seg_weight)!r})")
    source = replace_once(source, r"^detection_number\s*=.*$", f"detection_number = {trial.detection_number}")
    source = replace_once(source, r"^DET_EPOCH_NUMBER\s*=.*$", f"DET_EPOCH_NUMBER = {trial.detection_epoch}")
    source = replace_once(source, r"^det_model_path\s*=.*$", f"det_model_path = Path({str(trial.det_weight)!r})")

    # 元コード後半の旧比較処理には正解・予測の逆転があるため実行しない。
    # 推論画像を保存した時点で止め、評価はevaluate_three_trials.pyが担当する。
    marker = 'print("Segmentation Finished")'
    position = source.find(marker)
    if position < 0:
        raise RuntimeError("テンプレートの推論終了位置を確認できませんでした。")
    source = source[: position + len(marker)] + "\n"
    header = (
        f"# AUTO-GENERATED FOR TRIAL {trial.number}; DO NOT EDIT\n"
        f"# Seg{trial.segmentation_number} epoch {trial.segmentation_epoch} / "
        f"Det{trial.detection_number} epoch {trial.detection_epoch}\n"
    )
    return header + source


def preflight(selected: list[Trial]) -> list[str]:
    errors: list[str] = []
    for path in (TEMPLATE, TEST_CSV):
        if not path.is_file():
            errors.append(f"必須ファイルがありません: {path}")
    for trial in selected:
        for path in (trial.det_weight, trial.seg_weight):
            if not path.is_file():
                errors.append(f"試行{trial.number}の重みがありません: {path}")
    for module in (
        "cv2",
        "matplotlib",
        "numpy",
        "pandas",
        "PIL",
        "segmentation_models_pytorch",
        "sklearn",
        "torch",
        "torchvision",
        "tqdm",
    ):
        if importlib.util.find_spec(module) is None:
            errors.append(f"Pythonパッケージがありません: {module}")
    return errors


def run_trial(trial: Trial) -> None:
    if trial.output_dir.exists():
        raise FileExistsError(
            f"出力先が既にあります: {trial.output_dir}\n"
            "過去結果との混在を防ぐため停止しました。説明書の『再実行』を確認してください。"
        )

    generated = SCRIPT_DIR / f".generated_trial_{trial.number}.py"
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"trial_{trial.number}_{stamp}.log"
    generated.write_text(build_trial_source(trial), encoding="utf-8")

    print(f"\n=== 試行{trial.number}を開始 ===")
    print(f"Detection: V{trial.detection_number} epoch {trial.detection_epoch}")
    print(f"Segmentation: V{trial.segmentation_number} epoch {trial.segmentation_epoch}")
    print(f"出力先: {trial.output_dir}")
    print(f"ログ: {log_path}")

    try:
        with log_path.open("w", encoding="utf-8") as log:
            command = [sys.executable, "-u", str(generated)]
            caffeinate = shutil.which("caffeinate")
            if sys.platform == "darwin" and caffeinate:
                command = [caffeinate, "-i", *command]
            process = subprocess.Popen(
                command,
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)
    finally:
        generated.unlink(missing_ok=True)


def write_manifest(selected: list[Trial]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "template": str(TEMPLATE),
        "test_csv": str(TEST_CSV),
        "trials": [
            {
                "trial": t.number,
                "detection": f"V{t.detection_number}",
                "detection_epoch": t.detection_epoch,
                "segmentation": f"V{t.segmentation_number}",
                "segmentation_epoch": t.segmentation_epoch,
                "output_dir": str(t.output_dir),
            }
            for t in selected
        ],
    }
    manifest_dir = SCRIPT_DIR / "three_trial_run_outputs"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="パイプラインBを独立重み3組で実行します。")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial", type=int, choices=(1, 2, 3), help="指定した1試行だけ実行")
    group.add_argument("--all", action="store_true", help="試行1→2→3を順番に実行")
    group.add_argument("--check", action="store_true", help="実行せず、必要ファイルだけ確認")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = list(TRIALS.values()) if args.all or args.check else [TRIALS[args.trial]]
    errors = preflight(selected)
    if errors:
        print("事前確認: 失敗", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("事前確認: PASS")
    for trial in selected:
        print(
            f"試行{trial.number}: Det{trial.detection_number}/epoch {trial.detection_epoch} + "
            f"Seg{trial.segmentation_number}/epoch {trial.segmentation_epoch}"
        )
    if args.check:
        print("--check のため推論は実行していません。")
        return 0

    try:
        for trial in selected:
            run_trial(trial)
    except FileExistsError as error:
        print(f"実行を中止しました。\n{error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(
            f"試行がエラーで停止しました（終了コード {error.returncode}）。logs内の記録を確認してください。",
            file=sys.stderr,
        )
        return error.returncode or 1
    write_manifest(selected)
    print("\n推論が完了しました。次に evaluate_three_trials.py を実行してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""パイプラインBの3試行を教師データと比較し、平均・標本標準偏差を出す。"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TEST_CSV = REPO_ROOT / "Dataset" / "20260925 CSV_Data" / "Detection" / "Original" / "test.csv"
EVALUATION_DIR = SCRIPT_DIR / "evaluation_outputs"


@dataclass(frozen=True)
class Trial:
    number: int
    version: int
    segmentation_number: int
    segmentation_epoch: int
    detection_number: int
    detection_epoch: int

    @property
    def output_dir(self) -> Path:
        return SCRIPT_DIR / f"InferenceSeg{self.segmentation_number}_Det{self.detection_number}_{self.version}_outputs"


TRIALS = (
    Trial(1, 11, 20811, 46, 1631, 24),
    Trial(2, 22, 20822, 90, 1632, 18),
    Trial(3, 33, 20833, 74, 1633, 18),
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_ground_truth_boxes() -> list[list[list[float]]]:
    rows: list[list[list[float]]] = []
    with TEST_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(ast.literal_eval(row["boxes"]))
    return rows


def box_iou(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    gt_boxes: list[list[float]], pred_boxes: list[list[float]], threshold: float
) -> list[tuple[int, int, float]]:
    """IoUの高い組から重複なしで対応付ける。"""
    candidates = [
        (box_iou(gt, pred), gt_index, pred_index)
        for gt_index, gt in enumerate(gt_boxes)
        for pred_index, pred in enumerate(pred_boxes)
    ]
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if iou < threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return sorted(matches)


def remove_small_islands(mask: np.ndarray, max_area: int) -> np.ndarray:
    cleaned = mask.astype(np.uint8).copy()
    for _ in range(10):
        changed = False
        for value in (0, 1):
            component_input = (cleaned == value).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(component_input, connectivity=8)
            for label in range(1, count):
                if int(stats[label, cv2.CC_STAT_AREA]) <= max_area:
                    cleaned[labels == label] = 1 - value
                    changed = True
        if not changed:
            break
    return cleaned


def validate_predicted_mask(mask: np.ndarray, noise_max_area: int) -> tuple[bool, np.ndarray, list[str]]:
    cleaned = remove_small_islands(mask, noise_max_area)
    reasons: list[str] = []
    for value in (0, 1):
        count, _, _, _ = cv2.connectedComponentsWithStats((cleaned == value).astype(np.uint8), connectivity=8)
        components = count - 1
        if components != 1:
            reasons.append(f"{value}領域の連結成分数が1ではない: {components}")
    if not np.all(cleaned[0] == 0):
        reasons.append("最上段がすべて背景ではない")
    if not np.all(cleaned[-1] == 1):
        reasons.append("最下段がすべて基板ではない")
    return not reasons, cleaned, reasons


def trench_width(row: np.ndarray) -> int | None:
    zero_positions = np.flatnonzero(row == 0)
    if zero_positions.size == 0:
        return None
    first, last = int(zero_positions[0]), int(zero_positions[-1])
    if first == 0 or last == len(row) - 1:
        return None
    if not np.all(row[:first] == 1) or not np.all(row[first : last + 1] == 0) or not np.all(row[last + 1 :] == 1):
        return None
    return last - first + 1


def measure_mask(mask: np.ndarray) -> dict[str, Any] | None:
    """野島氏コードと同様、深さ0～100%の11点を取り、内側9点をw1～w9とする。"""
    rows: list[dict[str, int | None]] = []
    valid_indices: list[int] = []
    started = False
    for row_index, row in enumerate(mask):
        if started and np.all(row == 1):
            break
        width = trench_width(row)
        if width is not None:
            started = True
            valid_indices.append(row_index)
        rows.append({"row_index": row_index, "trench": width})

    if not valid_indices:
        return None
    start = valid_indices[0]
    measured_rows = rows[start:]
    height = len(measured_rows)
    if height < 11:
        return None

    requested = [0] + [round(height * i / 10) for i in range(1, 10)] + [height - 1]
    selected: list[int] = []
    widths: list[int] = []
    for target in requested:
        candidates = sorted(
            (abs(index - target), index)
            for index, item in enumerate(measured_rows)
            if item["trench"] is not None and index not in selected
        )
        if not candidates:
            return None
        _, index = candidates[0]
        selected.append(index)
        widths.append(int(measured_rows[index]["trench"]))

    if len(widths) != 11 or any(value <= 0 for value in widths[1:-1]):
        return None
    w_values = widths[1:-1]
    return {
        "depth": height,
        "w": w_values,
        "aspect_ratio": height / w_values[0],
    }


def read_binary_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"画像を読めません: {path}")
    return (image > 127).astype(np.uint8)


def relative_error(predicted: float, reference: float) -> float | None:
    return abs(predicted - reference) / abs(reference) if reference not in (0, None) else None


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def load_crop_map(trial: Trial) -> dict[tuple[int, int], int]:
    crop_root = trial.output_dir / "Detection_outputs" / f"cropped_epoch{trial.detection_epoch}"
    mapping: dict[tuple[int, int], int] = {}
    for directory in crop_root.glob("image*"):
        try:
            crop_index = int(directory.name.removeprefix("image"))
        except ValueError:
            continue
        metadata_path = directory / f"pred_{crop_index}.json"
        if not metadata_path.is_file():
            continue
        image_index = str(load_json(metadata_path).get("image_index", ""))
        parts = image_index.split("_")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            mapping[(int(parts[0]), int(parts[1]))] = crop_index
    return mapping


def evaluate_trial(trial: Trial, gt_by_image: list[list[list[float]]], iou_threshold: float, noise_max_area: int) -> dict[str, Any]:
    prediction_json = trial.output_dir / "Detection_outputs" / f"test_predictions_epoch{trial.detection_epoch}.json"
    predictions = load_json(prediction_json)["test_predictions"]
    crop_map = load_crop_map(trial)

    false_positives = 0
    false_negatives = 0
    segmentation_failures = 0
    reference_failures = 0
    depth_errors: list[float] = []
    aspect_errors: list[float] = []
    width_errors: list[list[float]] = [[] for _ in range(9)]
    mask_records: list[dict[str, Any]] = []

    for image_index, gt_boxes in enumerate(gt_by_image):
        pred_boxes = predictions[image_index]["boxes"] if image_index < len(predictions) else []
        matches = match_boxes(gt_boxes, pred_boxes, iou_threshold)
        false_positives += len(pred_boxes) - len(matches)
        false_negatives += len(gt_boxes) - len(matches)

        for gt_index, pred_index, iou in matches:
            record: dict[str, Any] = {
                "source_image": image_index,
                "gt_box_index": gt_index,
                "pred_box_index": pred_index,
                "detection_iou": iou,
            }
            crop_index = crop_map.get((image_index, pred_index))
            if crop_index is None:
                segmentation_failures += 1
                record.update(status="segmentation_failure", reason="対応する切出し画像がない")
                mask_records.append(record)
                continue

            seg_dir = trial.output_dir / "Segmentation_outputs" / f"test_epoch{trial.segmentation_epoch}" / f"image_{crop_index}"
            pred_path = seg_dir / f"pred_{crop_index}.tif"
            ref_path = seg_dir / f"label_{crop_index}.tif"
            try:
                pred_mask = read_binary_mask(pred_path)
                ref_mask = read_binary_mask(ref_path)
            except (ValueError, OSError) as error:
                segmentation_failures += 1
                record.update(status="segmentation_failure", reason=str(error))
                mask_records.append(record)
                continue

            valid, cleaned_pred, reasons = validate_predicted_mask(pred_mask, noise_max_area)
            pred_measurement = measure_mask(cleaned_pred) if valid else None
            ref_measurement = measure_mask(ref_mask)
            if pred_measurement is None:
                segmentation_failures += 1
                record.update(status="segmentation_failure", reason="; ".join(reasons) or "11点を測定できない")
                mask_records.append(record)
                continue
            if ref_measurement is None:
                reference_failures += 1
                record.update(status="reference_failure", reason="教師マスクから11点を測定できない")
                mask_records.append(record)
                continue

            depth_error = relative_error(pred_measurement["depth"], ref_measurement["depth"])
            aspect_error = relative_error(pred_measurement["aspect_ratio"], ref_measurement["aspect_ratio"])
            per_width = [
                relative_error(predicted, reference)
                for predicted, reference in zip(pred_measurement["w"], ref_measurement["w"])
            ]
            assert depth_error is not None and aspect_error is not None and all(value is not None for value in per_width)
            depth_errors.append(depth_error)
            aspect_errors.append(aspect_error)
            for index, value in enumerate(per_width):
                width_errors[index].append(float(value))
            record.update(
                status="evaluated",
                depth_relative_error=depth_error,
                aspect_ratio_relative_error=aspect_error,
                width_relative_errors=per_width,
            )
            mask_records.append(record)

    summary = {
        "trial": trial.number,
        "models": {
            "detection": f"V{trial.detection_number} epoch {trial.detection_epoch}",
            "segmentation": f"V{trial.segmentation_number} epoch {trial.segmentation_epoch}",
        },
        "settings": {"detection_iou_threshold": iou_threshold, "noise_max_area": noise_max_area},
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "segmentation_failure_count": segmentation_failures,
        "reference_failure_count": reference_failures,
        "evaluated_mask_count": len(depth_errors),
        "depth_relative_error_mean_percent": None if not depth_errors else mean_or_none(depth_errors) * 100,
        "aspect_ratio_relative_error_mean_percent": None if not aspect_errors else mean_or_none(aspect_errors) * 100,
        "width_relative_error_mean_percent": [None if not values else mean_or_none(values) * 100 for values in width_errors],
        "mask_records": mask_records,
    }
    return summary


def sample_stats(values: list[float | int | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "mean": statistics.fmean(clean) if clean else None,
        "sample_std": statistics.stdev(clean) if len(clean) >= 2 else None,
        "n": len(clean),
    }


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "false_positive_count",
        "false_negative_count",
        "segmentation_failure_count",
        "evaluated_mask_count",
        "depth_relative_error_mean_percent",
        "aspect_ratio_relative_error_mean_percent",
    )
    result = {key: sample_stats([summary[key] for summary in summaries]) for key in keys}
    result["width_relative_error_mean_percent"] = [
        sample_stats([summary["width_relative_error_mean_percent"][index] for summary in summaries])
        for index in range(9)
    ]
    result["source_trials"] = [summary["trial"] for summary in summaries]
    return result


def fmt(value: float | int | None, digits: int = 3) -> str:
    return "算出不能" if value is None else f"{value:.{digits}f}"


def write_markdown(summaries: list[dict[str, Any]], combined: dict[str, Any], path: Path) -> None:
    lines = [
        "# パイプラインB 3試行評価結果",
        "",
        "相対誤差は教師データを分母とし、百分率で示す。標準偏差は3試行間の標本標準偏差（ddof=1）である。",
        "",
        "| 評価項目 | 試行1 | 試行2 | 試行3 | 3試行平均 | 標本標準偏差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = [
        ("誤検出数（件）", "false_positive_count"),
        ("未検出数（件）", "false_negative_count"),
        ("形状評価不能マスク数（枚）", "segmentation_failure_count"),
        ("形状評価対象マスク数（枚）", "evaluated_mask_count"),
        ("深さ相対誤差平均（%）", "depth_relative_error_mean_percent"),
        ("アスペクト比相対誤差平均（%）", "aspect_ratio_relative_error_mean_percent"),
    ]
    for label, key in rows:
        values = [summary[key] for summary in summaries]
        lines.append(
            f"| {label} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | "
            f"{fmt(combined[key]['mean'])} | {fmt(combined[key]['sample_std'])} |"
        )
    for index in range(9):
        values = [summary["width_relative_error_mean_percent"][index] for summary in summaries]
        stats = combined["width_relative_error_mean_percent"][index]
        lines.append(
            f"| w{index + 1}相対誤差平均（%） | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | "
            f"{fmt(stats['mean'])} | {fmt(stats['sample_std'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def required_paths(trial: Trial) -> list[Path]:
    return [
        trial.output_dir / "Detection_outputs" / f"test_predictions_epoch{trial.detection_epoch}.json",
        trial.output_dir / "Detection_outputs" / f"cropped_epoch{trial.detection_epoch}",
        trial.output_dir / "Segmentation_outputs" / f"test_epoch{trial.segmentation_epoch}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3試行の検出・マスク・寸法精度を評価します。")
    parser.add_argument("--check", action="store_true", help="評価せず入力ファイルだけ確認")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="正検出とするbox IoU（既定0.5）")
    parser.add_argument("--noise-max-area", type=int, default=1000, help="除去を許す小領域の最大画素数（既定1000）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [path for trial in TRIALS for path in required_paths(trial) if not path.exists()]
    if not TEST_CSV.is_file():
        missing.insert(0, TEST_CSV)
    if missing:
        print("評価に必要なファイルがありません:")
        for path in missing:
            print(f"- {path}")
        return 1
    print("評価入力の事前確認: PASS")
    if args.check:
        print("--check のため評価計算は実行していません。")
        return 0

    EVALUATION_DIR.mkdir(exist_ok=True)
    gt_by_image = load_ground_truth_boxes()
    summaries = [evaluate_trial(trial, gt_by_image, args.iou_threshold, args.noise_max_area) for trial in TRIALS]
    for summary in summaries:
        path = EVALUATION_DIR / f"trial_{summary['trial']}_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"保存: {path}")
    combined = aggregate(summaries)
    combined_path = EVALUATION_DIR / "three_trial_mean_sample_std.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    table_path = EVALUATION_DIR / "evaluation_table.md"
    write_markdown(summaries, combined, table_path)
    print(f"保存: {combined_path}")
    print(f"保存: {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Det163をM1 MacBook Air（16GB）で安全に動作確認するための複製版。
# 完全無変更の元コード: ../reference/20250525FasterRCNN_TrainingV163_original.py
# 主な変更: 5 epoch、少数データ、batch size 1、CPU 4スレッド、モデル内部の縮小、検証閾値調査。
# この版で作る重みは接続確認用であり、論文再現や最終精度の評価には使用しない。
#
# 変更理由の一覧: ../../docs/DETECTION_MAC_CHANGES.md
# コード内の [MAC-*] は計算量調整、[ENV-*] はパス適合、[TRACE-*] は再現性記録、
# [EVAL-*] は学習条件を変えない評価・閾値診断を表す。
# %% モジュール＆パッケージ
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import pandas as pd
import ast
import csv
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# %% 設定
# [MAC-01] 元のV163出力と混同・上書きしないため、Mac確認用の名前へ分離する。
# 精度への影響: なし。保存フォルダ名だけが変わる。
version_number = "163_local_mac_threshold_sweep"
IMAGE_SIZE = None

# [ENV-01] 実行場所に依存せず、現在の20260925 Datasetを参照する。
# 精度への影響: なし。CSVや画像の内容は変更せず、保存場所だけを解決する。
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
CSV_DIR = REPOSITORY_ROOT / "Dataset" / "20260925 CSV_Data" / "Detection"

# [MAC-07] CSV自体は編集せず、読込み後のメモリ上で使用件数だけを絞る。
# 精度への影響: 大。75/8/9枚ではなく8/2/2枚なので、精度比較には使用できない。
LOCAL_SMOKE_TEST = True
TRAIN_SAMPLE_COUNT = 8
VAL_SAMPLE_COUNT = 2
TEST_SAMPLE_COUNT = 2

# [MAC-08] Faster R-CNN内部のリサイズ上限を下げ、特徴マップの計算量を減らす。
# 精度への影響: あり。小さな特徴の検出精度が変わる可能性がある。
MODEL_MIN_SIZE = 400
MODEL_MAX_SIZE = 640

# [MAC-10] CPU使用を4スレッドに制限し、Mac全体の応答低下と発熱を抑える。
# 精度への影響: なし。主に速度と端末負荷へ影響する。
CPU_THREADS = 4

# [TRACE-02] 比較可能な実験にするため乱数を固定する。
# 精度への影響: 学習結果のばらつき方は変わるが、再実行しやすくなる。
RANDOM_SEED = 42


def resolve_data_path(path_value):
    """CSVの相対パスを、このスクリプトの配置場所を基準に解決する。"""
    path = Path(path_value)
    return str(path if path.is_absolute() else (SCRIPT_DIR / path).resolve())
# ハイパーパラメータ
# [MAC-02] 30 → 5: 重み保存と検出傾向を短時間で確認する。
# 精度への影響: 大。5 epochの重みは本実験用ではない。
EPOCH_NUMBER = 5

# [MAC-03] 8 → 1: 16GB Macでのメモリ使用量を抑える。
# 精度への影響: あり。勾配更新の性質が本実験と異なる。
BATCH_SIZE = 1
LEARNING_RATE = 0.01
SHUFFLE = True

# [MAC-04] 4 → 0: macOSの別プロセス読込みによる不安定さを避ける。
# 精度への影響: なし。データ読込み速度だけに影響する。
NUM_WORKERS = 0

# [MAC-05] True → False: CUDAを使用しないため固定メモリは不要。
# 精度への影響: なし。CPU環境での転送方式だけが変わる。
PIN_MEMORY = False

# [MAC-06] True → False: 少数データを捨てずに8枚すべて使う。
# 精度への影響: 本確認版ではあり。ただし本実験との比較対象にはしない。
DROP_LAST = False
# [EVAL-01] 閾値選定には検証データだけを使い、テストデータは使わない。
# 学習への影響: なし。モデル更新後のBBoxを残す基準だけを比較する。
VALIDATION_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

# [EVAL-02] Precision/Recall/F1で正解とみなすBBox IoUの下限。
# 学習への影響: なし。評価指標の定義だけに使用する。
MATCH_IOU_THRESHOLD = 0.5
# 通常画像かセグメンテーション済み画像か
use_segmented = False
# モデル、出力結果を保存するディレクトリ
output_dirs = {
    "models": SCRIPT_DIR / f"TrainingV{version_number}_pthfiles",
    "results": SCRIPT_DIR / f"TrainingV{version_number}_outputs"
}
for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

model_dir   = output_dirs["models"]
results_dir = output_dirs["results"]

# [TRACE-01] 実験条件を出力と一緒に残し、後からGit差分と対応付ける。
# 精度への影響: なし。run_config.jsonを追加するだけである。
run_config = {
    "purpose": "Mac local validation-threshold diagnostic; not for thesis accuracy comparison",
    "version": version_number,
    "epochs": EPOCH_NUMBER,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "train_samples": TRAIN_SAMPLE_COUNT,
    "val_samples": VAL_SAMPLE_COUNT,
    "test_samples": TEST_SAMPLE_COUNT,
    "model_min_size": MODEL_MIN_SIZE,
    "model_max_size": MODEL_MAX_SIZE,
    "cpu_threads": CPU_THREADS,
    "random_seed": RANDOM_SEED,
    "validation_thresholds": VALIDATION_THRESHOLDS,
    "match_iou_threshold": MATCH_IOU_THRESHOLD,
    "threshold_selection_data": "validation only",
}
with open(results_dir / "run_config.json", "w", encoding="utf-8") as config_file:
    json.dump(run_config, config_file, ensure_ascii=False, indent=2)

# %% 学習データの読み込み
def read_df(csv_file):
    dataset = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            imgpath = resolve_data_path(row['imgpath'])
            boxes = ast.literal_eval(row['boxes'])
            labels = ast.literal_eval(row['labels'])
            dataset.append({
                'imgpath': imgpath,
                'boxes': boxes,
                'labels': labels
            })
    return dataset

dataset_kind = "Segmented" if use_segmented else "Original"
train_file = CSV_DIR / dataset_kind / "train.csv"
val_file = CSV_DIR / dataset_kind / "val.csv"
test_file = CSV_DIR / dataset_kind / "test.csv"

train_df = read_df(train_file)
val_df = read_df(val_file)
test_df = read_df(test_file)

if LOCAL_SMOKE_TEST:
    # [MAC-07] ここでリストだけを絞る。元CSVの行は削除・変更しない。
    train_df = train_df[:TRAIN_SAMPLE_COUNT]
    val_df = val_df[:VAL_SAMPLE_COUNT]
    test_df = test_df[:TEST_SAMPLE_COUNT]

print(
    "[LOCAL SMOKE TEST] "
    f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
    f"epochs={EPOCH_NUMBER}, batch={BATCH_SIZE}"
)

# %% データ前処理
class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None, use_segmented=False, image_size=(256, 256)):
        self.dataset = dataset
        self.transform = transform
        self.use_segmented = use_segmented
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        image_path = entry['imgpath']
        image = Image.open(image_path)
        image_np = np.array(image)
        original_height, original_width = image_np.shape[:2]

        # セグメンテーション済み画像を用いる場合
        if self.use_segmented:
            # チャネルが2の場合は2値化画像であり、２層目を抽出
            if image_np.ndim == 3 and image_np.shape[2] == 2:
                image_np = image_np[:, :, 1]
                image_np = (image_np * 255).astype(np.uint8)
            # チャネルが1の場合はグレースケールであり、厳密な２値化が済んでいないため閾値処理で２値化
            else:
                image_np = np.where(image_np >= 120, 255, 0).astype(np.uint8)

            image = Image.fromarray(image_np).convert("RGB")

        else:
            image = image.convert("RGB")
        # リサイズ
        if self.image_size is not None:
            image = image.resize(self.image_size, resample=Image.BILINEAR)
            resized_width, resized_height = self.image_size
        else:
            resized_width, resized_height = original_width, original_height

        # アノテーション形式の変換
        boxes = np.array(entry["boxes"], dtype=np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * resized_width / original_width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * resized_height / original_height
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(entry["labels"], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        image = self.transform(image)

        return image, target

transform = transforms.Compose([
    transforms.ToTensor(),
])

# %% データのロード
train_dataset = CustomDataset(train_df, transform=transform, use_segmented=use_segmented, image_size=IMAGE_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST, collate_fn=lambda x: tuple(zip(*x)))
val_dataset = CustomDataset(val_df, transform=transform, use_segmented=use_segmented, image_size=IMAGE_SIZE)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=lambda x: tuple(zip(*x)))
test_dataset = CustomDataset(test_df, transform=transform, use_segmented=use_segmented, image_size=IMAGE_SIZE)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=lambda x: tuple(zip(*x)))

# %% 初期化関数
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# %% モデル設定
# モデルをロード（事前学習済み）。
# [MAC-08] IMAGE_SIZEだけでなくモデル内部のmin/maxも指定し、再拡大を防ぐ。
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT",
    min_size=MODEL_MIN_SIZE,
    max_size=MODEL_MAX_SIZE,
)

# 分類クラス数（背景 + 1クラス）
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

# ヘッドを上書き（FastRCNNPredictor をインポートせずに使う）
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# 初期化適用（box_predictor のみ）
model.roi_heads.box_predictor.apply(initialize_weights)

# [MAC-09] 再現性とTorchVision Detectionの互換性を優先し、Mac版はCPU固定。
# 元コードもCUDAがないM1ではCPUになるが、意図を明示してMPSへの自動変更を防ぐ。
# 精度への影響: 原則なし。実行速度と対応演算が変わる。
device = torch.device("cpu")

# [MAC-10] 上で定義した4スレッド制限をPyTorchへ適用する。
torch.set_num_threads(CPU_THREADS)

# [TRACE-02] PyTorchとNumPyの乱数系列を固定する。
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"Device: {device}, CPU threads: {torch.get_num_threads()}")
model.to(device)

# %% 学習設定
# SGDオプティマイザ
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

# %% 予測結果のフィルタリング（分類の信頼度が低いものを検出しない）
def filter_predictions(prediction, score_threshold):
    # 検出ボックス
    boxes = prediction['boxes'].cpu().numpy()
    # 分類クラス
    labels = prediction['labels'].cpu().numpy()
    # 分類の信頼度
    scores = prediction['scores'].cpu().numpy()

    # 検出ボックスのx座標でソート（正解データと照合するため）
    sorted_indices = np.argsort(boxes[:, 0])
    boxes = boxes[sorted_indices]
    labels = labels[sorted_indices]
    scores = scores[sorted_indices]

    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            filtered_boxes.append(box.tolist())
            filtered_labels.append(int(label))
            filtered_scores.append(float(score))

    return filtered_boxes, filtered_labels, filtered_scores

# %% トレーニング
def train(model, train_loader, optimizer, device):
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    return epoch_loss

# %% 評価関数
# IoUの定義（２つの矩形領域のズレを求める）
def define_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    x_overlap_min = max(x1_min, x2_min)
    y_overlap_min = max(y1_min, y2_min)
    x_overlap_max = min(x1_max, x2_max)
    y_overlap_max = min(y1_max, y2_max)

    if x_overlap_max <= x_overlap_min or y_overlap_max <= y_overlap_min:
        return 0.0

    overlap_area = (x_overlap_max - x_overlap_min) * (y_overlap_max - y_overlap_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = overlap_area / (box1_area + box2_area - overlap_area)
    return iou

# 各予測画像に含まれる検出ボックスと正解のIoU（各検出ボックスのIoUとそれらの平均を算出）
def calculate_ious(gt_boxes, pred_boxes, define_iou):
    # 予測画像と正解の検出ボックス数が一致しない場合はスキップ
    if len(gt_boxes) != len(pred_boxes):
        return None, None

    ious = []
    for gt_box, pred_box in zip(gt_boxes, pred_boxes):
        iou = define_iou(gt_box, pred_box)
        ious.append(float(iou))
    image_iou = sum(ious) / len(ious) if ious else 0.0
    return ious, image_iou

# %% モデルの推論結果を1回だけ収集
def collect_model_predictions(model, data_loader, device):
    """閾値適用前の生予測を収集し、複数閾値で同じ予測を比較できるようにする。"""
    model.eval()
    collected = []
    image_counter = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for j, prediction in enumerate(predictions):
                # -infを指定し、TorchVisionが返した候補をスコア順に捨てず取得する。
                boxes, labels, scores = filter_predictions(prediction, float("-inf"))
                gt_boxes = targets[j]["boxes"].cpu().numpy().tolist()
                gt_boxes = sorted(gt_boxes, key=lambda box: box[0])
                collected.append({
                    "image_index": f"pred_{image_counter}",
                    "raw_boxes": boxes,
                    "raw_labels": labels,
                    "raw_scores": scores,
                    "gt_boxes": gt_boxes,
                })
                image_counter += 1

    return collected


def greedy_match_boxes(gt_boxes, pred_boxes, iou_threshold):
    """IoUが高い組から1対1対応させ、TPと対応IoUを返す。"""
    candidates = []
    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            iou = float(define_iou(gt_box, pred_box))
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))

    matched_gt = set()
    matched_pred = set()
    matched_ious = []
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matched_ious.append(iou)

    return matched_ious


def evaluate_collected_predictions(collected, score_threshold, match_iou_threshold):
    """収集済み予測へ閾値を適用し、BBox数・IoU・PR/F1を集計する。"""
    predictions = []
    per_image = []
    exact_count_image_ious = []
    all_matched_ious = []
    total_gt = 0
    total_pred = 0
    total_tp = 0

    for record in collected:
        selected = [
            (box, label, score)
            for box, label, score in zip(
                record["raw_boxes"], record["raw_labels"], record["raw_scores"]
            )
            if score >= score_threshold
        ]
        pred_boxes = [item[0] for item in selected]
        pred_labels = [item[1] for item in selected]
        pred_scores = [item[2] for item in selected]
        gt_boxes = record["gt_boxes"]

        ordered_ious, image_iou = calculate_ious(gt_boxes, pred_boxes, define_iou)
        if image_iou is not None:
            exact_count_image_ious.append(float(image_iou))

        matched_ious = greedy_match_boxes(gt_boxes, pred_boxes, match_iou_threshold)
        tp = len(matched_ious)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        total_tp += tp
        all_matched_ious.extend(matched_ious)

        predictions.append({
            "image_index": record["image_index"],
            "boxes": pred_boxes,
            "labels": pred_labels,
            "scores": pred_scores,
            "ious": ordered_ious,
            "image_iou": image_iou,
        })
        per_image.append({
            "image_index": record["image_index"],
            "gt_box_count": len(gt_boxes),
            "pred_box_count": len(pred_boxes),
            "count_difference": len(pred_boxes) - len(gt_boxes),
            "exact_count_match": len(pred_boxes) == len(gt_boxes),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "max_selected_score": max(pred_scores) if pred_scores else None,
        })

    total_fp = total_pred - total_tp
    total_fn = total_gt - total_tp
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_count_images = sum(item["exact_count_match"] for item in per_image)

    summary = {
        "score_threshold": score_threshold,
        "match_iou_threshold": match_iou_threshold,
        "image_count": len(per_image),
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "mean_pred_boxes_per_image": total_pred / len(per_image) if per_image else 0.0,
        "mean_absolute_count_error": (
            float(np.mean([abs(item["count_difference"]) for item in per_image]))
            if per_image else 0.0
        ),
        "exact_count_images": exact_count_images,
        "exact_count_rate": exact_count_images / len(per_image) if per_image else 0.0,
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(all_matched_ious)) if all_matched_ious else 0.0,
        "mean_iou_exact_count_images": (
            float(np.mean(exact_count_image_ious)) if exact_count_image_ious else 0.0
        ),
        "skipped_count_mismatch_images": len(per_image) - exact_count_images,
        "per_image": per_image,
    }
    return predictions, summary


def select_validation_threshold(summaries):
    """検証F1を優先し、同点時はBBox数一致とIoUで候補閾値を決める。"""
    return max(
        summaries,
        key=lambda item: (
            item["f1"],
            item["exact_count_rate"],
            -item["mean_absolute_count_error"],
            item["mean_matched_iou"],
            item["score_threshold"],
        ),
    )

# %% 予測結果の記録
def save_predictions_to_json(predictions, summary, results_dir, epoch, prefix):
    filename = os.path.join(results_dir, f"{prefix}_predictions_epoch{epoch}.json")
    data = {
        f"{prefix}_predictions": predictions,
        f"{prefix}_summary": summary,
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def save_raw_predictions_to_json(collected, results_dir, epoch):
    filename = os.path.join(results_dir, f"validation_raw_predictions_epoch{epoch}.json")
    with open(filename, 'w') as f:
        json.dump(collected, f, indent=4)


def save_threshold_sweep(summaries, selected_summary, results_dir, epoch):
    """検証閾値ごとの集計をJSONとCSVへ保存する。"""
    json_path = os.path.join(results_dir, f"validation_threshold_sweep_epoch{epoch}.json")
    with open(json_path, 'w') as f:
        json.dump({
            "epoch": epoch,
            "selection_data": "validation only",
            "selection_rule": (
                "highest F1, then exact-count rate, lower count error, "
                "matched IoU, and higher threshold"
            ),
            "selected_threshold": selected_summary["score_threshold"],
            "selection_warning": (
                "diagnostic only; 2 validation images are insufficient for a final threshold"
            ),
            "threshold_summaries": summaries,
        }, f, indent=4)

    csv_path = os.path.join(results_dir, f"validation_threshold_sweep_epoch{epoch}.csv")
    csv_rows = [{key: value for key, value in item.items() if key != "per_image"} for item in summaries]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # [EVAL-01] 閾値ごとのBBox数、F1、IoUを1枚で比較できるグラフを保存する。
    thresholds = [item["score_threshold"] for item in summaries]
    predicted_counts = [item["total_pred_boxes"] for item in summaries]
    expected_count = summaries[0]["total_gt_boxes"]
    f1_values = [item["f1"] for item in summaries]
    matched_ious = [item["mean_matched_iou"] for item in summaries]

    figure, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(thresholds, predicted_counts, marker='o', label='Predicted BBoxes')
    axes[0].axhline(
        expected_count, color='red', linestyle='--', label=f'Ground truth BBoxes ({expected_count})'
    )
    axes[0].set_ylabel('BBox count')
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(thresholds, f1_values, marker='o', label='F1 @ IoU 0.5')
    axes[1].plot(thresholds, matched_ious, marker='o', label='Mean matched IoU')
    axes[1].axvline(
        selected_summary["score_threshold"],
        color='green', linestyle='--',
        label=f'Selected ({selected_summary["score_threshold"]:.1f})',
    )
    axes[1].set_xlabel('Score threshold')
    axes[1].set_ylabel('Metric')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid()
    axes[1].legend()

    figure.suptitle(f'Validation threshold sweep - epoch {epoch}')
    figure.tight_layout()
    figure.savefig(
        os.path.join(results_dir, f"validation_threshold_sweep_epoch{epoch}.jpg"),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(figure)

# %% 検出ボックスを描画した予測画像の保存
def save_images(data_loader, predictions, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_idx = 0
    with torch.no_grad():
        for images, _ in data_loader:
            for img_tensor in images:
                if image_idx >= len(predictions):
                    break

                pred = predictions[image_idx]

                img = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_img)

                for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                    xmin, ymin, xmax, ymax = map(int, box)
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                    draw.text((xmin, ymin), f"class {label}: score {score:.2f}", fill="red")

                img_name = f"pred_{image_idx}.tif"
                pil_img.save(os.path.join(output_folder, img_name))
                image_idx += 1

# %% 学習の実行
# 各エポックにおける平均IoU（検証・テスト）の記録をまとめる
mean_ious = []
for epoch in range(EPOCH_NUMBER):
    # トレーニング
    epoch_loss = train(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}")
    # モデル保存
    torch.save(model.state_dict(), f"{model_dir}/train_{epoch+1}.pth")

    # [EVAL-01] 検証データを1回だけ推論し、同じ生予測で5閾値を公平に比較する。
    raw_val_predictions = collect_model_predictions(model, val_loader, device)
    save_raw_predictions_to_json(raw_val_predictions, results_dir, epoch + 1)
    validation_results = []
    validation_predictions_by_threshold = {}
    for threshold in VALIDATION_THRESHOLDS:
        val_predictions, val_summary = evaluate_collected_predictions(
            raw_val_predictions, threshold, MATCH_IOU_THRESHOLD
        )
        validation_results.append(val_summary)
        validation_predictions_by_threshold[threshold] = val_predictions

        threshold_name = str(threshold).replace('.', '_')
        threshold_image_dir = (
            results_dir / "validation_threshold_sweep" /
            f"epoch_{epoch+1}" / f"threshold_{threshold_name}"
        )
        save_images(val_loader, val_predictions, threshold_image_dir)

        print(
            f"Validation threshold={threshold:.1f}: "
            f"boxes={val_summary['total_pred_boxes']}, "
            f"exact_count_rate={val_summary['exact_count_rate']:.3f}, "
            f"matched_IoU={val_summary['mean_matched_iou']:.3f}, "
            f"P/R/F1={val_summary['precision']:.3f}/"
            f"{val_summary['recall']:.3f}/{val_summary['f1']:.3f}"
        )

    selected_val_summary = select_validation_threshold(validation_results)
    selected_threshold = selected_val_summary["score_threshold"]
    val_predictions = validation_predictions_by_threshold[selected_threshold]
    save_threshold_sweep(
        validation_results, selected_val_summary, results_dir, epoch + 1
    )
    print(f"Selected threshold from validation only: {selected_threshold:.1f}")

    # [EVAL-01] テストデータでは閾値を選び直さず、検証で選んだ値だけを適用する。
    raw_test_predictions = collect_model_predictions(model, test_loader, device)
    test_predictions, test_summary = evaluate_collected_predictions(
        raw_test_predictions, selected_threshold, MATCH_IOU_THRESHOLD
    )
    print(
        f"Test with validation-selected threshold={selected_threshold:.1f}: "
        f"boxes={test_summary['total_pred_boxes']}, "
        f"matched_IoU={test_summary['mean_matched_iou']:.3f}, "
        f"F1={test_summary['f1']:.3f}"
    )

    mean_ious.append({
        "epoch": epoch + 1,
        "selected_threshold": selected_threshold,
        "mean_val_iou": selected_val_summary["mean_iou_exact_count_images"],
        "mean_test_iou": test_summary["mean_iou_exact_count_images"],
        "val_f1": selected_val_summary["f1"],
        "test_f1": test_summary["f1"],
    })
    # 予測結果の保存
    save_predictions_to_json(
        val_predictions, selected_val_summary, results_dir, epoch + 1, "val"
    )
    save_predictions_to_json(
        test_predictions, test_summary, results_dir, epoch + 1, "test"
    )
    save_images(
        test_loader,
        test_predictions,
        results_dir / f"test_epoch{epoch+1}_threshold_{str(selected_threshold).replace('.', '_')}",
    )

# 各エポックにおける平均IoU（検証・テスト）の記録
mean_iou_filename = f"{results_dir}/mean_ious.json"
with open(mean_iou_filename, 'w') as json_file:
    json.dump(mean_ious, json_file, indent=4)

# %% IoUのグラフ化
epochs = [item['epoch'] for item in mean_ious]
val_ious = [item['mean_val_iou'] for item in mean_ious]
test_ious = [item['mean_test_iou'] for item in mean_ious]

plt.figure(figsize=(10, 6))
plt.plot(epochs, val_ious, label='Validation IoU', color='blue')
plt.plot(epochs, test_ious, label='Test IoU', color='red')
plt.xlabel('Epochs')
plt.ylabel('Mean IoU')
plt.legend()
plt.grid()
plt.ylim(0.0, 1.0)

# グラフの保存
plt.savefig(os.path.join(results_dir, 'mean_iou.jpg'))
plt.close()

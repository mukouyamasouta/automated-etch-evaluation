# SEM画像　エッジ強調の削除　初期化関数定義 バッチサイズ0.01 損失関数GIoU DROP_LASTなど設定追加 #データフォルダ名をsqauredに変更 paddingの学習 新データ(Original)試す, バッチ４ リサイズなし 損失関数戻す
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

# %% 設定
version_number = 163
IMAGE_SIZE = None
# ハイパーパラメータ
EPOCH_NUMBER = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.01
SHUFFLE = True
NUM_WORKERS = 4
PIN_MEMORY = True
DROP_LAST = True
# 分類における信頼度の閾値
score_threshold = 0.9
# 通常画像かセグメンテーション済み画像か
use_segmented = False
# モデル、出力結果を保存するディレクトリ
output_dirs = {
    "models": f"TrainingV{version_number}_pthfiles",
    "results": f"TrainingV{version_number}_outputs"
}
for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

model_dir   = output_dirs["models"]
results_dir = output_dirs["results"]

# %% 学習データの読み込み
def read_df(csv_file):
    dataset = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            imgpath = row['imgpath']
            boxes = ast.literal_eval(row['boxes'])
            labels = ast.literal_eval(row['labels'])
            dataset.append({
                'imgpath': imgpath,
                'boxes': boxes,
                'labels': labels
            })
    return dataset

train_file = "../../Dataset/20250515 CSV_Data/Detection/Segmented/train.csv" if use_segmented else "../../Dataset/20250515 CSV_Data/Detection/Original/train.csv"
val_file = "../../Dataset/20250515 CSV_Data/Detection/Segmented/val.csv" if use_segmented else "../../Dataset/20250515 CSV_Data/Detection/Original/val.csv"
test_file = "../../Dataset/20250515 CSV_Data/Detection/Segmented/test.csv" if use_segmented else "../../Dataset/20250515 CSV_Data/Detection/Original/test.csv"

train_df = read_df(train_file)
val_df = read_df(val_file)
test_df = read_df(test_file)

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
# モデルをロード（事前学習済み）
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")  # デフォルト = COCO_V1

# 分類クラス数（背景 + 1クラス）
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

# ヘッドを上書き（FastRCNNPredictor をインポートせずに使う）
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# 初期化適用（box_predictor のみ）
model.roi_heads.box_predictor.apply(initialize_weights)

# デバイス設定
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

# %% モデルの評価（検証・テスト）
def evaluate_model(model, data_loader, device, score_threshold, define_iou):
    model.eval()
    all_predictions = []
    all_image_ious = []

    image_counter = 0
    skipped_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            # 予測（検出ボックスとその分類クラス、信頼度）
            predictions = model(images)

            for j, prediction in enumerate(predictions):
                # 予測結果の信頼度によるフィルタリング
                filtered_boxes, filtered_labels, filtered_scores = filter_predictions(prediction, score_threshold)
                gt_boxes = targets[j]["boxes"].cpu().numpy()
                # 予測結果の評価（1枚の画像についてFiltered検出ボックスと正解のIoUを算出、またそれらの平均を算出）
                ious, image_iou = calculate_ious(gt_boxes, filtered_boxes, define_iou)

                if image_iou is not None:
                    all_image_ious.append(image_iou)
                # 予測画像と正解の検出ボックス数が一致しないサンプルはスキップ
                else:
                    skipped_samples += 1 

                all_predictions.append({
                    "image_index": f"pred_{image_counter}",
                    "boxes": filtered_boxes,
                    "labels": filtered_labels,
                    "scores": filtered_scores,
                    "ious": ious,
                    "image_iou": image_iou
                })

                image_counter += 1
    # 全画像の平均IoUを算出
    mean_iou = sum(all_image_ious) / len(all_image_ious) if all_image_ious else 0.0
    return all_predictions, mean_iou, skipped_samples

# %% 予測結果の記録
def save_predictions_to_json(predictions, mean_iou, skipped_count, results_dir, epoch, prefix):
    filename = os.path.join(results_dir, f"{prefix}_predictions_epoch{epoch}.json")
    data = {
        f"{prefix}_predictions": predictions,
        f"mean_{prefix}_iou (excluding unmatched predictions)": mean_iou,
        f"{prefix}_skipped_samples": skipped_count
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# %% 検出ボックスを描画した予測画像の保存
def save_images(data_loader, device, predictions, results_dir, epoch, score_threshold):
    output_folder = os.path.join(results_dir, f"test_epoch{epoch}")
    os.makedirs(output_folder, exist_ok=True)

    image_idx = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = [img.to(device) for img in images]

            for img_tensor in images:
                if image_idx >= len(predictions):
                    break

                pred = predictions[image_idx]

                img = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_img)

                for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                    if score > score_threshold:
                        xmin, ymin, xmax, ymax = map(int, box)
                        # 検出ボックスの描画
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
    # 検証
    val_predictions, mean_val_iou, val_skipped = evaluate_model(
        model, val_loader, device, score_threshold, define_iou
    )
    print(f"Validation IoU (excluding unmatched predictions): {mean_val_iou:.4f}, Skipped: {val_skipped}")
    # テスト
    test_predictions, mean_test_iou, test_skipped = evaluate_model(
        model, test_loader, device, score_threshold, define_iou
    )
    print(f"Test IoU (excluding unmatched predictions): {mean_test_iou:.4f}, Skipped: {test_skipped}")

    mean_ious.append({
        "epoch": epoch + 1,
        "mean_val_iou": mean_val_iou,
        "mean_test_iou": mean_test_iou
    })
    # 予測結果の保存
    save_predictions_to_json(val_predictions, mean_val_iou, val_skipped, results_dir, epoch + 1, "val")
    save_predictions_to_json(test_predictions, mean_test_iou, test_skipped, results_dir, epoch + 1, "test")
    save_images(test_loader, device, test_predictions, results_dir, epoch + 1, score_threshold)

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
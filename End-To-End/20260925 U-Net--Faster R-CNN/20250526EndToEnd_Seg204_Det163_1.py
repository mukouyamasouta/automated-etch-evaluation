# Detのリサイズ削除 196 163の内容に更新 形状評価 6に対する修正 ラベルのリサイズが01なのでうまくいかない（それ以外はできた）応急処置　いい画像狙い　リニア補間
# # %% モジュール＆パッケージ
import os
import shutil
import json
import ast
import csv
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision.transforms import functional
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_recall_curve
import re
from pathlib import Path

# %% 設定
version_number = 1
IMAGE_SIZE = (256,256)
NUM_WORKERS = 4
PIN_MEMORY = True
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DATASET_DIR = REPOSITORY_ROOT / "Dataset" / "20260925 CSV_Data"

def resolve_data_path(path_value):
    """CSV内の相対パスを、元コードと同じくスクリプト位置基準で解決する。"""
    path = Path(path_value)
    return str(path if path.is_absolute() else (SCRIPT_DIR / path).resolve())

# 使用するセグメンテーションモデルのバージョン
segmentation_number = 204
SEG_EPOCH_NUMBER = 87
use_seg_annotations = True
seg_model_path = REPOSITORY_ROOT / "Segmentation" / "20250108 U-Net" / f"TrainingV{segmentation_number}_pthfiles" / f"train_{SEG_EPOCH_NUMBER}.pth"

# 使用する物体検出モデルのバージョン
detection_number = 163
DET_EPOCH_NUMBER = 30
score_threshold = 0.9
use_det_annotations = True
det_model_path = REPOSITORY_ROOT / "Detection" / "20250208 Faster R-CNN" / f"TrainingV{detection_number}_pthfiles" / f"train_{DET_EPOCH_NUMBER}.pth"

# 物体検出→セグメンテーションの順で推論を行う
use_detected = True
use_segmented = False

# 出力結果を保存するディレクトリ
results_dir = SCRIPT_DIR / f"InferenceSeg{segmentation_number}_Det{detection_number}_{version_number}_outputs"
os.makedirs(results_dir, exist_ok=True)

seg_dir = os.path.join(results_dir, "Segmentation_outputs")
det_dir = os.path.join(results_dir, "Detection_outputs")
measurement_dir = os.path.join(results_dir, "Measurement_and_Assessment_outputs")
os.makedirs(seg_dir, exist_ok=True)
os.makedirs(det_dir, exist_ok=True)
os.makedirs(measurement_dir, exist_ok=True)

# 推論データの読み込み
if use_detected and not use_segmented:
    def read_df(csv_file, use_det_annotations=False, use_seg_annotations=False):
        dataset = []
        with open(csv_file, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                imgpath = resolve_data_path(row['imgpath'])
                if use_det_annotations:
                    boxes = ast.literal_eval(row['boxes'])
                    labels = ast.literal_eval(row['labels'])
                else:
                    boxes = []
                    labels = []
                if use_seg_annotations:
                    segpath = resolve_data_path(row['segpath'])
                else:
                    segpath = []

                dataset.append({
                    'imgpath': imgpath,
                    'boxes': boxes,
                    'labels': labels,
                    'segpath': segpath
                })
        return dataset
    test_file = DATASET_DIR / "Detection" / "Original" / "test.csv"
    test_df = read_df(test_file, use_det_annotations=use_det_annotations, use_seg_annotations=use_seg_annotations)

elif use_segmented and not use_detected:
    test_file = DATASET_DIR / "Segmentation" / "Original" / "test.csv"
    test_df = pd.read_csv(test_file)
    for column in ("imgpath", "labelpath"):
        if column in test_df.columns:
            test_df[column] = test_df[column].map(resolve_data_path)

else:
    raise ValueError("Invalid combination: use_detected=True and use_segmented=True is not supported.")

# %% 物体検出
# データ前処理
class DetectionDataset(Dataset):
    def __init__(self, dataset, transform=None, use_segmented=False, use_det_annotations=False, use_seg_annotations=False):
        self.dataset = dataset
        self.transform = transform
        self.use_segmented = use_segmented
        self.use_det_annotations = use_det_annotations
        self.use_seg_annotations = use_seg_annotations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        image_path = entry['imgpath']
        image = Image.open(image_path)
        image_np = np.array(image)
        
        # セグメンテーション済み画像を用いる場合
        if self.use_segmented:
            image = Image.fromarray(image_np).convert("RGB")
        else:
            image = image.convert("RGB")

        # アノテーション形式の変換
        if self.use_det_annotations:
            boxes = np.array(entry["boxes"], dtype=np.float32)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(entry["labels"], dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}
        else:
            target = {}

        # セグメンテーションラベル画像の読み込み（別で返す）
        if self.use_seg_annotations and "segpath" in entry:
            seg_image_path = entry["segpath"]
            seg_image = Image.open(seg_image_path).convert("L")
            seg_image = transforms.ToTensor()(seg_image)
        else:
            seg_image = None

        image = self.transform(image)

        return image, target, seg_image

transform = transforms.Compose([
    transforms.ToTensor(),
])

# %% データのロード
det_test_dataset = DetectionDataset(test_df, transform=transform, use_segmented=use_segmented, use_det_annotations=use_det_annotations, use_seg_annotations=use_seg_annotations)
det_test_loader = DataLoader(det_test_dataset, batch_size=1, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=lambda x: tuple(zip(*x)))

# %% モデル設定
# モデル構造
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
# 分類クラスの数（0が背景、1が物体）
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
# 学習済みモデルのロード
model.load_state_dict(torch.load(det_model_path, map_location=device))

# %% 学習設定
# Adamオプティマイザ
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

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

# %% 評価関数
# IoUの定義（２つの矩形領域のズレを求める）
def define_det_iou(box1, box2):
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
def calculate_det_ious(gt_boxes, pred_boxes, define_det_iou):
    # 予測画像と正解の検出ボックス数が一致しない場合はスキップ
    if len(gt_boxes) != len(pred_boxes):
        return None, None 
        
    ious = []
    for gt_box, pred_box in zip(gt_boxes, pred_boxes):
        iou = define_det_iou(gt_box, pred_box)
        ious.append(float(iou))
    image_iou = sum(ious) / len(ious) if ious else 0.0
    return ious, image_iou

# %% 推論（検証・テスト）+ セグメンテーション用トリミング画像生成
def evaluate_det_model_and_crop(model, data_loader, device, score_threshold, define_iou, use_det_annotations, use_seg_annotations):
    model.eval()
    all_predictions = []
    all_image_ious = []
    cropped_image_data = []  # ← 追加

    image_counter = 0
    skipped_samples = 0

    with torch.no_grad():
        for images, targets, seg_images in data_loader:
            images = [img.to(device) for img in images]
            # 予測（検出ボックスとその分類クラス、信頼度）
            predictions = model(images)

            for j, prediction in enumerate(predictions):
                original_img = images[j].cpu().numpy().transpose(1, 2, 0)
                original_img = (original_img * 255).astype(np.uint8)

                filtered_boxes, filtered_labels, filtered_scores = filter_predictions(prediction, score_threshold)

                if use_det_annotations:
                    # 予測結果の信頼度によるフィルタリング
                    gt_boxes = targets[j]["boxes"].cpu().numpy()
                    # 予測結果の評価（1枚の画像についてFiltered検出ボックスと正解のIoUを算出、またそれらの平均を算出）
                    ious, image_iou = calculate_det_ious(gt_boxes, filtered_boxes, define_iou)
                    if image_iou is not None:
                        all_image_ious.append(image_iou)
                    # 予測画像と正解の検出ボックス数が一致しないサンプルはスキップ
                    else:
                        skipped_samples += 1
                else:
                    ious = []
                    image_iou = None

                # セグメンテーション画像があれば追加
                if use_seg_annotations:
                    seg_np = seg_images[j].squeeze().cpu().numpy()
                for i, box in enumerate(filtered_boxes):
                    xmin, ymin, xmax, ymax = map(int, box)
                    crop_img = original_img[ymin:ymax, xmin:xmax]

                    if use_seg_annotations:
                        crop_seg = seg_np[ymin:ymax, xmin:xmax]
                    else:
                        crop_seg = None

                    cropped_image_data.append({
                        "img": crop_img,
                        "seg": crop_seg,
                        "meta": {
                            "image_index": f"{image_counter}_{i}",
                            "label": 0
                        }
                    })

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
    return all_predictions, mean_iou, skipped_samples, cropped_image_data

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
        for images, _, seg_images in data_loader:  # ← seg_images を追加
            images = [img.to(device) for img in images]

            for img_tensor, seg_tensor in zip(images, seg_images):  # ← seg_tensor を扱う
                if image_idx >= len(predictions):
                    break

                pred = predictions[image_idx]

                # 入力画像の保存
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

                # セグメンテーションラベル画像の保存（use_seg_annotations=True の場合のみ）
                if seg_tensor is not None:
                    seg_np = seg_tensor.squeeze().cpu().numpy()
                    seg_img = Image.fromarray((seg_np * 255).astype(np.uint8)).convert("RGB")  # RGBに変換して描画可能に
                    seg_draw = ImageDraw.Draw(seg_img)

                    for box, score in zip(pred["boxes"], pred["scores"]):
                        if score > score_threshold:
                            xmin, ymin, xmax, ymax = map(int, box)
                            # 検出ボックスの描画
                            seg_draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

                    seg_img.save(os.path.join(output_folder, f"seg_{image_idx}.tif"))

                image_idx += 1

# %% 推論の実行
mean_ious = []
epoch = DET_EPOCH_NUMBER - 1
test_predictions, mean_test_iou, test_skipped, cropped_data = evaluate_det_model_and_crop(
    model, det_test_loader, device, score_threshold, define_det_iou, use_det_annotations, use_seg_annotations
)
if use_det_annotations:
    print(f"Detection Test IoU (excluding unmatched predictions): {mean_test_iou:.4f}, Skipped: {test_skipped}")

    mean_ious.append({
        "epoch": epoch + 1,
        "mean_test_iou": mean_test_iou
    })
    # 平均IoU（検証・テスト）の記録
    mean_iou_filename = f"{det_dir}/mean_ious.json"
    with open(mean_iou_filename, 'w') as json_file:
        json.dump(mean_ious, json_file, indent=4)
else:
    print("Detection Test IoU: Skipped (No annotations)")

# 予測結果の保存
save_predictions_to_json(test_predictions, mean_test_iou, test_skipped, det_dir, epoch + 1, "test")
save_images(det_test_loader, device, test_predictions, det_dir, epoch + 1, score_threshold)
print("Detection Finished")

def save_cropped_data(cropped_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for idx, entry in enumerate(cropped_data):
        img = entry["img"]
        seg = entry["seg"]
        meta = entry["meta"]

        # 新しいインデックスを使って名前を統一
        image_name = f"image{idx}"
        pred_name = f"pred_{idx}"

        # ディレクトリ構造: output_dir/image{idx}/pred_{idx}.*
        sample_dir = os.path.join(output_dir, image_name)
        os.makedirs(sample_dir, exist_ok=True)

        # 画像の保存
        img_path = os.path.join(sample_dir, f"{pred_name}.tif")
        Image.fromarray(img).save(img_path)

        # セグメンテーション画像の保存（存在する場合）
        if seg is not None:
            seg_path = os.path.join(sample_dir, f"{pred_name}_seg.tif")
            seg_img = Image.fromarray((seg * 255).astype(np.uint8))
            seg_img.save(seg_path)

        # メタ情報の保存
        meta_path = os.path.join(sample_dir, f"{pred_name}.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)

cropped_output_dir = os.path.join(det_dir, f"cropped_epoch{epoch+1}")
save_cropped_data(cropped_data, cropped_output_dir)
print("Cropped image data saved.")

# %% セグメンテーション
# データ前処理
class SegmentationDataset(Dataset):
    #一貫コード用に修正
    def __init__(
        self, cropped_data, transform=None, 
        classes=None,
        image_size=(256, 256),
        use_seg_annotations=False,
    ):
        self.image_size = image_size
        self.use_seg_annotations  = use_seg_annotations
        self.cropped_data = cropped_data

    def __len__(self):
        return len(self.cropped_data)

    def __getitem__(self, i):
        # SEM画像の前処理
        data = self.cropped_data[i]
        img = data["img"]

        img = cv2.resize(img, dsize=self.image_size)

        # 輝度均等化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

        # エッジ強調（Laplacianを加算）
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = np.clip(laplacian, -255, 255).astype(np.float32)
        img = np.clip(img.astype(np.float32) + 0.5 * laplacian, 0, 255).astype(np.uint8)

        img = torch.from_numpy(img.astype(np.float32)).clone()
        img = img.permute(2, 0, 1)

        # 正解データの前処理
        if self.use_seg_annotations:
            label = data["seg"]
            label = np.asarray(label)

            # 1. もとの画像がすでに0/1または0/255なら最近傍補間で構造維持
            label = cv2.resize(label, dsize=self.image_size, interpolation=cv2.INTER_NEAREST)

            # 2. 必要に応じて再2値化（安全性確保）
            _, label = cv2.threshold(label, 0.5, 1, cv2.THRESH_BINARY)  # 入力が0~1なら
            # または
            #_, label = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)  # 入力が0~255なら

            label = label.astype(np.uint8)
            label = torch.from_numpy(label).long()
            label = torch.nn.functional.one_hot(label, num_classes=2).permute(2, 0, 1).to(torch.float32)

            return {
                "img": img,
                "label": label
            }

        return {
            "img": img
        }

# %% データのロード
# セグメンテーションでアノテーションを使用するか
seg_test_dataset  = SegmentationDataset(cropped_data, image_size=IMAGE_SIZE, use_seg_annotations=use_seg_annotations)
seg_test_loader   = DataLoader(seg_test_dataset,  batch_size = 1, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY)

# %% U-Netの構造
# 2回の畳み込み＋BatchNorm＋ReLUブロック
class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  middle_channels, kernel_size=3, padding="same")
        self.bn1   = nn.BatchNorm2d(middle_channels)
        self.rl    = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding="same")
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x

# アップサンプリング＋Convのブロック
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1  = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same")
        self.bn2  = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

# U-Net本体
class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()

        # エンコーダ（ダウンサンプリング）部
        self.TCB1 = TwoConvBlock(3,    64,   64  )
        self.TCB2 = TwoConvBlock(64,   128,  128 )
        self.TCB3 = TwoConvBlock(128,  256,  256 )
        self.TCB4 = TwoConvBlock(256,  512,  512 )
        self.TCB5 = TwoConvBlock(512,  1024, 1024)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        # デコーダ（アップサンプリング）部
        self.UC1 = UpConv(1024, 512)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.UC2 = UpConv(512, 256)
        self.TCB7 = TwoConvBlock(512, 256, 256)
        self.UC3 = UpConv(256, 128)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.UC4 = UpConv(128, 64)
        self.TCB9 = TwoConvBlock(128, 64, 64)

        # 出力層（クラス数 = 2）
        self.conv1 = nn.Conv2d(64, 2, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # エンコーダ（特徴抽出）
        x  = self.TCB1(x)
        x1 = x
        x  = self.maxpool(x)
        x  = self.TCB2(x)
        x2 = x
        x  = self.maxpool(x)
        x  = self.TCB3(x)
        x3 = x
        x  = self.maxpool(x)
        x  = self.TCB4(x)
        x4 = x
        x  = self.maxpool(x)

        # ボトルネック
        x = self.TCB5(x)

        # デコーダ（アップサンプリング＋スキップ接続）
        x = self.UC1(x)
        x = torch.cat([x4, x], dim=1)
        x = self.TCB6(x)
        x = self.UC2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.TCB7(x)
        x = self.UC3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.TCB8(x)
        x = self.UC4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.TCB9(x)

        # 出力層
        x = self.conv1(x)

        return x

# %% 損失関数
# TverskyLoss(alpha=1.0, beta=1.0) = IoULoss
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False, alpha=1.0, beta=1.0)
def criterion(pred, target):
    return TverskyLoss(pred, target)

# １エポックの各損失の記録
def save_loss_history(loss_list, filename):
    with open(filename, "w") as f:
        for loss in loss_list:
            f.write(f"{loss}\n")

# １エポックの平均損失の記録
def save_epoch_mean_loss(loss_list, filename, epoch):
    avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
    with open(filename, "a") as f:
        f.write(f"Epoch {epoch + 1}: {avg_loss:.5f}\n")

# %% 推論（テスト）
def test(model_path, test_loader, criterion, epoch, history, results_dir, use_seg_annotations):
    model = UNet_2D()
    # 学習済みモデルのロード
    model.load_state_dict(torch.load(model_path))
    model.eval()

    sigmoid = nn.Sigmoid()
    all_preds, all_inputs, all_labels = [], [], []
    total_loss = 0
    history["test_loss_history"] = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", unit="iter"):
            inputs = data["img"]
            logits = model(inputs)

            probs = sigmoid(logits)
            outputs = torch.argmax(probs, axis=1)
            pred = torch.nn.functional.one_hot(outputs, num_classes=2).to(torch.float32)

            all_preds.append(pred)
            all_inputs.append(inputs)

            if use_seg_annotations:
                labels = data["label"]
                loss = criterion(logits, labels)
                history["test_loss_history"].append(loss.item())
                total_loss += loss.item()
                all_labels.append(labels)

    if use_seg_annotations:
        save_loss_history(history["test_loss_history"], f"{results_dir}/test_loss_epoch{epoch+1}.txt")
        save_epoch_mean_loss(history["test_loss_history"], f"{results_dir}/mean_test_loss.txt", epoch)
        all_labels_tensor = torch.cat(all_labels, dim=0)
    else:
        all_labels_tensor = None

    return torch.cat(all_preds, dim=0), torch.cat(all_inputs, dim=0), all_labels_tensor

# %% 評価関数
# IoUの定義
def define_seg_iou(mask1, mask2):
    intersection = torch.sum(mask1 * mask2)
    union = torch.sum(mask1) + torch.sum(mask2) - intersection
    return (intersection / union).item() if union != 0 else float('nan')

# 各予測画像と正解画像のIoU（IoU_1は背景のIoU、IoU_2は形状のIoU）
def calculate_seg_ious(all_labels, all_preds, i):
    preds_iou = all_preds[i].permute(2, 0, 1)
    iou_1 = define_seg_iou(all_labels[i][0], preds_iou[0])
    iou_2 = define_seg_iou(all_labels[i][1], preds_iou[1])
    return iou_1, iou_2

# IoUの保存
def write_iou_to_txt(all_labels, all_preds, epoch, results_dir):
    iou_epoch_file = f"{results_dir}/iou_epoch{epoch+1}.txt"
    mean_iou_file = f"{results_dir}/mean_iou.txt"

    with open(iou_epoch_file, "w") as f:
        ious_1, ious_2 = [], []
        for i in range(len(all_labels)):
            iou_1, iou_2 = calculate_seg_ious(all_labels, all_preds, i)
            ious_1.append(iou_1)
            ious_2.append(iou_2)
            f.write(f"Data {i}: IoU 1: {iou_1:.5f}, IoU 2: {iou_2:.5f}\n")
        
        mean_1 = sum(ious_1) / len(ious_1) if ious_1 else float("nan")
        mean_2 = sum(ious_2) / len(ious_2) if ious_2 else float("nan")
        std_1 = (sum((x - mean_1) ** 2 for x in ious_1) / len(ious_1)) ** 0.5 if ious_1 else float("nan")
        std_2 = (sum((x - mean_2) ** 2 for x in ious_2) / len(ious_2)) ** 0.5 if ious_2 else float("nan")

        f.write(f"Mean IoU 1: {mean_1:.5f} ± {std_1:.5f}\n")
        f.write(f"Mean IoU 2: {mean_2:.5f} ± {std_2:.5f}\n")

    # エポックごとの IoU 2（形状のみ）を保存
    with open(mean_iou_file, "a") as f:
        f.write(f"{mean_2:.5f} ± {std_2:.5f}\n")

    print(f"Segmentation Test IoU: {mean_2:.5f} ± {std_2:.5f}")

# %% 画像の保存
def save_test_images(all_preds, all_inputs, all_labels, epoch, results_dir):
    base_dir = f"{results_dir}/test_epoch{epoch+1}"
    os.makedirs(base_dir, exist_ok=True)
    all_preds_permute = all_preds.permute(0, 3, 1, 2)

    for i in tqdm(range(min(101, len(all_preds))), desc="Saving Test Images"):
        image_dir = os.path.join(base_dir, f"image_{i}")
        os.makedirs(image_dir, exist_ok=True)

        pred = (all_preds_permute[i][1].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(image_dir, f"pred_{i}.tif"), pred)

        if all_labels is not None:
            label = (all_labels[i][1].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(image_dir, f"label_{i}.tif"), label)

        input_data = all_inputs[i].permute(1, 2, 0).cpu().numpy()[:, :, 0].astype(np.uint8)
        Image.fromarray(input_data, mode='L').save(os.path.join(image_dir, f"input_{i}.tif"))

# %% 推論の実行
history = {"test_loss_history": []}
epoch = SEG_EPOCH_NUMBER - 1

all_preds, all_inputs, all_labels = test(
    # セグメンテーションモデルのパス
    model_path=seg_model_path,
    test_loader=seg_test_loader,
    criterion=criterion,
    epoch=epoch,
    history=history,
    # セグメンテーション用のディレクトリ
    results_dir=seg_dir,
    # セグメンテーションでアノテーションを使用するか
    use_seg_annotations=use_seg_annotations
)
if use_seg_annotations:
    # IoUの記録
    write_iou_to_txt(all_labels, all_preds, epoch, seg_dir)
else:
    print("Segmentation Test IoU: Skipped (No annotations)")
# 画像の保存
save_test_images(all_preds, all_inputs, all_labels, epoch, seg_dir)
print("Segmentation Finished")

# %% 測定
# 検出領域の構造（背景=0、基板=1）抽出
def extract_structure(all_preds, epoch, measurement_dir, original_size_list):
    for i, pred_tensor in enumerate(all_preds):
        pred_np = pred_tensor.cpu().numpy() if hasattr(pred_tensor, "cpu") else np.array(pred_tensor)
        pred_np = np.argmax(pred_np, axis=-1)

        # 元のサイズにリサイズを戻す
        original_height, original_width = original_size_list[i]
        pred_np = cv2.resize(pred_np.astype(np.uint8), dsize=(original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # 出力ディレクトリ
        image_dir = os.path.join(measurement_dir, f"image{i}")
        os.makedirs(image_dir, exist_ok=True)

        # 画像保存
        img_out_path = os.path.join(image_dir, f"pred_{i}.tif")
        Image.fromarray((pred_np * 255).astype(np.uint8)).save(img_out_path)

        # 画像の各行の01パターンを調査
        temp_rows = []
        valid_indices = []

        for row_idx, row in enumerate(pred_np):
            row_str = ''.join(['1' if px == 1 else '0' for px in row])

            match_101 = re.fullmatch(r'(1+)(0+)(1+)', row_str)
            # 101パターン（構造領域）
            if match_101:
                left = len(match_101.group(1))
                trench = len(match_101.group(2))
                right = len(match_101.group(3))
                valid_indices.append(row_idx)
            else:
                left = trench = right = None

            temp_rows.append({
                "row_index": row_idx,
                "left_sidewall": left,
                "trench": trench,
                "right_sidewall": right
            })

        # 最初の101パターンから最後の101パターンまで抽出
        if valid_indices:
            start = valid_indices[0]
            end = valid_indices[-1]
            rows = temp_rows[start:end + 1]
            # 構造の高さ
            structure_height = end - start + 1
        else:
            rows = []
            structure_height = 0

        # 構造情報の記録
        json_out_path = os.path.join(image_dir, f"pred_{i}.json")
        with open(json_out_path, 'w') as f:
            json.dump({
                "structure_height": structure_height,
                "rows": rows
            }, f, indent=2)

# 各画像の元サイズのリストを用意
original_size_list = [(data["img"].shape[0], data["img"].shape[1]) for data in cropped_data]
# リサイズした画像を元サイズに戻し測定
extract_structure(all_preds, epoch, measurement_dir, original_size_list)
print("Measurement Finished")

# 暫定
def analyze_saved_segmentation_images(seg_root_dir):
    image_dirs = sorted([d for d in os.listdir(seg_root_dir) if os.path.isdir(os.path.join(seg_root_dir, d))])

    for image_dir in image_dirs:
        full_dir = os.path.join(seg_root_dir, image_dir)

        # pred_*.tif ファイル名を特定（例: pred_0_seg.tif）
        pred_seg_files = [f for f in os.listdir(full_dir) if f.endswith("_seg.tif")]
        if not pred_seg_files:
            continue

        seg_file = pred_seg_files[0]
        pred_prefix = seg_file.replace("_seg.tif", "")  # 例: pred_0
        seg_path = os.path.join(full_dir, seg_file)

        # 画像読み込み・2値化
        seg_image = Image.open(seg_path).convert("L")
        seg_np = np.array(seg_image)
        seg_bin = (seg_np > 127).astype(np.uint8)

        temp_rows = []
        valid_indices = []

        for row_idx, row in enumerate(seg_bin):
            row_str = ''.join(['1' if px == 1 else '0' for px in row])

            match_101 = re.fullmatch(r'(1+)(0+)(1+)', row_str)
            if match_101:
                left = len(match_101.group(1))
                trench = len(match_101.group(2))
                right = len(match_101.group(3))
                valid_indices.append(row_idx)
            else:
                left = trench = right = None

            temp_rows.append({
                "row_index": row_idx,
                "left_sidewall": left,
                "trench": trench,
                "right_sidewall": right
            })

        if valid_indices:
            start = valid_indices[0]
            end = valid_indices[-1]
            rows = temp_rows[start:end + 1]
            structure_height = end - start + 1
        else:
            rows = []
            structure_height = 0

        # JSON保存（pred_x.json）
        json_path = os.path.join(full_dir, f"{pred_prefix}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "structure_height": structure_height,
                "rows": rows
            }, f, indent=2)

    print("Structure analysis and JSON export completed.")

cropped_seg_dir = os.path.join(det_dir, f"cropped_epoch{DET_EPOCH_NUMBER}")
analyze_saved_segmentation_images(cropped_seg_dir)

def compare_structure_jsons(pred_dir, ref_dir, output_path):
    result = {}
    image_dirs = sorted(os.listdir(pred_dir))

    for image_dir in image_dirs:
        pred_json_path = os.path.join(pred_dir, image_dir, f"pred_{image_dir.replace('image', '')}.json")
        ref_json_path = os.path.join(ref_dir, image_dir, f"pred_{image_dir.replace('image', '')}.json")

        if not os.path.exists(pred_json_path) or not os.path.exists(ref_json_path):
            continue

        with open(pred_json_path, 'r') as f:
            pred_data = json.load(f)

        with open(ref_json_path, 'r') as f:
            ref_data = json.load(f)

        pred_rows = {r["row_index"]: r for r in pred_data["rows"] if None not in r.values()}
        ref_rows = {r["row_index"]: r for r in ref_data["rows"] if None not in r.values()}

        common_indices = set(pred_rows.keys()) & set(ref_rows.keys())
        row_errors = []

        for idx in sorted(common_indices):
            pred_row = pred_rows[idx]
            ref_row = ref_rows[idx]

            row_error = {}
            for key in ["left_sidewall", "trench", "right_sidewall"]:
                ref_val = ref_row[key]
                pred_val = pred_row[key]
                if ref_val != 0:
                    rel_err = abs(pred_val - ref_val) / ref_val
                    row_error[key] = rel_err
            row_error["row_index"] = idx
            row_errors.append(row_error)

        # structure_height の相対誤差
        pred_height = pred_data["structure_height"]
        ref_height = ref_data["structure_height"]
        if ref_height != 0:
            structure_height_rel_error = abs(pred_height - ref_height) / ref_height
        else:
            structure_height_rel_error = None

        result[image_dir] = {
            "structure_height_relative_error": structure_height_rel_error,
            "row_relative_errors": row_errors
        }

    # JSON出力
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Comparison results saved to: {output_path}")

def analyze_comparison_result(comparison_json_path, summary_output_path):
    with open(comparison_json_path, 'r') as f:
        data = json.load(f)

    structure_errors = []
    trench_errors = []

    for image_key, values in data.items():
        h_err = values.get("structure_height_relative_error")
        if h_err is not None:
            structure_errors.append(h_err)

        for row in values.get("row_relative_errors", []):
            if "trench" in row:
                trench_errors.append(row["trench"])

    def compute_stats(error_list):
        if not error_list:
            return {
                "mean": None,
                "max": None,
                "min": None,
                "std": None,
                "count": 0
            }
        arr = np.array(error_list)
        return {
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "std": float(np.std(arr)),
            "count": len(error_list)
        }

    summary = {
        "structure_height_relative_error_stats": compute_stats(structure_errors),
        "trench_relative_error_stats": compute_stats(trench_errors)
    }

    with open(summary_output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_output_path}")

pred_dir = os.path.join(det_dir, f"cropped_epoch{DET_EPOCH_NUMBER}")
ref_dir = measurement_dir
output_json_path = os.path.join(det_dir, f"comparison_epoch{DET_EPOCH_NUMBER}.json")

compare_structure_jsons(pred_dir, ref_dir, output_json_path)
comparison_json_path = os.path.join(det_dir, f"comparison_epoch{DET_EPOCH_NUMBER}.json")
summary_output_path = os.path.join(det_dir, f"comparison_summary_epoch{DET_EPOCH_NUMBER}.json")

analyze_comparison_result(comparison_json_path, summary_output_path)

# # %% 形状評価
# # CD(Critical Dimension)の評価
# def evaluate_trench_trend(rows):
#     # 評価対象が14行以下なら不可
#     if len(rows) <= 10:
#         return "Insufficient Data (Requires ≥15 rows)"

#     center_rows = rows[5:-5]
#     trench_values = [row["trench"] for row in center_rows]

#     # 全て同じなら垂直
#     if all(trench_values[0] == val for val in trench_values):
#         return "Vertical"

#     # 単調性判定
#     is_increasing = all(x <= y for x, y in zip(trench_values, trench_values[1:]))
#     is_decreasing = all(x >= y for x, y in zip(trench_values, trench_values[1:]))

#     # Bowing（前半減少、後半増加）
#     mid = len(trench_values) // 2
#     first_half = trench_values[:mid]
#     second_half = trench_values[mid:]
#     is_bowing = (
#         all(x >= y for x, y in zip(first_half, first_half[1:])) and
#         all(x <= y for x, y in zip(second_half, second_half[1:]))
#     )

#     if is_bowing:
#         return "Bowing"
#     elif is_increasing:
#         return "Positive Taper"
#     elif is_decreasing:
#         return "Negative Taper"
#     else:
#         return "Other"

# # 判定結果を記録するための辞書
# category_summary = {
#     "Vertical": [],
#     "Bowing": [],
#     "Positive Taper": [],
#     "Negative Taper": [],
#     "Other": [],
#     "Insufficient Data (Requires ≥15 rows)": []
# }

# # 評価と保存
# for filename, data in structure_data.items():
#     try:
#         rows = data["rows"]
#         result = evaluate_trench_trend(rows)
#     except Exception as e:
#         result = f"Error: {e}"

#     # 結果をカテゴリ別に記録
#     category_summary.setdefault(result, []).append(filename)

#     result_json = {
#         "filename": filename,
#         "trend_evaluation": result
#     }

#     # 保存（個別ファイル）
#     result_name = data["json_name"].replace(".json", "_assessed.json")
#     result_path = os.path.join(data["json_dir"], result_name)
#     with open(result_path, 'w', encoding='utf-8') as f:
#         json.dump(result_json, f, indent=4, ensure_ascii=False)

# # summary.json を出力
# summary_path = os.path.join(measurement_dir, "assessment_summary.json")
# with open(summary_path, 'w', encoding='utf-8') as f:
#     json.dump(category_summary, f, indent=4, ensure_ascii=False)

# # 内容を表示
# print("=== Summary of Evaluation Results ===")
# print(json.dumps(category_summary, indent=4, ensure_ascii=False))
# print("Assessment Finished")

# %%

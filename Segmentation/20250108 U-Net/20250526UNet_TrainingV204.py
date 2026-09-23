# Detected 増強なし 輝度均等化削除 画像サイズ(128,128)　paddedデータ データ更新 画像サイズ(256,256) 誤差範囲示す 増強を反転に変更 輝度均等化　エッジ強調0.5 バッチサイズ16 198と同条件
# %% モジュール＆パッケージ
import pandas as pd
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import functional
import segmentation_models_pytorch as smp
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
import os
import shutil
from pathlib import Path

# %% 設定
version_number = 204
IMAGE_SIZE = (256,256)
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
CSV_DIR = REPOSITORY_ROOT / "Dataset" / "20260925 CSV_Data" / "Segmentation"

def resolve_data_path(path_value):
    """CSVの相対パスを、このスクリプトの配置場所を基準に解決する。"""
    path = Path(path_value)
    return str(path if path.is_absolute() else (SCRIPT_DIR / path).resolve())
AUGMENTATION = True
NUM_WORKERS = 4
PIN_MEMORY = True
DROP_LAST = True
SHUFFLE = True
#ハイパーパラメータ
EPOCH_NUMBER = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
#ステップサイズ = トレーニングデータ数 // バッチサイズ * 減衰エポック間隔
STEP_SIZE = 4000 // BATCH_SIZE * 100
#学習率（減衰率）
GAMMA = 0.1

# 通常画像か物体検出済み画像か
use_detected = True
# モデル、出力結果を保存するディレクトリ
output_dirs = {
    "models": SCRIPT_DIR / f"TrainingV{version_number}_pthfiles",
    "results": SCRIPT_DIR / f"TrainingV{version_number}_outputs"
}
for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

model_dir   = output_dirs["models"]
results_dir = output_dirs["results"]

#学習データの読み込み
dataset_kind = "Detected_padded" if use_detected else "Original"
train_file = CSV_DIR / dataset_kind / "train.csv"
val_file = CSV_DIR / dataset_kind / "val.csv"
test_file = CSV_DIR / dataset_kind / "test.csv"

train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

for dataframe in (train_df, val_df, test_df):
    for column in ("imgpath", "labelpath"):
        dataframe[column] = dataframe[column].map(resolve_data_path)

# %% データ前処理
class Dataset(BaseDataset):
    def __init__(
        self, df, transform=None, 
        classes=None, augmentation=False,
        image_size=(256, 256)
    ):
        self.augmentation = augmentation
        self.image_size = image_size

        imgpaths = list(df.imgpath)
        labelpaths = list(df.labelpath)
        # データ増強
        if self.augmentation:
            self.imgpath_list = []
            self.labelpath_list = []
            self.flip_list = []

            for imgpath, labelpath in zip(imgpaths, labelpaths):
                for flip in [False, True]:  # 元画像と左右反転画像の2通り
                    self.imgpath_list.append(imgpath)
                    self.labelpath_list.append(labelpath)
                    self.flip_list.append(flip)
        else:
            self.imgpath_list = imgpaths
            self.labelpath_list = labelpaths
            self.flip_list = [False] * len(imgpaths)

    def __getitem__(self, i):
        # SEM画像の前処理
        imgpath = self.imgpath_list[i]

        if not isinstance(imgpath, str):
            print(f"[ERROR] imgpath is not str: {imgpath} (type: {type(imgpath)})")

        img = cv2.imread(imgpath)
        img = cv2.resize(img, dsize=self.image_size)

        # 輝度均等化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

        # エッジ強調（Laplacianを加算）
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = np.clip(laplacian, -255, 255).astype(np.float32)
        img = np.clip(img.astype(np.float32) + 0.5 * laplacian, 0, 255).astype(np.uint8)

        flip = self.flip_list[i]
        if flip:
            img = cv2.flip(img, 1)

        img = torch.from_numpy(img.astype(np.float32)).clone()
        img = img.permute(2, 0, 1)

        # 正解データの前処理
        labelpath = self.labelpath_list[i]
        label = Image.open(labelpath)
        label = np.asarray(label)
        label = cv2.resize(label, dsize=self.image_size)

        if flip:
            label = cv2.flip(label, 1)

        label = torch.from_numpy(label.astype(np.float32)).clone()

        # 2値化
        threshold = 100
        label = (label > threshold).long()
        label = torch.nn.functional.one_hot(label, num_classes=2)

        label = label.to(torch.float32)
        label = label.permute(2, 0, 1)

        data = {"img": img, "label": label, "imgpath": imgpath, "labelpath": labelpath}
        return data

    def __len__(self):
        return len(self.imgpath_list)


# %% データのロード
train_dataset = Dataset(train_df, image_size=IMAGE_SIZE, augmentation=AUGMENTATION)
train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST, shuffle = SHUFFLE)
val_dataset   = Dataset(val_df, image_size=IMAGE_SIZE)
val_loader    = DataLoader(val_dataset,   batch_size = 1, pin_memory=PIN_MEMORY, num_workers = NUM_WORKERS)
test_dataset  = Dataset(test_df, image_size=IMAGE_SIZE)
test_loader   = DataLoader(test_dataset,  batch_size = 1, pin_memory=PIN_MEMORY,  num_workers = NUM_WORKERS)

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

# %% 初期化関数の定義
def initialize_weights(m):
    # Conv2d レイヤーの場合
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    # BatchNorm2d レイヤーの場合
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)
    # Linier レイヤーの場合
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

# %% モデル設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet_2D().to(device)

# %% 学習設定
# 初期化関数の適用
unet.apply(initialize_weights)
# Adamオプティマイザ
optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
# StepLRスケジューラ
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# %% 損失関数 TverskyLoss(alpha=1.0, beta=1.0) = IoULoss
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False, alpha=1.0, beta=1.0)
def criterion(pred, target):
    return TverskyLoss(pred, target)

# # BCE Loss（logits出力に対して安定的に使える）
# BCE_Loss = nn.BCEWithLogitsLoss()
# def criterion(pred, target):
#     return BCE_Loss(pred, target)

# 損失関数の記録
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

# %% 学習プロセス
# トレーニング
def train(unet, train_loader, optimizer, scheduler, criterion, device, epoch, history, results_dir):
    train_loss = 0
    n = 0
    history["train_loss_history"] = []
    unet.train()
    
    for i, data in tqdm(enumerate(train_loader), desc=f"Training Epoch {epoch+1}", unit="iter"):
        inputs, labels = data["img"].to(device), data["label"].to(device)
        optimizer.zero_grad()
        outputs = unet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        history["train_loss_history"].append(loss.item())
        n += 1

        if i % ((len(train_loader.dataset) // train_loader.batch_size) // 1) == (len(train_loader.dataset) // train_loader.batch_size) // 1 - 1:
            print(f"epoch:{epoch + 1}  index:{i + 1}  train_loss:{train_loss / n:.5f}")
            n = 0
            train_loss = 0

    save_loss_history(history["train_loss_history"], f"{results_dir}/train_loss_epoch{epoch+1}.txt")
    save_epoch_mean_loss(history["train_loss_history"], f"{results_dir}/mean_train_loss.txt", epoch)

# 検証
def validate(unet, val_loader, criterion, device, epoch, history, results_dir):
    val_loss = 0
    m = 0
    history["val_loss_history"] = []
    unet.eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), desc=f"Validation Epoch {epoch+1}", unit="iter"):
            inputs, labels = data["img"].to(device), data["label"].to(device)
            outputs = unet(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            history["val_loss_history"].append(loss.item())
            m += 1

            if i % (len(val_loader.dataset) // val_loader.batch_size) == (len(val_loader.dataset) // val_loader.batch_size) - 1:
                print(f"epoch:{epoch+1} index:{i+1} val_loss:{val_loss/m:.5f}")
                m = 0
                val_loss = 0

    save_loss_history(history["val_loss_history"], f"{results_dir}/val_loss_epoch{epoch+1}.txt")
    save_epoch_mean_loss(history["val_loss_history"], f"{results_dir}/mean_val_loss.txt", epoch)

# テスト
def test(unet, test_loader, criterion, epoch, history, results_dir, device):
    unet.eval()

    sigmoid = nn.Sigmoid()
    all_preds, all_inputs, all_labels = [], [], []
    total_loss = 0
    history["test_loss_history"] = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", unit="iter"):
            inputs = data["img"].to(device)
            labels = data["label"].to(device)
            outputs = unet(inputs)
            loss = criterion(outputs, labels)
            history["test_loss_history"].append(loss.item())
            total_loss += loss.item()

            outputs = sigmoid(outputs)
            outputs = torch.argmax(outputs, axis=1)
            pred = torch.nn.functional.one_hot(outputs, num_classes=2).to(torch.float32)

            all_preds.append(pred)
            all_inputs.append(inputs)
            all_labels.append(labels)

    save_loss_history(history["test_loss_history"], f"{results_dir}/test_loss_epoch{epoch+1}.txt")
    save_epoch_mean_loss(history["test_loss_history"], f"{results_dir}/mean_test_loss.txt", epoch)

    return torch.cat(all_preds, dim=0), torch.cat(all_inputs, dim=0), torch.cat(all_labels, dim=0)
    
# %% 評価関数
# IoUの定義
def define_iou(mask1, mask2):
    intersection = torch.sum(mask1 * mask2)
    union = torch.sum(mask1) + torch.sum(mask2) - intersection
    return (intersection / union).item() if union != 0 else float('nan')

# 各予測画像と正解画像のIoU（IoU_1は背景のIoU、IoU_2は形状のIoU）
def calculate_ious(all_labels, all_preds, i):
    preds_iou = all_preds[i].permute(2, 0, 1)
    iou_1 = define_iou(all_labels[i][0], preds_iou[0])
    iou_2 = define_iou(all_labels[i][1], preds_iou[1])
    return iou_1, iou_2

# IoUの保存
def write_iou_to_txt(all_labels, all_preds, epoch, results_dir):
    iou_epoch_file = f"{results_dir}/iou_epoch{epoch+1}.txt"
    mean_iou_file = f"{results_dir}/mean_iou.txt"

    with open(iou_epoch_file, "w") as f:
        ious_1, ious_2 = [], []
        for i in range(len(all_labels)):
            iou_1, iou_2 = calculate_ious(all_labels, all_preds, i)
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

    print(f"Test IoU: {mean_2:.5f} ± {std_2:.5f}")

# %% 画像の保存
def save_test_images(all_preds, all_inputs, all_labels, epoch, results_dir):
    base_dir = f"{results_dir}/test_epoch{epoch+1}"
    os.makedirs(base_dir, exist_ok=True)
    all_preds_permute = all_preds.permute(0, 3, 1, 2)

    for i in tqdm(range(min(101, len(all_preds))), desc="Saving Test Images"):
        image_dir = os.path.join(base_dir, f"image_{i}")
        os.makedirs(image_dir, exist_ok=True)

        pred = (all_preds_permute[i][1].cpu().numpy() * 255).astype(np.uint8)
        label = (all_labels[i][1].cpu().numpy() * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(image_dir, f"pred_{i}.tif"), pred)
        cv2.imwrite(os.path.join(image_dir, f"label_{i}.tif"), label)

        input_data = all_inputs[i].permute(1, 2, 0).cpu().numpy()[:, :, 0].astype(np.uint8)
        Image.fromarray(input_data, mode='L').save(os.path.join(image_dir, f"input_{i}.tif"))

# %% 学習の実行
history = {"train_loss_history": [], "val_loss_history": [], "test_loss_history": []}
for epoch in range(EPOCH_NUMBER):
    train(unet, train_loader, optimizer, scheduler, criterion, device, epoch, history, results_dir)
    validate(unet, val_loader, criterion, device, epoch, history, results_dir)
    # モデルの保存
    torch.save(unet.state_dict(), f"{model_dir}/train_{epoch+1}.pth")

    all_preds, all_inputs, all_labels = test(
        unet=unet,
        test_loader=test_loader,
        criterion=criterion,
        epoch=epoch,
        history=history,
        results_dir=results_dir,
        device=device
    )
    # IoUの記録
    write_iou_to_txt(all_labels, all_preds, epoch, results_dir)
    # 画像の保存
    save_test_images(all_preds, all_inputs, all_labels, epoch, results_dir)

# %% 損失のグラフ化
def extract_loss(filepath):
    loss_list = []
    with open(filepath, 'r') as f:
        for line in f:
            if ":" in line:
                value = float(line.strip().split(":")[-1])
                loss_list.append(value)
    return loss_list

train_loss_load = extract_loss(f'{results_dir}/mean_train_loss.txt')
val_loss_load = extract_loss(f'{results_dir}/mean_val_loss.txt')
test_loss_load = extract_loss(f'{results_dir}/mean_test_loss.txt')

epochs = np.arange(1, len(train_loss_load) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_load, label='Train Loss', color='blue')
plt.plot(epochs, val_loss_load, label='Validation Loss', color='orange')
plt.plot(epochs, test_loss_load, label='Test Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.ylim(0.0, 0.8)

# グラフの保存
plt.savefig(f'{results_dir}/mean_loss.jpg')
plt.close()

# %% IoUのグラフ化
IoU_load = np.loadtxt(f'{results_dir}/mean_iou.txt', usecols=0)

x_IoU = np.arange(1, len(IoU_load)+1)

plt.figure(figsize=(10, 6))
plt.plot(x_IoU, IoU_load, label='IoU', color='blue')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.grid()
plt.ylim(0.25, 1.0)

# グラフの保存
plt.savefig(f'{results_dir}/mean_iou.jpg')
plt.close()

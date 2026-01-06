# Low-Quality Pill Images Classification

Intro2AI project - Advanced image classification using Masked Autoencoder with ResNet+FPN+CBAM architecture

## 📋 Mô tả dự án

Dự án phân loại ảnh viên thuốc chất lượng thấp sử dụng kiến trúc deep learning tiên tiến kết hợp:
- **ResNet Backbone 6 layers**: Trích xuất đặc trưng đa tầng
- **Feature Pyramid Network (FPN)**: Tổng hợp thông tin đa tỉ lệ
- **CBAM (Convolutional Block Attention Module)**: Cơ chế attention kênh và không gian
- **Masked Autoencoder**: Học biểu diễn robust thông qua reconstruction

## 🏗️ Kiến trúc Model

### Tổng quan luồng dữ liệu

```
Input (224×224×3)
    ↓
ResNetBackbone6Layers
    ├─→ [c1, c2, c3, c4] → FPN → SharedFC → MainHead → Main Classification
    └─→ c6 (2048ch) → CBAM → AuxiliaryHead → Aux Classification
                        ↓
                   AttentionMap
                        ↓
                 GridMaskSelector
                        ↓
              Masked Input → Backbone+FPN → SharedFC → ReconstructionHead
```

### Chi tiết các thành phần

#### 1. ResNetBackbone6Layers (kế thừa CNN/ResNet18)
- **Input**: 224×224×3
- **Stem**: Conv7×7 s2 → BN/ReLU → AdaptivePool → 96×96×64
- **6 Layers**:
  - Layer 1: 96×96×64 (cho FPN)
  - Layer 2: 48×48×128 (cho FPN)
  - Layer 3: 24×24×256 (cho FPN)
  - Layer 4: 12×12×512 (cho FPN)
  - Layer 5: 6×6×1024 (trung gian)
  - Layer 6: 3×3×2048 (cho CBAM)
- **Output**: `[c1, c2, c3, c4], c6`

#### 2. Feature Pyramid Network (FPN)
- **Input**: [c1:64ch, c2:128ch, c3:256ch, c4:512ch]
- **Lateral Conv**: Chuẩn hóa về 256 kênh
- **Top-down pathway**: Tổng hợp thông tin từ thô → tinh
- **Output**: [p2:96×96, p3:48×48, p4:24×24, p5:12×12] × 256ch

#### 3. CBAM Attention
- **Input**: c6 (3×3×2048)
- **Channel Attention**: Avg/Max pooling → FC → Sigmoid
- **Spatial Attention**: Channel-wise avg/max → Conv → Sigmoid
- **Output**: Enhanced feature + Attention map

#### 4. GridMaskSelector
- **Input**: Attention map + Original image
- **Logic**: 
  - Chia attention map thành lưới 3×3
  - Tìm vùng 2×2 có tổng attention cao nhất
  - Tạo mask che 4/9 ảnh
- **Output**: Masked image

#### 5. SharedFC (Encoder)
- **Input**: FPN feature lớn nhất (96×96×256)
- **CNN Encoder**: 
  - 96×96 → 48×48 → 24×24 → 12×12 → 6×6 (512ch)
- **FC Block**: 
  - Flatten → Linear(18432→1024) → Linear(1024→512)
- **Output**: Latent vector 512-dim

#### 6. Classification Heads
- **MainHead**: Latent 512 → FC → 15 classes (từ clean image)
- **AuxiliaryHead**: CBAM feature 2048 → AvgPool → FC → 15 classes

#### 7. ReconstructionHead
- **Input**: Latent 512-dim (từ masked image)
- **FC Expand**: 512 → 25088 (512×7×7)
- **Decoder**: 5 ConvTranspose layers
  - 7×7 → 14×14 → 28×28 → 56×56 → 112×112 → 224×224
- **Output**: Reconstructed image 224×224×3

## 📊 Loss Function

```python
Total Loss = Main Loss + λ_aux × Aux Loss + λ_rec × Rec Loss
```

- **Main Loss**: CrossEntropyLoss (classification chính)
- **Aux Loss**: CrossEntropyLoss (auxiliary supervision, λ=0.4)
- **Rec Loss**: Masked MSE Loss (reconstruction, λ=0.5)

## 🔧 Hyperparameters

```python
BATCH_SIZE = 8
NUM_CLASSES = 15
EPOCHS = 10-30 (khuyến nghị 20-30 cho ~47M params)
LEARNING_RATE = 1e-4
OPTIMIZER = Adam
LAMBDA_AUX = 0.4
LAMBDA_REC = 0.5
```

## 🛠️ Requirements

```
torch >= 1.10.0
torchvision >= 0.11.0
pandas
PIL
tqdm
```

## 📝 Notes

- **Checkpoint tự động**: Lưu mỗi N epochs vào `model_checkpoints/`
- **Loss weights**: Lambda có thể điều chỉnh tùy dataset
- **Grid size**: Hiện tại 3×3, có thể tùy chỉnh trong `GridMaskSelector`
- **Image normalization**: ImageNet statistics [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]

## 📧 Contact

Intro2AI Project - Nguyễn Công Hùng - HUST IT-E10 03 K69
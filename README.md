# Label_semantic_consistent_sentence_4_scene

本專案目標是：**針對同一場景（scene）的多張圖片，從多個候選句子中找出「與該場景所有圖片平均語意最相近」的句子**。做法是先用 **OpenAI GPT API** 為每張圖片生成一句描述，再用 **CLIP** 計算「句子 vs 場景內所有圖片」的平均相似度（平均 CLIP score），最後選出每個場景平均分數最高的句子，作為該場景的 **semantic consistent sentence**。

---

## Use case

你有一個場景資料集：

- **場景數**：20 個場景
- **每個場景**：包含多張圖片

本專案會：

- **對每張圖片產生 1 句描述**（使用 GPT）
- **對每個場景**：在該場景的多個句子中，找出讓「該句子與該場景所有圖片」的**平均 CLIP score 最高**的那一句

---

## Installation

### 環境需求

- **Python**：建議 Python 3.10+

### 安裝套件：

```bash
conda create -n label python=3.10
conda activate label
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install -r requirements.txt
```

---

## 方法概述（Pipeline）

### 1) 為每張圖片生成一句描述（GPT）

- 輸入：場景 \(s\) 中的每張圖片 \(I_{s,i}\)
- 輸出：每張圖片一個句子 \(T_{s,i}\)

> 也就是：同一場景若有 \(N\) 張圖，會得到 \(N\) 個候選句子。

### 2) 用 CLIP 計算相似度並做「場景內平均」

對於場景 \(s\) 的每個候選句子 \(T_{s,k}\)，計算它與該場景所有圖片的平均相似度：

$$
\operatorname{avg\_score}(s,k)=\frac{1}{N_s}\sum_{j=1}^{N_s}\operatorname{CLIP}(I_{s,j},T_{s,k})
$$

### 3) 選出每個場景的最佳句子

$$
T^{*}_s=\arg\max_{k}\operatorname{avg\_score}(s,k)
$$

輸出：每個場景一個最佳句子 \(T^{*}_s\)，並可保留完整分數明細以利檢查。

---

## 預期輸入資料結構（建議）

本專案支援你目前的 Unreal 資料夾結構：每個 `scene_*` 底下會有一個內容資料夾（例如 `InsideOut/`），並分成 `training_view/` 與 `testing_view/`，每個 view 都包含 `rgb/depth/normal/camera_info`。

```text
dataset/
  scene_01/
    <scene_content_name>/               # 例如 InsideOut
      training_view/
        rgb/                            # 會用於 GPT 產句 + CLIP 計分
          el_000_az_000.png
          el_000_az_019.png
          ...
        depth/
          el_000_az_000.png
          ...
        normal/
          el_000_az_000.png
          ...
        camera_info/
          el_000_az_000.json
          ...
      testing_view/
        rgb/
        depth/
        normal/
        camera_info/
  scene_02/
    ...
```

**實際使用時**，我們會把同一個 scene 的 `training_view/rgb/*.png` + `testing_view/rgb/*.png` 全部視為「同一場景的多張圖片」來做候選句子的平均 CLIP score 計分。


---

## 輸出（建議）

### 場景最佳句子

我建議（也符合你要的工作流）是：**每個 scene 輸出一個 JSON 檔**，並放在該 scene 的內容資料夾下：

```text
dataset/
  scene_01/
    <scene_content_name>/
      scene_01.json
  scene_02/
    <scene_content_name>/
      scene_02.json
```

其中 `scene_01.json` 內容範例：

```json
{
  "best_text": "...",
  "best_avg_clip_score": 0.0
}
```

---

## 實作細節

- **GPT 產生的句子風格**：
```bash
DEFAULT_PROMPT = (
    "Write ONE concise English sentence describing the scene in the image.\n"
    "Requirements:\n"
    "- Describe only visible objects, layout, and environment; avoid speculation.\n"
    "- Keep it short (about 12–25 words).\n"
    "- Do not mention camera parameters, azimuth/elevation, or filenames.\n"
    "- Do not use bullet points.\n"
)
```
- **CLIP 模型**：OpenCLIP
- **相似度**：cosine
- **計分策略**：mean（場景內平均）

---

## Usage
### 1) 清理 skybox / 無效視角

蒐集資料時，偶爾會出現某些視角「整張只拍到 skybox」，造成對應的 `depth` / `normal` 幾乎是常數或無效。建議在開始標注前，先用本專案提供的腳本把這些視角整組刪除（`rgb/depth/normal/camera_info` 同名檔案會一起移除）。

#### Dry-run（不刪除，只產生報告）

```bash
python filter_invalid_views.py --dataset-root dataset --report filter_report.json
```

#### 實際刪除(確認 `filter_report.json` 後再刪)

```bash
python filter_invalid_views.py --dataset-root dataset --report filter_report.json --delete
```

#### 可調參數（避免誤刪）

- `--mode-frac-threshold`：影像中「最常出現的像素值」所占比例，越接近 1 越像常數圖
- `--std-mean-threshold`：像素標準差（多通道取平均），越小越像常數圖
- `--require-both`：若你希望更保守，設定後會要求 depth **與** normal 都符合無效條件才刪

---

#### 多場景批次處理：

```bash
mkdir -p filter_reports
for scene_dir in /media/iverson/KINGSTON/dataset/scene_*; do
  echo "==> DELETE: $scene_dir"
  python filter_invalid_views.py \
    --dataset-root "$scene_dir" \
    --report "filter_reports/$(basename "$scene_dir").json" \
    --delete
done
```

### 2) 用 GPT 產生每張圖一句描述，並用 CLIP 挑選該 scene 最佳句

以 `scene_01` 為例：

```bash
# 設定 OpenAI 金鑰：
export OPENAI_API_KEY="your key"
# Label scene：
python label_scene.py \
  --scene-root /media/iverson/KINGSTON/dataset/scene_01 \
  --openai-model gpt-5.5-pro-2026-04-23 \
  --clip-model ViT-L-14 \
  --clip-pretrained datacomp_xl_s13b_b90k
```

（可選）若你要輸出除錯明細，可加上 `--write-details` 產生 `scene_01.clip_scores.json`。

#### 批次標注 20 個場景（scene_*）

（預設會跳過 `scene_12`）

```bash
export OPENAI_API_KEY="your key"
python label_dataset.py \
  --dataset-root /media/iverson/KINGSTON/dataset \
  --openai-model gpt-5.5-pro-2026-04-23 \
  --clip-model ViT-L-14 \
  --clip-pretrained datacomp_xl_s13b_b90k \
  --device cuda
```

若你已經產生過 captions，想只重跑 CLIP（不重打 GPT）：

```bash
python label_dataset.py \
  --dataset-root /media/iverson/KINGSTON/dataset \
  --skip-gpt \
  --device cuda
```

若你要指定只跑部分場景：

```bash
python label_dataset.py \
  --dataset-root /media/iverson/KINGSTON/dataset \
  --scenes scene_01,scene_02,scene_03
```

若你想跳過不同的場景（或不跳過任何場景）：

```bash
# 跳過 scene_12 與 scene_15
python label_dataset.py --dataset-root /media/iverson/KINGSTON/dataset --skip-scenes scene_12,scene_15

# 不跳過任何 scene
python label_dataset.py --dataset-root /media/iverson/KINGSTON/dataset --skip-scenes ""
```

---

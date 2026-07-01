# 多目標車種辨識與跨幀追蹤

這個專案使用 Ultralytics YOLO 進行全畫面偵測，並用簡單的 IoU + 質心距離匹配維持跨幀固定 ID。

## 支援類別

- 轎車：COCO `car`
- 機車：COCO `motorcycle`
- 貨車：COCO `truck`

每個物件會在影片上標示：

```text
ID 1 | sedan/car | 0.86
```

## 安裝

```powershell
py -m pip install -r requirements.txt
```

第一次使用 `yolov8n.pt` 時，Ultralytics 會自動下載模型權重。也可以用 `--model` 指定你已經訓練好的車種模型。

## 執行

```powershell
py vehicle_yolo_tracker.py --source input.mp4 --output outputs/tracked_vehicles.mp4
```

使用攝影機：

```powershell
py vehicle_yolo_tracker.py --source 0 --show
```

## 常用參數

```powershell
py vehicle_yolo_tracker.py `
  --source input.mp4 `
  --output outputs/tracked_vehicles.mp4 `
  --model yolov8n.pt `
  --conf 0.35 `
  --track-iou 0.25 `
  --track-distance 90 `
  --max-missed 18
```

- `--conf`：YOLO 偵測信心門檻。
- `--track-iou`：同一台車跨幀匹配的 IoU 門檻。
- `--track-distance`：當 IoU 偏低時，允許用質心距離維持同一 ID。
- `--max-missed`：車輛暫時被遮擋或漏偵時，保留 ID 的最大幀數。

## 設計說明

1. YOLO 對整張畫面做偵測。
2. 只保留 `car`、`motorcycle`、`truck` 三種車輛類別。
3. 每一幀先用 IoU 尋找最合理的舊軌跡。
4. 如果車框變化大但質心仍接近，使用質心距離補救。
5. 成功匹配的車輛沿用原本 ID；新車輛取得新 ID。
6. 消失超過 `--max-missed` 幀的軌跡會被移除。

py vehicle_yolo_tracker.py --source 123.mp4 --output result.mp4
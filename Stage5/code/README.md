# 局部光流遮罩與雙向軌跡繪製（密集光流）

這個工具用 Stage3、Stage4 抓出的物件框作為局部遮罩，在每個框內計算密集光流，並把平均光流畫成箭頭代表車輛運動方向。同時會記錄每台車的中心點歷史位置，將中心點連線成軌跡。

## 稀疏光流 vs 密集光流

稀疏光流只追蹤少量特徵點，例如角點、車牌邊緣或車體明顯紋理，常見方法是 Lucas-Kanade。優點是速度快、適合追蹤穩定特徵；缺點是沒有特徵的車身表面、陰影或模糊區域可能追不到。

密集光流會估計影像中大量像素，甚至每個像素的位移向量，常見方法是 Farneback、RAFT。優點是能得到整個物件框內的整體運動趨勢，適合本需求的「用物件框遮罩後計算局部運動方向」；缺點是計算量較高，也比較需要用遮罩、門檻與方向規則排除雜訊。

## 影片運動向量計畫

1. 讀取連續影格，將前一幀與目前幀轉灰階。
2. 使用 Farneback 密集光流計算每個像素的 `(dx, dy)`。
3. 將 Stage3、Stage4 的車輛框裁切為局部遮罩。
4. 只在框內取光流，並用光流 magnitude 過濾弱雜訊。
5. 對框內有效像素取平均，得到該車的局部運動向量。
6. 在車輛中心點畫箭頭，箭頭方向即平均光流方向。
7. 保存每個車輛 ID 的歷史中心點，逐幀畫出軌跡線。

## 方向性邏輯

影像座標中，`x > 0` 代表往右，`x < 0` 代表往左；`y > 0` 代表往下，`y < 0` 代表往上。

預設規則：

- 左側車道：若平均光流 `dy > 0`，判定為 `left:oncoming`，代表迎面而來、由遠變近。
- 右側車道：若平均光流 `dy < 0`，判定為 `right:departing`，代表背對駛離、由近變遠。

不同鏡頭角度可能讓主要運動軸落在 `x` 或正負號相反，因此腳本提供參數校準：

```powershell
python dense_optical_flow_tracks.py `
  --video input.mp4 `
  --stage3 stage3_boxes.json `
  --stage4 stage4_boxes.json `
  --output outputs/flow_tracks.mp4 `
  --left-approach-axis y `
  --left-approach-sign positive `
  --right-depart-axis y `
  --right-depart-sign negative
```

## 偵測框格式

支援 JSON 與 CSV。JSON 可使用逐幀格式：

```json
[
  {
    "frame_index": 1,
    "boxes": [
      {"track_id": "car-1", "bbox": [100, 180, 220, 320], "score": 0.91}
    ]
  }
]
```

也支援單筆 detection 格式：

```json
[
  {"frame_index": 1, "track_id": "car-1", "x1": 100, "y1": 180, "x2": 220, "y2": 320}
]
```

CSV 欄位可用：

```text
frame_index,track_id,x1,y1,x2,y2,score
1,car-1,100,180,220,320,0.91
```

如果 Stage3/Stage4 沒有 `track_id`，腳本會用中心點最近鄰做簡易追蹤，並以 `--match-distance` 控制最大匹配距離。

## 輸出顏色

- 紅色：左側車道迎面而來 `left:oncoming`
- 藍色：右側車道背對駛離 `right:departing`
- 黃色：方向與預期相反或未分類
- 灰色：光流太弱，視為靜止或不可靠

## 防呆重點

- 物件框會自動裁切到影像邊界內。
- 太小或無效的框會略過。
- 光流太弱會標為 `static/weak`，避免把雜訊誤判成車流方向。
- Stage3 與 Stage4 可分開輸入，也可用 `--use-stages stage4` 只處理某一階段。



py sparse_optical_flow.py                     
   
py dense_optical_flow.py
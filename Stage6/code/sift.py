
Claude finished the response
使用orb對影片進行特徵點抓取

下午1:53
以下是使用 ORB（Oriented FAST and Rotated BRIEF）對影片逐幀抓取特徵點的 Python 程式，支援即時預覽與輸出影片。


Ran 3 commands, viewed a file, created a file
Ran 3 commands, viewed a file, created a file
測試通過，124 fps 處理速度（640×480 合成影片）。

基本用法：

bash
# 輸出結果影片
python orb_video.py --input input.mp4 --output output.mp4

# 即時預覽（不存檔）
python orb_video.py --input input.mp4 --preview

# 兩者同時
python orb_video.py --input input.mp4 --output output.mp4 --preview
調整 ORB 參數：

bash
# 抓更多特徵點（預設 500）
python orb_video.py -i input.mp4 -o out.mp4 --nfeatures 1000

# 改用 FAST 評分（速度快但精度略低）
python orb_video.py -i input.mp4 -o out.mp4 --fast-score

# 只畫圓點，不顯示方向/尺度圓圈
python orb_video.py -i input.mp4 -o out.mp4 --simple-draw
主要 flag 說明：

Flag	說明	預設
--nfeatures	每幀最大特徵點數	500
--scale-factor	金字塔縮放比例	1.2
--nlevels	金字塔層數	8
--edge-threshold	邊緣忽略區域大小	31
--fast-score	改用 FAST 評分	否（Harris）
--simple-draw	只畫圓點不畫方向	否（Rich）
HUD 上會即時顯示當前幀的特徵點數量、幀編號、處理 FPS，方便判斷參數是否合適。


Orb video
PY 
影片為 123.mp4

下午1:55

Ran 2 commands
Ran 2 commands
上傳的目錄裡找不到 123.mp4，看起來檔案沒有一起上傳進來。

請在這個對話中附上 123.mp4（點擊迴紋針/附件按鈕），上傳後我就能直接執行 ORB 特徵點抓取並輸出結果影片給你。

我說路徑

下午1:55
了解，請提供 123.mp4 的完整路徑（例如 /home/user/123.mp4 或 C:\Users\...\123.mp4），我來修改指令直接執行。

和程式同路徑

下午1:55
那就是 /home/claude/123.mp4，直接執行：


Ran 2 commands
Ran 2 commands
找不到這個檔案。這個環境是 Claude 的沙盒容器，無法存取你本機的檔案系統，所以「和程式同路徑」在你電腦上才有意義，在這裡沒有。

你有兩個選擇：

在你自己的機器上執行，把 orb_video.py 下載後放到和 123.mp4 同一個資料夾，再跑：
bash
   python orb_video.py --input 123.mp4 --output 123_orb.mp4
上傳影片到這個對話，我幫你在沙盒裡直接處理並輸出結果。
ge6
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---         2026/6/26 上午 01:07       77491157 123.mp4
-a---         2026/6/26 下午 01:54           6355 123.py
-a---         2026/6/24 上午 02:10           3359 stage6.drawio
-a---         2026/6/23 下午 11:29          36202 stage6.jpg
PS C:\Users\HC\Desktop\practice\practice_project\Stage6>

下午1:56
了解，你是在 Windows 本機執行。直接在那個資料夾跑：

powershell
python orb_video.py --input 123.mp4 --output 123_orb.mp4
不過你資料夾裡已經有 123.py 了，不需要用我剛才給的 orb_video.py。先看一下 123.py 內容是什麼，如果是你自己的舊版本我可以幫你整合或修改。

或者你想直接把 orb_video.py 也放進去一起用，那就把下載的檔案複製到 C:\Users\HC\Desktop\practice\practice_project\Stage6\ 然後執行上面那行指令即可。

寫在城市內

下午1:56
好，把 ORB 處理邏輯整合進你的 123.py。先看一下現有內容：


Read 123.py
Read 123.py
123.py 沒有上傳，請把內容貼過來或上傳檔案，我才能整合進去。

我說路徑寫在程式內

下午1:57
了解，就是在程式裡直接寫死路徑：


Edited 2 files, ran 2 commands, read a file
Edited 2 files, ran 2 commands, read a file
把 orb_video.py 放到和 123.mp4 同一個資料夾，直接執行：

powershell
python orb_video.py
輸出會自動存成 123_orb.mp4，不需要任何參數。


Orb video
PY 
點用成一個綠色小點就好

下午1:59
直接改 draw_rich 和 color：


Edited 2 files, ran 2 commands, read a file
Edited 2 files, ran 2 commands, read a file
改成每個特徵點畫一個半徑 3px 的實心綠點（cv2.circle 填滿），不再有方向圓圈或大小資訊。


Orb video
PY 
目前有做跨禎匹配嗎

下午2:01
沒有，目前程式只做單幀偵測：每幀獨立跑 ORB detect + compute，特徵點之間沒有任何跨幀關係。

每幀的綠點是各自獨立算出來的，不會追蹤同一個點在下一幀跑到哪裡。

如果要做跨幀匹配，常見兩種方向：

1. BFMatcher / FLANN（特徵描述子匹配）
用 ORB 的描述子，比對前後兩幀哪些點「長得像」，畫出對應線。適合靜態場景比對。

2. Optical Flow（光流追蹤）
用 cv2.calcOpticalFlowPyrLK，把上一幀的特徵點位置帶入下一幀追蹤，畫出移動軌跡。適合連續動態追蹤。

要加哪一種？

提高點的數量

下午2:02

Edited a file, ran a command, read a file
Edited a file, ran a command, read a file
nfeatures 從 500 調高到 5000。如果點還是不夠密，可以繼續往上加，ORB 理論上沒有上限，但太高會影響處理速度。


Orb video
PY 
給我SIFT版本

下午2:03

Ran 2 commands, viewed a file, read a file
Ran 2 commands, viewed a file, read a file
放到同資料夾直接執行，輸出 123_sift.mp4：

powershell
python sift_video.py
與 ORB 版本的主要差異：

ORB	SIFT
速度	快	慢（約 3~5x）
特徵點品質	一般	較精準、尺度不變
旋轉不變	有	有
授權	免費	OpenCV 4.4+ 免費
如果跑出來報錯 cv2.SIFT_create 找不到，代表 OpenCV 版本太舊，執行 pip install opencv-contrib-python 更新即可。


Sift video
PY 
SIFT orb誰的處理時間較快

你幫我在程式加上處理經過秒數

下午2:15
ORB 快很多，因為它是專為速度設計的（FAST 偵測 + BRIEF 描述子），SIFT 需要建多尺度高斯金字塔，計算量大約是 ORB 的 3~5 倍。

兩支程式都已經有 FPS 顯示在 HUD，只要加上總經過秒數就好：


Edited 2 files, ran 2 commands, read a file
Edited 2 files, ran 2 commands, read a file
兩支程式的 HUD 都新增了 Elapsed: X.Xs，影片跑完後終端機也會印出總耗時，方便直接比較兩者速度。


Orb video
PY 

Sift video
PY 

Claude Fable 5 is currently unavailable.
Learn more(opens in new tab)




Claude is AI and can make mistakes. Please double-check responses.
Sift video · PY
#!/usr/bin/env python3
"""
SIFT 特徵點抓取 - 影片版本
"""
 
import cv2
import sys
import time
 
def process_video(input_path: str, output_path: str | None = None, preview: bool = False,
                  nfeatures: int = 5000, contrast_threshold: float = 0.04,
                  edge_threshold: float = 10, sigma: float = 1.6):
 
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片: {input_path}")
        sys.exit(1)
 
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[資訊] {width}x{height} @ {fps:.1f}fps，共 {total} 幀")
 
    # 建立 SIFT 偵測器
    sift = cv2.SIFT_create(
        nfeatures         = nfeatures,
        contrastThreshold = contrast_threshold,
        edgeThreshold     = edge_threshold,
        sigma             = sigma,
    )
 
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[資訊] 輸出路徑: {output_path}")
 
    frame_idx = 0
    t0 = time.time()
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, _ = sift.detectAndCompute(gray, None)
 
        # 畫綠色小點
        vis = frame.copy()
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
 
        # HUD
        n_kp = len(keypoints)
        elapsed = time.time() - t0
        cur_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        cv2.putText(vis, f"Keypoints: {n_kp}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"Frame: {frame_idx+1}/{total}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, f"FPS: {cur_fps:.1f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, f"Elapsed: {elapsed:.1f}s", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
 
        if writer:
            writer.write(vis)
 
        if preview:
            cv2.imshow("SIFT Feature Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("[資訊] 使用者中斷")
                break
 
        frame_idx += 1
 
        if frame_idx % 100 == 0:
            print(f"  處理中... {frame_idx}/{total} ({100*frame_idx/total:.1f}%)")
 
    cap.release()
    if writer:
        writer.release()
        print(f"[完成] 已輸出 {frame_idx} 幀至: {output_path}")
    if preview:
        cv2.destroyAllWindows()
 
    total_time = time.time() - t0
    print(f"[完成] 共處理 {frame_idx} 幀，耗時 {total_time:.1f}s ({frame_idx/total_time:.1f} fps)")
 
 
def main():
    INPUT_PATH  = r"123.mp4"
    OUTPUT_PATH = r"123_sift.mp4"
 
    process_video(
        input_path         = INPUT_PATH,
        output_path        = OUTPUT_PATH,
        preview            = False,
        nfeatures          = 5000,
        contrast_threshold = 0.04,
        edge_threshold     = 10,
        sigma              = 1.6,
    )
 
 
if __name__ == "__main__":
    main()
 

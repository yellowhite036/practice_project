import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label as scipy_label
import random

def segment_on_dt(img_gray, threshold=100):
    dt = cv2.distanceTransform(img_gray, cv2.DIST_L2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    dt = cv2.threshold(dt, threshold, 255, cv2.THRESH_BINARY)[1]

    lbl, ncc = scipy_label(dt)
    lbl[img_gray == 0] = lbl.max() + 1
    lbl = lbl.astype(np.int32)

    cv2.watershed(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), lbl)
    lbl[lbl == -1] = 0
    return lbl, ncc


def run(image_path, threshold=100):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[錯誤] 找不到圖片：{image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    binary = 255 - binary  # 前景白、背景黑

    ws_result, _ = segment_on_dt(binary, threshold)

    # 偽彩色上色
    height, width = ws_result.shape
    ws_color = np.zeros((height, width, 3), dtype=np.uint8)
    lbl, ncc = scipy_label(ws_result)
    print(f"偵測到物件數量：{ncc}")

    for l in range(1, ncc + 1):
        a, b = np.nonzero(lbl == l)
        if binary[a[0], b[0]] == 0:  # 跳過背景
            continue
        rgb = [random.randint(0, 255) for _ in range(3)]
        ws_color[lbl == l] = tuple(rgb)

    # 顯示
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("Binary")
    axes[1].axis("off")
    axes[2].imshow(ws_color)
    axes[2].set_title(f"Segmented (threshold={threshold})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("watershed_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("結果已儲存為 watershed_result.png")


# ← 修改圖片路徑和閾值
run("2.jpg", threshold=150)
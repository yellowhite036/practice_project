#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys

import cv2
import numpy as np

# 預設色盤（BGR），依序循環分配給新建立的 ROI
PALETTE_BGR = [
    (0, 0, 255),      # 紅
    (0, 165, 255),    # 橙
    (0, 255, 255),    # 黃
    (0, 255, 0),       # 綠
    (255, 255, 0),    # 青
    (255, 0, 0),       # 藍
    (255, 0, 255),    # 紫
    (180, 105, 255),  # 粉
    (128, 128, 0),     # 深青
    (0, 128, 128),     # 深黃
]

VERTEX_RADIUS = 6          # 頂點顯示半徑（畫面座標）
VERTEX_HIT_RADIUS = 10     # 判定「點中頂點」的容許範圍（畫面座標）
FILL_ALPHA = 0.3           # 半透明填色透明度
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720


class ROI:
    """單一 ROI 物件"""
    _next_id = 1

    def __init__(self, name, color_bgr, polygon=None, roi_id=None):
        self.id = roi_id if roi_id is not None else ROI._next_id
        if roi_id is None:
            ROI._next_id += 1
        else:
            ROI._next_id = max(ROI._next_id, roi_id + 1)
        self.name = name
        self.color = tuple(color_bgr)
        # polygon：原始影片座標系的 [(x, y), ...]
        self.polygon = list(polygon) if polygon else []

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": "polygon",
            "color": list(self.color),
            "polygon": [[int(x), int(y)] for x, y in self.polygon],
        }

    @staticmethod
    def from_dict(d):
        return ROI(
            name=d.get("name", f"ROI{d.get('id')}"),
            color_bgr=tuple(d.get("color", (0, 255, 0))),
            polygon=[tuple(p) for p in d.get("polygon", [])],
            roi_id=d.get("id"),
        )


class ROIEditor:
    def __init__(self, video_path, roi_path="roi.json"):
        self.video_path = video_path
        self.roi_path = roi_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[錯誤] 無法開啟影片：{video_path}")
            sys.exit(1)
        ok, frame = cap.read()
        if not ok:
            print(f"[錯誤] 無法讀取影片第一幀：{video_path}")
            sys.exit(1)
        cap.release()

        self.frame = frame  # 原始影片解析度的第一幀 (BGR)
        self.frame_h, self.frame_w = frame.shape[:2]

        # 計算顯示縮放比例（原始座標 -> 顯示座標）
        self.scale = min(
            MAX_DISPLAY_W / self.frame_w,
            MAX_DISPLAY_H / self.frame_h,
            1.0,
        )
        self.disp_w = int(self.frame_w * self.scale)
        self.disp_h = int(self.frame_h * self.scale)
        self.help_bar_height = 50
        self.rois = []           # list[ROI]
        self.selected_idx = None  # 目前選取的 ROI index

        # 繪製中狀態
        self.drawing = False
        self.draw_points = []    # 原始座標系暫存點

        # 拖曳頂點狀態
        self.dragging = False
        self.drag_roi_idx = None
        self.drag_vertex_idx = None

        self.mouse_pos_orig = (0, 0)  # 滑鼠目前位置（原始座標系），用於繪製預覽線

        self.window_name = "ROI Editor  (N:新增 Enter:完成 U:undo Del:刪除 Tab:切換 S:存檔 L:讀檔 Q:離開)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        if os.path.exists(roi_path):
            self._load(roi_path)

    # ---------- 座標轉換 ----------
    def _to_orig(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def _to_disp(self, x, y):
        return int(x * self.scale), int(y * self.scale)

    # ---------- 滑鼠事件 ----------
    def _on_mouse(self, event, x, y, flags, param):
        ox, oy = self._to_orig(x, y)
        self.mouse_pos_orig = (ox, oy)

        if event == cv2.EVENT_LBUTTONDOWN:
            # 先判斷是否點到既有頂點 -> 進入拖曳模式
            hit = self._hit_test_vertex(x, y)
            if hit is not None:
                self.drag_roi_idx, self.drag_vertex_idx = hit
                self.dragging = True
                self.selected_idx = self.drag_roi_idx
                return

            if self.drawing:
                self.draw_points.append((ox, oy))
                return

            # 不在繪製模式、也沒點到頂點 -> 嘗試選取 ROI（點在多邊形內部）
            idx = self._hit_test_polygon(ox, oy)
            self.selected_idx = idx

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.drag_roi_idx is not None:
                self.rois[self.drag_roi_idx].polygon[self.drag_vertex_idx] = (ox, oy)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.dragging = False
                self.drag_roi_idx = None
                self.drag_vertex_idx = None

    def _hit_test_vertex(self, disp_x, disp_y):
        """回傳 (roi_idx, vertex_idx) 或 None，用畫面座標判斷距離"""
        for r_idx, roi in enumerate(self.rois):
            for v_idx, (ox, oy) in enumerate(roi.polygon):
                dx, dy = self._to_disp(ox, oy)
                if (dx - disp_x) ** 2 + (dy - disp_y) ** 2 <= VERTEX_HIT_RADIUS ** 2:
                    return r_idx, v_idx
        return None

    def _hit_test_polygon(self, ox, oy):
        """回傳點擊點所在的 ROI index（取最上層/最後建立者），或 None"""
        for idx in range(len(self.rois) - 1, -1, -1):
            poly = np.array(self.rois[idx].polygon, dtype=np.int32)
            if len(poly) < 3:
                continue
            if cv2.pointPolygonTest(poly, (float(ox), float(oy)), False) >= 0:
                return idx
        return None

    # ---------- ROI 動作 ----------
    def _start_new_roi(self):
        self.drawing = True
        self.draw_points = []
        self.selected_idx = None
        print("[提示] 進入新增 ROI 模式，請用左鍵點擊至少 3 個頂點，完成後按 Enter。")

    def _undo_point(self):
        if self.drawing and self.draw_points:
            self.draw_points.pop()
            print("[Undo] 取消上一個點")
        else:
            print("[Undo] 目前非繪製狀態，無點可取消")

    def _finish_roi(self):
        if not self.drawing:
            return
        if len(self.draw_points) < 3:
            print("[錯誤] ROI 至少需要 3 個點才能完成")
            return
        color = PALETTE_BGR[len(self.rois) % len(PALETTE_BGR)]
        name = f"LANE{len(self.rois) + 1}"
        roi = ROI(name=name, color_bgr=color, polygon=self.draw_points)
        self.rois.append(roi)
        self.selected_idx = len(self.rois) - 1
        self.drawing = False
        self.draw_points = []
        print(f"[完成] 已新增 ROI「{name}」，共 {len(roi.polygon)} 個點")

    def _delete_selected(self):
        if self.selected_idx is None:
            print("[提示] 目前沒有選取任何 ROI")
            return
        removed = self.rois.pop(self.selected_idx)
        print(f"[刪除] ROI「{removed.name}」已刪除")
        self.selected_idx = None

    def _cycle_selection(self):
        if not self.rois:
            return
        if self.selected_idx is None:
            self.selected_idx = 0
        else:
            self.selected_idx = (self.selected_idx + 1) % len(self.rois)
        print(f"[選取] 目前選取：{self.rois[self.selected_idx].name}")

    # ---------- 存讀檔 ----------
    def _save(self):
        path = self.roi_path
        data = {
            "video": os.path.basename(self.video_path),
            "frame_width": self.frame_w,
            "frame_height": self.frame_h,
            "rois": [r.to_dict() for r in self.rois],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[儲存] 已寫入 {path}（共 {len(self.rois)} 個 ROI）")

    def _load(self, path=None):
        path = path or self.roi_path
        if not os.path.exists(path):
            print(f"[錯誤] 找不到檔案：{path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.rois = [ROI.from_dict(d) for d in data.get("rois", [])]
        self.selected_idx = None
        self.drawing = False
        self.draw_points = []
        print(f"[載入] 已讀取 {path}（共 {len(self.rois)} 個 ROI）")

    # ---------- 繪製 ----------
    def _render(self):
        base = cv2.resize(self.frame, (self.disp_w, self.disp_h))
        overlay = base.copy()

        # 已完成的 ROI：半透明填色 + 邊框 + 頂點
        for idx, roi in enumerate(self.rois):
            if len(roi.polygon) < 3:
                continue
            pts_disp = np.array(
                [self._to_disp(x, y) for x, y in roi.polygon], dtype=np.int32
            )
            cv2.fillPoly(overlay, [pts_disp], roi.color)
            thickness = 3 if idx == self.selected_idx else 2
            cv2.polylines(base, [pts_disp], True, roi.color, thickness)
            for (dx, dy) in pts_disp:
                cv2.circle(base, (dx, dy), VERTEX_RADIUS, roi.color, -1)
                cv2.circle(base, (dx, dy), VERTEX_RADIUS, (255, 255, 255), 1)
            label_pos = tuple(pts_disp[0] + np.array([5, -8]))
            cv2.putText(base, roi.name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(base, roi.name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, roi.color, 1, cv2.LINE_AA)

        base = cv2.addWeighted(overlay, FILL_ALPHA, base, 1 - FILL_ALPHA, 0)

        # 繪製中的 ROI：畫線 + 頂點 + 到滑鼠游標的預覽線
        if self.drawing and self.draw_points:
            pts_disp = [self._to_disp(x, y) for x, y in self.draw_points]
            for i in range(len(pts_disp) - 1):
                cv2.line(base, pts_disp[i], pts_disp[i + 1], (0, 255, 0), 2)
            mouse_disp = self._to_disp(*self.mouse_pos_orig)
            cv2.line(base, pts_disp[-1], mouse_disp, (0, 255, 0), 1, cv2.LINE_AA)
            for p in pts_disp:
                cv2.circle(base, p, VERTEX_RADIUS, (0, 255, 0), -1)

        self._draw_info_panel(base)

        # 建立含底部提示列的完整畫布，影片內容完全不受影響
        canvas = np.zeros((self.disp_h + self.help_bar_height, self.disp_w, 3), dtype=np.uint8)
        canvas[: self.disp_h, :, :] = base
        self._draw_help_bar(canvas)

        cv2.imshow(self.window_name, canvas)

    def _draw_info_panel(self, base):
        y = 20
        cv2.putText(base, f"ROIs: {len(self.rois)}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for idx, roi in enumerate(self.rois):
            y += 20
            mark = ">" if idx == self.selected_idx else " "
            text = f"{mark} {roi.name}  {len(roi.polygon)} points"
            cv2.putText(base, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, roi.color, 1, cv2.LINE_AA)
        if self.drawing:
            y += 20
            cv2.putText(base, f"[繪製中] {len(self.draw_points)} points (Enter 完成 / U undo)",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
    def _draw_help_bar(self, canvas):
        y0 = self.disp_h
        cv2.rectangle(canvas, (0, y0), (self.disp_w, y0 + self.help_bar_height), (30, 30, 30), -1)

        line1 = "N:新增  Enter:完成  U:undo  Delete:刪除"
        line2 = "Tab:切換  S:存檔  L:讀檔  Q/ESC:離開"

        cv2.putText(canvas, line1, (10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (10, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)        

    # ---------- 主迴圈 ----------
    def run(self):
        print("=" * 60)
        print("ROI Editor 已啟動")
        print("N:新增ROI  Enter:完成  U:undo  Delete:刪除選取ROI")
        print("Tab:切換選取  S:儲存  L:載入  Q/ESC:離開")
        print("=" * 60)
        while True:
            self._render()
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # Q / ESC
                break
            elif key in (ord('n'), ord('N')):
                self._start_new_roi()
            elif key in (13, 10):  # Enter
                self._finish_roi()
            elif key in (ord('u'), ord('U')):
                self._undo_point()
            elif key in (8, 127, ord('d'), ord('D')):  # Delete / Backspace / D
                self._delete_selected()
            elif key == 9:  # Tab
                self._cycle_selection()
            elif key in (ord('s'), ord('S')):
                self._save()
            elif key in (ord('l'), ord('L')):
                self._load()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO 車輛分析用 ROI 標註工具")
    parser.add_argument("--video", default="../../123.mp4", help="影片路徑（支援 mp4 / avi，預設 ../../123.mp4）")
    parser.add_argument("--roi", default="roi.json", help="ROI 存讀檔路徑（預設 roi.json）")
    args = parser.parse_args()

    editor = ROIEditor(args.video, args.roi)
    editor.run()


if __name__ == "__main__":
    main()
# run_pipeline.py
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_VIDEO = ROOT / "123.mp4"

# ==================== 直播擷取設定 ====================
LIVE_SCRIPT = ROOT / "live.py"
LIVE_STREAM_URL = "https://www.youtube.com/watch?v=Nz9-_x5ecWc"
DURATION_SECONDS = 60
# ====================================================

MODES = {
    "0": {"name": "SKIP", "workdir": None, "script": None},
    "1": {"name": "FrameDiff", "workdir": ROOT / "Stage1" / "FrameDiff", "script": "FrameDiff.py"},
    "2": {"name": "KNN", "workdir": ROOT / "Stage1" / "KNN", "script": "KNN.py"},
    "3": {"name": "MOG2", "workdir": ROOT / "Stage1" / "MOG2", "script": "MOG2.py"},
    "4": {"name": "OPTICAL_FLOW", "workdir": ROOT / "Stage1" / "OPTICAL_FLOW", "script": "OPTICAL_FLOW.py"},
    "5": {"name": "YOLO", "workdir": ROOT / "Stage1" / "YOLO", "script": "YOLO.py"},
}

MODES1 = {
    "0": {"name": "SKIP", "workdir": None, "script": None},
    "1": {"name": "DENSE_OPTICAL_FLOW", "workdir": ROOT / "Stage5" / "code", "script": "dense_optical_flow.py"},
    "2": {"name": "SPARSE_OPTICAL_FLOW", "workdir": ROOT / "Stage5" / "code", "script": "sparse_optical_flow.py"},
}

STAGE4_DIR = ROOT / "Stage4" / "code"
STAGE4_SCRIPT = "vehicle_yolo_tracker.py"
STAGE4_OUTPUT_NAME = "YOLO_result.mp4"

RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "log"
MERGE_SCRIPT = ROOT / "merge_video.py"
MERGE_OUTPUT = ROOT / "output.mp4"

STAGE4_MODELS = {
    "1": "yolov8n.pt",
    "2": "yolov8s.pt",
    "3": "yolov8m.pt",
}

STAGE4_IMGSZ_OPTIONS = {
    "1": 640,
    "2": 960,
    "3": 1280,
    "4": 1920,
}

STAGE4_CONF_OPTIONS = {
    "1": 0.15,
    "2": 0.25,
    "3": 0.35,
    "4": 0.5,
}


# ==================== Log 自動輸出工具 ====================
class Tee:
    """將輸出同時寫到多個串流（例如：畫面 + log 檔案）"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def unique_path(path: Path) -> Path:
    """若 path 已存在，就在檔名後面加上 -2、-3... 直到找到沒被佔用的檔名，
    而不是直接覆蓋既有檔案。"""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def run_subprocess_logged(cmd, cwd) -> int:

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    for line in process.stdout:
        sys.stdout.write(line)
    process.wait()
    sys.stdout.flush()
    return process.returncode
# ==========================================================


def run_live_capture():
    """執行 live.py 擷取直播"""
    print(f"\n{'=' * 60}")
    print("開始執行 live.py 擷取直播畫面...")
    print(f"來源：{LIVE_STREAM_URL}  ({DURATION_SECONDS} 秒)")
    print(f"{'=' * 60}")

    if not LIVE_SCRIPT.exists():
        print(f"找不到 live.py：{LIVE_SCRIPT}")
        return False

    returncode = run_subprocess_logged([sys.executable, str(LIVE_SCRIPT)], ROOT)

    if returncode != 0:
        print("live.py 執行失敗")
        return False

    if not SOURCE_VIDEO.exists() or SOURCE_VIDEO.stat().st_size == 0:
        print("擷取失敗，123.mp4 不存在或大小為 0")
        return False

    print(f"直播擷取完成 → {SOURCE_VIDEO} ({SOURCE_VIDEO.stat().st_size / 1024:.1f} KB)")
    return True


def parse_order(raw: str, active_stages: list):
    normalized = raw.replace("-", " ").replace(",", " ").strip()
    tokens = [t.strip() for t in normalized.split() if t.strip() in ("1", "4", "5")]
    seen = set()
    order = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            order.append(t)
    if sorted(order) == sorted(active_stages):
        return order
    return None


def select_stages_and_order():
    print("=" * 80)
    print("=== Pipeline 自訂模式 ===")
    print("請決定各 Stage 是否執行（0=跳過，1=執行）")
    print("=" * 80)
    
    while True:
        s1 = input("Stage1 去背 (0/1)：").strip()
        s4 = input("Stage4 追蹤 (0/1)：").strip()
        s5 = input("Stage5 光流 (0/1)：").strip()
        if s1 in ('0','1') and s4 in ('0','1') and s5 in ('0','1'):
            break
        print("請只輸入 0 或 1")

    do_bg = s1 == "1"
    do_track = s4 == "1"
    do_flow = s5 == "1"

    active = []
    if do_bg: active.append("1")
    if do_track: active.append("4")
    if do_flow: active.append("5")

    if len(active) <= 1:
        return do_bg, do_track, do_flow, active

    print(f"\n目前要執行的 Stage： {' → '.join(active)}")
    print("請輸入執行順序（例如：4 1 5、4-5-1、1,4,5）")
    
    while True:
        raw = input("請輸入順序：").strip()
        order = parse_order(raw, active)
        if order:
            print(f"✓ 順序已設定：{' → '.join(order)}")
            return do_bg, do_track, do_flow, order
        print(f"輸入錯誤！請使用這些 Stage，例如：{' '.join(active)}")


def select_modes(do_bg):
    if not do_bg:
        return [MODES["0"]]
    
    print("=" * 60)
    print("請選擇 Stage1 去背模式（可複選，例如 1+2）")
    for k, v in MODES.items():
        if k != "0":
            print(f"{k}. {v['name']}")
    print("=" * 60)

    while True:
        raw = input("請輸入 (1~5，可用 + 或 , 分隔)：").strip()
        selected = parse_selection(raw, {k:v for k,v in MODES.items() if k!="0"})
        if selected:
            return [MODES[k] for k in selected]
        print("輸入錯誤，請確認 1~5")


def select_modes1(do_flow):
    if not do_flow:
        return [MODES1["0"]]
    
    print("=" * 60)
    print("請選擇 Stage5 光流模式（可複選，例如 1+2）")
    for k, v in MODES1.items():
        if k != "0":
            print(f"{k}. {v['name']}")
    print("=" * 60)

    while True:
        raw = input("請輸入 (1~2，可用 + 或 , 分隔)：").strip()
        selected = parse_selection(raw, {k:v for k,v in MODES1.items() if k!="0"})
        if selected:
            return [MODES1[k] for k in selected]
        print("輸入錯誤，請確認 1~2")


def select_stage4_settings(do_track):
    """選擇 Stage4 YOLO 追蹤要用的模型、推論解析度與信心閾值。"""
    if not do_track:
        return None, None, None

    print("=" * 60)
    print("請選擇 Stage4 YOLO 模型")
    for k, v in STAGE4_MODELS.items():
        print(f"{k}. {v}")
    print("=" * 60)

    while True:
        raw = input("請輸入模型編號 (1~3，預設 1)：").strip() or "1"
        if raw in STAGE4_MODELS:
            model = STAGE4_MODELS[raw]
            break
        print("輸入錯誤，請確認 1~3")

    print("=" * 60)
    print("請選擇 Stage4 推論解析度 (imgsz)")
    for k, v in STAGE4_IMGSZ_OPTIONS.items():
        print(f"{k}. {v}")
    print("=" * 60)

    while True:
        raw = input("請輸入解析度編號 (1~4，預設 1)：").strip() or "1"
        if raw in STAGE4_IMGSZ_OPTIONS:
            imgsz = STAGE4_IMGSZ_OPTIONS[raw]
            break
        print("輸入錯誤，請確認 1~4")

    print("=" * 60)
    print("請選擇 Stage4 偵測信心閾值 (conf)")
    for k, v in STAGE4_CONF_OPTIONS.items():
        print(f"{k}. {v}")
    print("5. 自訂數值")
    print("=" * 60)

    while True:
        raw = input("請輸入信心閾值編號 (1~5，預設 1)：").strip() or "1"
        if raw in STAGE4_CONF_OPTIONS:
            conf = STAGE4_CONF_OPTIONS[raw]
            break
        if raw == "5":
            custom = input("請輸入自訂信心閾值 (0~1 之間，例如 0.2)：").strip()
            try:
                conf = float(custom)
                if 0 <= conf <= 1:
                    break
                print("數值必須介於 0 到 1 之間")
            except ValueError:
                print("請輸入有效的數字")
            continue
        print("輸入錯誤，請確認 1~5")

    print(f"✓ Stage4 設定：model={model} | imgsz={imgsz} | conf={conf}")
    return model, imgsz, conf

def parse_selection(raw: str, valid):
    normalized = raw.replace("+", " ").replace(",", " ")
    tokens = normalized.split()
    selected = []
    for t in tokens:
        if t in valid and t not in selected:
            selected.append(t)
    return selected if selected else None


def build_step(step_type: str, mode: dict, mode5: dict, source: Path, output_dir=None,
                stage4_model: str = None, stage4_imgsz: int = None, stage4_conf: float = None):
    if step_type in ("1", "bg"):
        if mode["name"] == "SKIP": return None
        out_dir = output_dir or mode["workdir"]
        output = out_dir / f"{mode['name']}.mp4"
        cmd = ["python", mode["script"], "--source", str(source), "--output", str(output)]
        return f"Stage1 - {mode['name']}", mode["workdir"], cmd, output

    if step_type in ("4", "track"):
        out_dir = output_dir or STAGE4_DIR
        output = out_dir / STAGE4_OUTPUT_NAME
        cmd = ["python", STAGE4_SCRIPT, "--source", str(source), "--output", str(output)]
        if stage4_model:
            cmd += ["--model", stage4_model]
        if stage4_imgsz:
            cmd += ["--imgsz", str(stage4_imgsz)]
        if stage4_conf is not None:
            cmd += ["--conf", str(stage4_conf)]
        return "Stage4 - YOLO 追蹤", STAGE4_DIR, cmd, output

    if step_type in ("5", "flow"):
        if mode5["name"] == "SKIP": return None
        out_dir = output_dir or mode5["workdir"]
        output = out_dir / f"{mode5['name']}.mp4"
        cmd = ["python", mode5["script"], "--video", str(source), "--output", str(output)]
        return f"Stage5 - {mode5['name']}", mode5["workdir"], cmd, output


def run_one_combo(mode, mode5, order, stage4_model=None, stage4_imgsz=None, stage4_conf=None, combo_index=0):
    combo_start = time.time()
    current = SOURCE_VIDEO
    final = None

    # 先用暫存檔名記錄這個組合的完整輸出，等結果影片檔名確定後再改名成一樣的名稱
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tmp_log_path = LOG_DIR / f"_tmp_combo_{combo_index}.log"
    log_file = open(tmp_log_path, "a", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = sys.stdout

    try:
        for step in order:
            step_info = build_step(step, mode, mode5, current,
                                   RESULTS_DIR if step == order[-1] else None,
                                   stage4_model=stage4_model, stage4_imgsz=stage4_imgsz,
                                   stage4_conf=stage4_conf)
            if step_info is None:
                continue

            name, workdir, cmd, out_path = step_info
            print(f"\n{'='*50}\n執行 → {name}\n{'='*50}")

            returncode = run_subprocess_logged(cmd, workdir)
            if returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
                elapsed = time.time() - combo_start
                sys.stdout, sys.stderr = original_stdout, original_stderr
                log_file.close()
                fail_log_path = unique_path(LOG_DIR / f"combo{combo_index}_failed.log")
                tmp_log_path.replace(fail_log_path)
                print(f"（本組合 log 已存於：{fail_log_path}）")
                return False, f"{name} 執行失敗", elapsed

            print(f"完成 → {out_path.name}")
            current = out_path
            final = out_path

        # 檔名
        tags = []
        for s in order:
            if s == "1": tags.append(mode["name"].lower() if mode["name"] != "SKIP" else "skip")
            elif s == "4":
                model_tag = (stage4_model or "yolov8n.pt").replace("yolov8", "").replace(".pt", "")
                imgsz_tag = str(stage4_imgsz) if stage4_imgsz else "640"
                conf_tag = str(stage4_conf) if stage4_conf is not None else "0.15"
                tags.append(f"track-{model_tag}-{imgsz_tag}-c{conf_tag}")
            elif s == "5": tags.append(mode5["name"].lower().replace("_optical_flow", "") if mode5["name"] != "SKIP" else "skip")

        duration_tag = f"{int(time.time()-combo_start)//60}.{int(time.time()-combo_start)%60:02d}"
        final_name = "-".join(tags) + f"-{duration_tag}.mp4"
        final_path = unique_path(final.parent / final_name)
        final.replace(final_path)

        # log 檔名跟影片檔名一樣（只差副檔名），但存到 results/log 資料夾裡
        log_final_path = unique_path(LOG_DIR / f"{final_path.stem}.log")
        print(f"（本組合 log 已存於：{log_final_path}）")
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_file.close()
        tmp_log_path.replace(log_final_path)

        return True, final_path, time.time() - combo_start
    finally:
        # 保險：不論成功或例外，都要把 stdout/stderr 還原
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: 先選擇是否擷取
    print(f"\n{'=' * 60}")
    print("是否要執行 live.py 擷取直播畫面？")
    do_live = input("輸入 y 擷取，其餘則使用現有 123.mp4：").strip().lower() == 'y'

    # Step 2: 進行其他所有設定
    do_bg, do_track, do_flow, order = select_stages_and_order()
    modes = select_modes(do_bg)
    modes5 = select_modes1(do_flow)
    stage4_model, stage4_imgsz, stage4_conf = select_stage4_settings(do_track)

    # Step 3: 所有設定完成後，才開始擷取（如果選擇要的話）
    if do_live:
        if not run_live_capture():
            print("直播擷取失敗，程式終止。")
            sys.exit(1)
    else:
        if not SOURCE_VIDEO.exists():
            print(f"找不到 123.mp4 且未選擇擷取，程式無法繼續。")
            sys.exit(1)
        print(f"使用現有來源影片：{SOURCE_VIDEO} ({SOURCE_VIDEO.stat().st_size / 1024:.1f} KB)")

    # Step 4: 開始執行 Pipeline
    # 每個組合執行時會自動記錄 log，執行完成後 log 檔名會跟輸出的影片檔名相同（副檔名 .log）
    combos = list(itertools.product(modes, modes5))
    print(f"\n共 {len(combos)} 組合，執行順序：{' → '.join(order)}\n")

    for i, (m, m5) in enumerate(combos, 1):
        print(f"開始第 {i}/{len(combos)} 組合...")
        success, info, elapsed = run_one_combo(m, m5, order, stage4_model, stage4_imgsz, stage4_conf, combo_index=i)
        status = "成功" if success else "失敗"
        print(f"組合 {status}！耗時 {elapsed:.1f} 秒 → {info if success else '失敗'}")

    print("\n全部執行完畢！")


if __name__ == "__main__":
    main()
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LIVE_STREAM_URL = "https://www.youtube.com/watch?v=Nz9-_x5ecWc"
DURATION_SECONDS = 60
OUTPUT_PATH = ROOT / "123.mp4"

print(f"{'=' * 60}")
print(f" 擷取直播畫面：{LIVE_STREAM_URL}")
print(f" 長度：{DURATION_SECONDS} 秒")
print(f" 輸出：{OUTPUT_PATH}")
print(f"{'=' * 60}")

# 用 yt-dlp 解析出直播的實際串流網址（.m3u8）
# 用 `python -m yt_dlp` 而不是直接呼叫 `yt-dlp`，
# 避免 Windows 上 pip 安裝的執行檔沒加進 PATH 導致「找不到指令」
print(f"使用的 Python：{sys.executable}")

try:
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-g", "-f", "best", LIVE_STREAM_URL],
        capture_output=True,
        text=True,
        check=True,
    )
except FileNotFoundError as e:
    print(f"❌ 找不到 yt-dlp 套件，請先執行：{sys.executable} -m pip install yt-dlp")
    print(f"（詳細錯誤：{e}）")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"❌ yt-dlp 解析直播網址失敗：{e.stderr.strip()}")
    sys.exit(1)

stream_url = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
if not stream_url:
    print("❌ yt-dlp 沒有回傳有效的串流網址")
    sys.exit(1)

print(f"已取得串流網址，開始錄製...")

# 用 imageio-ffmpeg 套件內建的 ffmpeg 執行檔，
# 不用另外安裝 ffmpeg、也不用處理系統 PATH
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    print(f"❌ 找不到 imageio-ffmpeg 套件，請先執行：{sys.executable} -m pip install imageio-ffmpeg")
    sys.exit(1)

print(f"使用的 ffmpeg：{ffmpeg_path}")

# 如果舊檔案存在就先刪除，避免 ffmpeg 詢問是否覆蓋卡住
if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()

ffmpeg_result = subprocess.run(
    [
        ffmpeg_path,
        "-y",
        "-i", stream_url,
        "-t", str(DURATION_SECONDS),
        "-c", "copy",
        str(OUTPUT_PATH),
    ]
)

if ffmpeg_result.returncode != 0:
    print(f"❌ ffmpeg 擷取失敗（結束代碼 {ffmpeg_result.returncode}）")
    sys.exit(1)

if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
    print(f"❌ 擷取後找不到影片或檔案大小為 0：{OUTPUT_PATH}")
    sys.exit(1)

print(f"✅ 擷取完成：{OUTPUT_PATH}（{OUTPUT_PATH.stat().st_size / 1024:.1f} KB）")
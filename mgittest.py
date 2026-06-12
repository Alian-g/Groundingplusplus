import re
import shutil
import zipfile
from pathlib import Path

# =========================
# 需要修改的参数
# =========================
INPUT_DIR = Path(r"/data/ATCTrack/test/tracking_results/atctrack/atctrack_base_215-2/videocube_test_tiny")   # 原始结果文件夹
OUTPUT_BASE = Path(r"/data/ATCTrack/submit")       # 输出目录
TRACKER_NAME = "LGPTrack"                                # 改成你的算法名，例如 "LGPTrack"

# =========================
# 正则匹配
# =========================
pattern_result = re.compile(r"^(\d{3})\.txt$")
pattern_time = re.compile(r"^(\d{3})_time\.txt$")

result_files = {}
time_files = {}

for file in INPUT_DIR.iterdir():
    if not file.is_file():
        continue

    m1 = pattern_result.match(file.name)
    m2 = pattern_time.match(file.name)

    if m1:
        seq_id = m1.group(1)
        result_files[seq_id] = file
    elif m2:
        seq_id = m2.group(1)
        time_files[seq_id] = file

all_ids = sorted(set(result_files.keys()) | set(time_files.keys()))
missing_result = [sid for sid in all_ids if sid not in result_files]
missing_time = [sid for sid in all_ids if sid not in time_files]

if missing_result:
    print("以下序列缺少 result 文件：", missing_result)
if missing_time:
    print("以下序列缺少 time 文件：", missing_time)

# =========================
# 创建提交目录结构
# =========================
submit_root = OUTPUT_BASE / TRACKER_NAME
result_dir = submit_root / "result"
time_dir = submit_root / "time"

if submit_root.exists():
    shutil.rmtree(submit_root)

result_dir.mkdir(parents=True, exist_ok=True)
time_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 把 bbox 文件从空格分隔改成逗号分隔
# 并统一写成 3 位小数：231.000,119.000,248.000,454.000
# =========================
def convert_bbox_file(src_path: Path, dst_path: Path):
    with open(src_path, "r", encoding="utf-8") as f_in, open(dst_path, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 兼容空格、多个空格、tab，甚至原本有逗号的情况
            parts = re.split(r"[\s,]+", line)

            if len(parts) != 4:
                raise ValueError(
                    f"文件 {src_path.name} 第 {line_num} 行不是4列，而是 {len(parts)} 列：{line}"
                )

            # 转成 float，并格式化成 3 位小数
            vals = [float(x) for x in parts]
            new_line = ",".join(f"{v:.3f}" for v in vals)

            f_out.write(new_line + "\n")

# =========================
# 复制并重命名
# result 文件：做分隔符转换
# time 文件：直接复制
# =========================
for sid in sorted(result_files.keys()):
    src = result_files[sid]
    dst = result_dir / f"{TRACKER_NAME}_{sid}.txt"
    convert_bbox_file(src, dst)

for sid in sorted(time_files.keys()):
    src = time_files[sid]
    dst = time_dir / f"{TRACKER_NAME}_{sid}.txt"
    shutil.copyfile(src, dst)

print(f"提交目录已生成：{submit_root}")

# =========================
# 打包 zip
# =========================
zip_path = OUTPUT_BASE / f"{TRACKER_NAME}.zip"

if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in submit_root.rglob("*"):
        if file.is_file():
            zf.write(file, arcname=file.relative_to(OUTPUT_BASE))

print(f"ZIP 文件已生成：{zip_path}")
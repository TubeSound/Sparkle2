import os
from datetime import datetime

# 対象フォルダ（フルパスで指定 or 相対パス）
TARGET_FOLDER = r"C:\Users\docs9\OneDrive\デスクトップ\2025_06_15_window_wall"  # ここを書き換えて

# 曜日略称（Python標準のMon, Tue, ...）
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

for filename in os.listdir(TARGET_FOLDER):
    if filename.upper().endswith(".WAV"):
        name, ext = os.path.splitext(filename)
        # ファイル名が14桁かチェック
        if len(name) == 14 and name.isdigit():
            try:
                dt = datetime.strptime(name, "%Y%m%d%H%M%S")
                week = WEEKDAYS[dt.weekday()]
                new_name = f"{dt.strftime('%Y-%m-%d')}" \
                           f"({week})" \
                           f"{dt.strftime('%H-%M-%S')}{ext}"
                src = os.path.join(TARGET_FOLDER, filename)
                dst = os.path.join(TARGET_FOLDER, new_name)
                os.rename(src, dst)
                print(f"Renamed: {filename} -> {new_name}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Skipped (name format mismatch): {filename}")
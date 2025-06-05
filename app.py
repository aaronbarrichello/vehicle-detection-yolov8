from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import uuid
import subprocess
from pathlib import Path
import glob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "runs/detect/YOLOv8s/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8s.pt"
model = YOLO(MODEL_PATH)

os.makedirs("temp", exist_ok=True)

def convert_to_mp4_h264(input_path, output_path):
    cmd = [
        r"D:\Downloads\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe",
        "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-an", output_path
    ]
    subprocess.run(cmd, check=True)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1].lower()
    input_id = str(uuid.uuid4())
    input_path = f"temp/{input_id}_input.{ext}"
    output_path = f"temp/{input_id}_output.{ext}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if ext in ["jpg", "jpeg", "png"]:
        results = model(input_path)
        results[0].save(filename=output_path)
        return FileResponse(output_path, media_type="image/jpeg")
    elif ext in ["mp4", "avi", "mov", "mkv"]:
        home = str(Path.home())
        detect_dir = os.path.join(home, "runs", "detect")
        results = model.track(
            source=input_path,
            save=True,
            conf=0.25,
            project=detect_dir
        )
        import time
        time.sleep(2)
        # Cari semua file video di semua folder track*
        video_files = []
        for track_dir in glob.glob(os.path.join(detect_dir, "track*")):
            if os.path.isdir(track_dir):
                for f in os.listdir(track_dir):
                    if f.endswith((".mp4", ".avi", ".mov", ".mkv")):
                        video_files.append(os.path.join(track_dir, f))
        if video_files:
            latest_video = max(video_files, key=os.path.getmtime)
            if latest_video.endswith(".mp4"):
                return FileResponse(latest_video, media_type="video/mp4")
            else:
                mp4_path = os.path.splitext(latest_video)[0] + "_converted.mp4"
                convert_to_mp4_h264(latest_video, mp4_path)
                return FileResponse(mp4_path, media_type="video/mp4")
        else:
            print("[DEBUG] Tidak ada file video output di semua folder track*")
        return JSONResponse({"error": "Output video tidak ditemukan."}, status_code=500)
    else:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400) 

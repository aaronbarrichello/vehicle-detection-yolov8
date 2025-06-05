# Vehicle Detection Web

## Backend (FastAPI)
1. Install dependencies: ip install -r requirements.txt
2. Jalankan backend: python -m uvicorn app:app --reload
3. Pastikan model YOLO (`best.pt`) sudah ada di `runs/detect/YOLOv8s/weights/best.pt` atau gunakan model default `yolov8s.pt`.

## Frontend (HTML)
- Buka file `index.html` di browser.
- Upload gambar/video, klik "Start Detection", hasil akan muncul di bawahnya.

## Catatan
- Pastikan backend dan frontend berjalan di komputer yang sama, atau sesuaikan URL pada `index.html` jika berbeda. 

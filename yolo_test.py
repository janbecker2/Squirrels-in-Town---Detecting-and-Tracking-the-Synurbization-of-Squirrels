from ultralytics import YOLO
import cv2 as cv

# model.train( 
# data=r"D:\squirrel.v1i.yolov11\data.yaml", 
# epochs=100, 
# imgsz=640, 
# batch=16, 
# name="squirrel_yolo11" 
# )

model = YOLO(r"runs/detect/squirrel_yolo11/weights/best.pt")
video_path = r"C:\Users\job02\Downloads\squirrel_vid_short.mp4"
video_outside_path = r"C:\Users\job02\Documents\Squirrel_Videos\outside\20241030_TrepN_04_out (8)_short.mp4"
output_path = r"C:\Users\job02\Downloads\squirrel_small_yolo_output_ouside.mp4"

cap = cv.VideoCapture(video_outside_path)
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

max_frames = int(fps * 30)
frame_count = 0

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    for r in results:
        frame = r.plot()

    out.write(frame)

    #cv.imshow('YOLO Detection', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv.destroyAllWindows()


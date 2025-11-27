from ultralytics import YOLO
import cv2 as cv

model = YOLO(r"runs/detect/squirrel_yolo11/weights/best.pt")
video_path = r"C:\Users\job02\Downloads\squirrel_vid_short.mp4"
output_path = r"C:\Users\job02\Downloads\squirrel_yolo_output.mp4"

cap = cv.VideoCapture(video_path)
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

from ultralytics import YOLO
import cv2 as cv

model = YOLO("yolo11n.pt")
# model.train( 
# data=r"D:\squirrel.v1i.yolov11\data.yaml", 
# epochs=100, 
# imgsz=640, 
# batch=16, 
# name="squirrel_yolo11" 
# )

# model.train(
#     data=r"yolo_dataset_from_labelbox_squirrel\data.yaml",
#     epochs=1,
#     imgsz=640,
#     batch=8,
#     name="squirrel_model"
# )

model = YOLO(r"runs/detect/squirrel_model2/weights/best.pt")
video_path = r"C:\Users\job02\Documents\Hoernchen\study_project_ws25_26\study_project_ws25_26\20241108_TrepS_01_in (2)_cut_updated.mp4"
#video_outside_path = r"C:\Users\job02\Documents\Squirrel_Videos\outside\20241030_TrepN_04_out (8)_short.mp4"
#output_path = r"C:\Users\job02\Downloads\squirrel_small_yolo_output_ouside.mp4"

cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*"mp4v")
#out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

#max_frames = int(fps * 30)
frame_count = 0

#while frame_count < max_frames:
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)

    # Always start with the original frame
    output_frame = frame.copy()

    # Draw detections only if present
    if len(results) > 0:
        output_frame = results[0].plot()

    # Write every frame
    #out.write(output_frame)

    # Debug:
    cv.imshow("out", output_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
#out.release()
cv.destroyAllWindows()

from ultralytics import YOLO
import cv2 as cv


#model = YOLO('yolo11n.pt') 
model = YOLO(r"runs/detect/squirrel_yolo11/weights/best.pt")

# Load video
# video_path = r"D:\squirrel_vid_short.mp4"
#video_path = r"C:\Users\job02\Documents\Squirrel_Videos\20241107_TrepS_01_in (5).MOV"
video_path = r"C:\Users\job02\Downloads\squirrel_vid_short.mp4"
cap = cv.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    for r in results:
        img_with_boxes = r.plot()  # Draw boxes on the frame

        print(f"\n Found {len(r.boxes)} objects in this frame")
        for i, box in enumerate(r.boxes):
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = model.names[class_id]
            print(f"  {i+1}. {class_name} ({confidence:.1%} confidence)")

        cv.imshow('YOLO Detection', img_with_boxes)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

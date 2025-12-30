import json
import os
import cv2

# Pfade
ndjson_file = r"C:\Users\job02\Documents\Hoernchen\Squirrels_in_town_annotations_12_12_2025.ndjson"
videos_path = r"C:\Users\job02\Documents\Hoernchen\study_project_ws25_26\study_project_ws25_26"
output_base = "yolo_dataset_from_labelbox"

os.makedirs(output_base, exist_ok=True)
class_map = {}
current_class_id = 0

# NDJSON einlesen
with open(ndjson_file, "r") as f:
    data = [json.loads(line) for line in f]

for item in data:
    video_name = os.path.splitext(item["data_row"]["external_id"])[0]
    video_file = os.path.join(videos_path, item["data_row"]["external_id"])
    media_width = item["media_attributes"]["width"]
    media_height = item["media_attributes"]["height"]

    images_dir = os.path.join(output_base, "images", video_name)
    labels_dir = os.path.join(output_base, "labels", video_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Video öffnen
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Video {video_file} konnte nicht geöffnet werden.")
        continue

    # Frames extrahieren
    frame_annotations = {}
    projects = item["projects"]
    for project_id, project_data in projects.items():
        labels = project_data["labels"]
        for label in labels:
            frames = label["annotations"]["frames"]
            frame_annotations.update(frames)  # alle Frame-Annotations sammeln

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_id = str(frame_count)
        txt_file = os.path.join(labels_dir, f"{frame_count:05d}.txt")
        image_file = os.path.join(images_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(image_file, frame)

        yolo_lines = []
        if frame_id in frame_annotations:
            objects = frame_annotations[frame_id]["objects"]
            for obj_id, obj in objects.items():
                class_name = obj["name"]
                if class_name not in class_map:
                    class_map[class_name] = current_class_id
                    current_class_id += 1
                class_id = class_map[class_name]

                bbox = obj["bounding_box"]
                x_center = bbox["left"] + bbox["width"] / 2
                y_center = bbox["top"] + bbox["height"] / 2
                x_center /= media_width
                y_center /= media_height
                width_norm = bbox["width"] / media_width
                height_norm = bbox["height"] / media_height

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")

        with open(txt_file, "w") as f:
            f.write("\n".join(yolo_lines))

    cap.release()

# Klassen speichern
with open(os.path.join(output_base, "classes.txt"), "w") as f:
    for class_name, class_id in sorted(class_map.items(), key=lambda x: x[1]):
        f.write(f"{class_name}\n")

print("Fertig! Alle Videos wurden verarbeitet.")
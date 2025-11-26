import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time


def detect_entry_position(videoName, min_radius=20, max_radius=100):
    cap = cv.VideoCapture(videoName)
    if not cap.isOpened():
        print("Fehler beim Öffnen des Videos")
        return None

    entry_circle = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 5)

        circles = cv.HoughCircles(
            blur,
            cv.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            entry_circle = max(circles[0, :], key=lambda c: c[2])
            break  

    cap.release()

    if entry_circle is not None:
        x, y, r = entry_circle
        x1, y1 = int(x - r), int(y - r)
        x2, y2 = int(x + r), int(y + r)
        return (x1, y1, x2, y2)
    else:
        return None

    
def detect_entry_state(videoName):
    entry_ROI = (910, 100, 1100, 300)
    # entry_ROI = detect_entry_position(video_path)
    # print(entry_ROI)
    print("Start Reading Video")
    cap = cv.VideoCapture(videoName)

    if not cap.isOpened():
        print("Error: Could not open video")
        return None, []

    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=32)
    scale = 1.0
    frame_count = 0
    timeline = []

 
    first_full_frame = None


    sx1, sy1, sx2, sy2 = [int(v * scale) for v in entry_ROI]
    ENTRY = (sx1, sy1, sx2, sy2)
    ENTRY_AREA = (sx2 - sx1) * (sy2 - sy1)
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)
            h, w, _ = frame.shape
            frame_area = w * h

            # Background subtraction 
            fgMask = backSub.apply(frame)
            mask = cv.threshold(fgMask, 200, 255, cv.THRESH_BINARY)[1]
            mask = cv.medianBlur(mask, 5)

            full_pixels  = np.sum(mask > 0)
            entry_pixels = np.sum(mask[sy1:sy2, sx1:sx2] > 0)

            similarity = entry_pixels / full_pixels 

            if similarity > 0.8 and full_pixels < 5000: 
                state = 1 # Head State
            elif  5000 < full_pixels < 20000:
                state = 2  # Partial State
            elif full_pixels >= 20000:
                state = 3  # Full State
                if first_full_frame is None:
                    first_full_frame = frame_count
            else:
                state = 0

           
            timeline.append(state)
            frame_count += 1
            print(f"Frame {frame_count} | full={full_pixels} | entry={entry_pixels} | similarity={similarity} | state {state}")


            x1, y1, x2, y2 = ENTRY
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, "ENTRY ROI", (x1, y1 - 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            labels = ["NONE", "HEAD", "PARTIAL", "FULL"]
            cv.putText(frame, f"State: {labels[state]}", (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            #cv.imshow("Foreground Mask (Full Frame)", mask)
            #cv.imshow("Entry Mask (ENTRY ROI)", mask[sy1:sy2, sx1:sx2])
            cv.imshow("SES Detection", frame)

        key = cv.waitKey(30)

        if key == 27:  # ESC
            break
        if key in (ord('p'), ord('P')):  # toggle pause
            paused = not paused

    cap.release()
    cv.destroyAllWindows()
    print(f"Frames processed: {frame_count}")

    return timeline


video_path = r"C:\Users\Jan\Downloads\20241107_TrepS_01_in (2)_cut.mp4"
#video_path = r"D:\squirrel_vid_short.mp4"

timeline = detect_entry_state(video_path)

plt.figure(figsize=(10,4))
plt.plot(timeline, label="SES state")
plt.yticks([0,1,2,3], ["none","head","partial","full"])
plt.xlabel("Frame")
plt.ylabel("SES State")
plt.grid(True)
plt.legend()
plt.show()

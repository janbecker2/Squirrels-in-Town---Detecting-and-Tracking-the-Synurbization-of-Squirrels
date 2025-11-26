import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time


Entry_ROI = (910, 100, 1100, 300)

def ses_classify_video(videoName):
    print("▶ Start Reading Video")
    cap = cv.VideoCapture(videoName)

    if not cap.isOpened():
        print("Error: Could not open video")
        return None, []

    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=32)
    scale = 1.0  # now using full resolution
    frame_count = 0
    timeline = []

    ever_full = False
    ever_partial = False
    first_full_frame = None
    MIN_STAY_FRAMES = 30

    sx1, sy1, sx2, sy2 = [int(v * scale) for v in Entry_ROI]
    ENTRY = (sx1, sy1, sx2, sy2)
    ENTRY_AREA = (sx2 - sx1) * (sy2 - sy1)

    while True:
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

        full_ratio  = full_pixels / frame_area
        entry_ratio = entry_pixels / ENTRY_AREA if ENTRY_AREA else 0

        similarity = entry_pixels / full_pixels if full_pixels else 0

        if similarity > 0.8 and full_ratio < 0.05:
            state = 1  # Head 
        elif 0.05 < full_ratio < 0.15:
            state = 2  # Partial
            ever_partial = True
        elif full_ratio >= 0.15:
            state = 3  # Full
            ever_full = True
            if first_full_frame is None:
                first_full_frame = frame_count
        else:
            state = 0  

        timeline.append(state)
        frame_count += 1

        print(f"Frame {frame_count:4d} | full={full_ratio:.3f} | entry={entry_ratio:.3f} | similarity={similarity:.3f} → state {state}")


        x1, y1, x2, y2 = ENTRY
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, "ENTRY ROI", (x1, y1 - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        labels = ["NONE", "HEAD", "PARTIAL", "FULL"]
        cv.putText(frame, f"State: {labels[state]}", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        #cv.imshow("Foreground Mask (Full Frame)", mask)
        #cv.imshow("Entry Mask (ENTRY ROI)", mask[sy1:sy2, sx1:sx2])

        debug = mask.copy()
        cv.putText(debug, f"Similarity: {similarity:.2f}", (30, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv.imshow("Mask Similarity Debug", debug)

        cv.imshow("SES Detection", frame)

        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"Frames processed: {frame_count}")

    if ever_full and first_full_frame is not None:
        stay_len = frame_count - first_full_frame
        if stay_len >= MIN_STAY_FRAMES:
            final_state = "(iii) fully entered and stayed"
        else:
            final_state = "(iii) fully entered but did NOT stay long"
    elif ever_partial:
        final_state = "(ii) entered ~50% but never fully entered"
    else:
        final_state = "(i) only head seen, never progressed"

    print("\n🎯 SES result:", final_state)
    return final_state, timeline


video_path = r"C:\Users\Jan\Downloads\20241107_TrepS_01_in (2)_cut.mp4"
result, timeline = ses_classify_video(video_path)

plt.figure(figsize=(10,4))
plt.plot(timeline, label="SES state")
plt.yticks([0,1,2,3], ["none","head","partial","full"])
plt.xlabel("Frame")
plt.ylabel("SES State")
plt.grid(True)
plt.legend()
plt.show()

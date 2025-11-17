import cv2 as cv
import numpy as np
import time

def cut_video_by_motion(videoName, outName, motion_threshold=5000, scale=0.5):
    start_time = time.time()
    print("Start processing")

    cap = cv.VideoCapture(videoName)
    if not cap.isOpened():
        print("Error: Cannot open input video")
        return

    # Video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * scale)
    
    # Output video writer (H.264 in MP4)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(outName, fourcc, fps, (width, height))

    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=32)

    written_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Resize
        if scale != 1.0:
            frame = cv.resize(frame, (width, height))

        # Foreground mask
        fgMask = backSub.apply(frame)

        # Count number of moving pixels
        motion_pixels = np.sum(fgMask > 0)

        # Keep frame only if motion > threshold
        if motion_pixels > motion_threshold:
            out.write(frame)
            written_frames += 1

    cap.release()
    out.release()

    print(f"Total frames: {total_frames}")
    print(f"Frames written: {written_frames}")
    print(f"Saved cut video to: {outName}")
    print(f"--- {time.time() - start_time:.2f} seconds ---")


# Run it:
cut_video_by_motion(
    r"C:\Users\job02\Downloads\squirrel_vid_short.mp4",
    r"C:\Users\job02\Downloads\squirrel_cut.mp4",
    motion_threshold=5000,   # adjust this
    scale=0.5                 # same scale you used
)

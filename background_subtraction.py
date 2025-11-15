import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time

def readVideo_BGSub(videoName):
    start_time = time.time()
    print("Start Reading Video")
    cap = cv.VideoCapture(videoName)

    if not cap.isOpened():
        print(f"Error: Could not open video '{videoName}'")
        return

    # Initialize both Background Subtractors
    #backSub_KNN = cv.createBackgroundSubtractorKNN(detectShadows=True) 
    backSub_MOG2 = cv.createBackgroundSubtractorMOG2(history = 500, varThreshold= 32)

    scale = 0.5
    differences_knn = []
    differences_mog2 = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale the frame for speed
        if scale != 1.0:
            frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)

        # Apply both background subtraction methods
        #fgMask_KNN = backSub_KNN.apply(frame)
        fgMask_MOG2 = backSub_MOG2.apply(frame)

        # Count motion pixels for both
        #count_knn = np.sum(fgMask_KNN > 0)
        count_mog2 = np.sum(fgMask_MOG2 > 0)
        #differences_knn.append(count_knn)
        differences_mog2.append(count_mog2)

        #cv.imshow("Original Frame", frame)
        #cv.imshow("KNN Foreground Mask", fgMask_KNN)
        #cv.imshow("MOG2 Foreground Mask", fgMask_MOG2)
        
        key = cv.waitKey(1)  


        frame_count += 1

    cap.release()
    cv.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")
    print(f"--- {time.time() - start_time:.2f} seconds ---")

    return differences_knn, differences_mog2

# Run
# video_path = r"C:\Users\job02\Downloads\squirrel_vid_short.mp4"
diffs_knn, diffs_mog2 = readVideo_BGSub(r"C:\Users\job02\Downloads\squirrel_vid_short.mp4")

#diffs_knn, diffs_mog2 = readVideo_BGSub(r"D:\squirrel_vid_short.mp4")

# Plot both methods side by side for comparison
plt.figure(figsize=(12, 5))

# --- KNN ---
# plt.subplot(1, 2, 1)
# plt.plot(diffs_knn, label='KNN', color='blue')
# plt.xlabel('Frame Number')
# plt.ylabel('Motion Pixels')
# plt.title('KNN Background Subtraction')
# plt.grid(True)
# plt.legend()

# --- MOG2 ---
plt.subplot(1, 2, 2)
plt.plot(diffs_mog2, label='MOG2', color='red')
plt.xlabel('Frame Number')
plt.ylabel('Motion Pixels')
plt.title('MOG2 Background Subtraction')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


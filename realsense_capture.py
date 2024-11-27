import pyrealsense2 as rs
import cv2
import numpy as np
import os 
import time 


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)


# Start streaming
pipeline.start(config)

N_images = 100
saved_images = 0

save_dir = "calibration_images/"
os.makedirs(save_dir, exist_ok=True)

save_time = time.time()

try:
    while saved_images < N_images:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        dictionary =  cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

        charuco = cv2.aruco.CharucoBoard((11,7), 0.025, 0.011, dictionary)
        size = charuco.getChessboardSize()
        n_corners_in_charuco = (size[0]-1)*(size[1]-1)
        # Find the charuco corners
        corners,ids, rej = cv2.aruco.detectMarkers(gray, dictionary)
        print(corners)
        if corners and len(corners)>0:
            n, corners, ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco)
            print(corners)
            # If found, save the frame
            color = (255,0,0)
            if corners is not None and n == n_corners_in_charuco: # internal corners
                color = (0,255,0)
                if  save_time + 1 < time.time():
                    
                    cv2.imwrite(f"{save_dir}/{saved_images}.png", color_image)
                    save_time = time.time()
                    print("Chessboard detected and frame saved.")
                    saved_images += 1
            if corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(color_image, corners, ids,color)

        # Display the resulting frame
        # display number of saved images 
        cv2.putText(color_image, f"Saved images: {saved_images}/{N_images}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('RealSense', color_image)


        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
import cv2 as cv
import numpy as np
import glob

def calibrate_from_video(camera_id, chessboard_width, chessboard_height, num_images):
    # Define the size of the chessboard (number of inner corners)
    chessboard_size = (chessboard_width, chessboard_height)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Open the video capture
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    collected_images = 0
    while collected_images < num_images:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)

            # Refine the corner locations
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw the corners on the frame
            cv.drawChessboardCorners(frame, chessboard_size, corners2, ret)

            # Display the resulting frame
            cv.imshow('Chessboard Corners', frame)

            # Wait for the user to press a key
            if cv.waitKey(1) == 27:  # ESC key
                break

            collected_images += 1

    # Release the video capture and destroy the windows
    cap.release()
    cv.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)

    # Undistort the images
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Undistort
        h, w = frame.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv.undistort(frame, mtx, dist, None, newcameramtx)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # Save the undistorted image
        cv.imwrite('undistorted_frame.jpg', dst)

        # Display the undistorted image
        cv.imshow('Undistorted Image', dst)

        # Wait for the user to press a key
        if cv.waitKey(1) == 27:  # ESC key
            break

    # Release the video capture and destroy the windows
    cap.release()
    cv.destroyAllWindows()

def calibrate_from_images(chessboard_width, chessboard_height):
    # Define the size of the chessboard (number of inner corners)
    chessboard_size = (chessboard_width, chessboard_height)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Load the images
    images = glob.glob('calib_gopro/*.JPG')

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)

            # Refine the corner locations
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw the corners on the image
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow('Chessboard Corners', img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)

    # Undistort the images
    for fname in images:
        img = cv.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite(f'undistorted_{fname}', dst)

        # Display the undistorted image
        cv.imshow('Undistorted Image', dst)
        cv.waitKey(500)

    cv.destroyAllWindows()

def main():
    print("Choose the source for calibration:")
    print("1. Video Stream")
    print("2. Stored Images")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        camera_id = int(input("Enter the camera ID: "))
        chessboard_width = int(input("Enter the number of inner corners in the chessboard width: "))
        chessboard_height = int(input("Enter the number of inner corners in the chessboard height: "))
        num_images = int(input("Enter the number of images to use for calibration: "))
        calibrate_from_video(camera_id, chessboard_width, chessboard_height, num_images)
    elif choice == '2':
        chessboard_width = int(input("Enter the number of inner corners in the chessboard width: "))
        chessboard_height = int(input("Enter the number of inner corners in the chessboard height: "))
        calibrate_from_images(chessboard_width, chessboard_height)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()

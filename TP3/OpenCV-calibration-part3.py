import cv2 as cv
import numpy as np
import glob

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
    chessboard_width = int(input("Enter the number of inner corners in the chessboard width: "))
    chessboard_height = int(input("Enter the number of inner corners in the chessboard height: "))
    calibrate_from_images(chessboard_width, chessboard_height)

if __name__ == "__main__":
    main()

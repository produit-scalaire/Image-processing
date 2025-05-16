import cv2 as cv
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113


def parse_command_line_arguments():  # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # L'argument -k/--kp est conservé pour la compatibilité ou une utilisation future,
    # mais pour l'affichage simultané GFTT/SIFT/ORB, il n'est plus déterminant.
    parser.add_argument("-k", "--kp", default="SIFT",
                        help="key point (or corner) detector (Note: GFTT, SIFT, and ORB will be shown simultaneously if image2 is not provided for a specific detector run)")
    parser.add_argument("-n", "--nbKp", default=None, type=int,
                        help="Number of key point desired (if configurable by the detector) ")
    parser.add_argument("-d", "--descriptor", default=False, type=bool, # Defaulting to False as extraction is not implemented yet
                        help="compute descriptor associated with detector (if available)")
    parser.add_argument("-m", "--matching", default="NORM_L1",
                        help="Brute Force norm: NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2")
    parser.add_argument("-i1", "--image1", default="./IMG_1_reduced.jpg", help="path to image1")
    parser.add_argument("-i2", "--image2", default=None, help="path to image2")
    return parser


def check_image_loaded(img, image_path=""): # Renamed from test_load_image
    if img is None or img.size == 0 or (img.shape[0] == 0) or (img.shape[1] == 0):
        print(f"Could not load image {image_path}!")
        print("Exiting now...")
        exit(1)


def load_gray_image(path):
    img = None
    gray = None
    if (path is not None):
        img = cv.imread(path)
        check_image_loaded(img, path) # Use the renamed function
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray


def display_image(img, image_window_name):
    if img is not None:
        cv.imshow(image_window_name, img)


def feature_detector(type_detector, gray_img, nb_kp):
    kp = []  # Initialiser à une liste vide
    if gray_img is not None:
        # Ensure the image is uint8
        current_dtype = gray_img.dtype
        if current_dtype != np.uint8: # Basic type check, can be more robust
            # Attempt to convert to uint8
            if np.max(gray_img) <= 1.0 and (current_dtype == np.float32 or current_dtype == np.float64): # Example: float in [0,1]
                gray_img_processed = np.array(gray_img * 255, dtype=np.uint8)
            elif gray_img.dtype == np.uint16: # Example: uint16
                 gray_img_processed = cv.convertScaleAbs(gray_img, alpha=(255.0/65535.0))
            else: # Fallback or other types
                 gray_img_processed = np.array(gray_img, dtype=np.uint8) # This might not be ideal for all types
            print(f"Converted image for {type_detector} from {current_dtype} to {gray_img_processed.dtype}")
            gray_img_to_use = gray_img_processed
        else:
            gray_img_to_use = gray_img


        if len(gray_img_to_use.shape) == 3 and gray_img_to_use.shape[2] == 3: # Ensure it's grayscale
            gray_img_to_use = cv.cvtColor(gray_img_to_use, cv.COLOR_BGR2GRAY)


        if type_detector == "GFTT":
            qualityLevel = 0.01
            minDistance = 10
            blockSize = 3
            useHarrisDetector = False
            k_harris = 0.04
            max_corners_gftt = nb_kp if nb_kp is not None and nb_kp > 0 else 100

            corners = cv.goodFeaturesToTrack(
                gray_img_to_use,
                maxCorners=max_corners_gftt,
                qualityLevel=qualityLevel,
                minDistance=minDistance,
                blockSize=blockSize,
                useHarrisDetector=useHarrisDetector,
                k=k_harris
            )
            if corners is not None:
                for i in range(corners.shape[0]):
                    kp.append(cv.KeyPoint(x=corners[i, 0, 0], y=corners[i, 0, 1], size=20)) # size is arbitrary for GFTT keypoints
            print(f"GFTT: Found {len(kp)} keypoints.")

        elif type_detector == "SIFT":
            try:
                sift_n_features = nb_kp if nb_kp is not None and nb_kp > 0 else 0
                sift = cv.SIFT_create(nfeatures=sift_n_features)
                kp_detected = sift.detect(gray_img_to_use, None)
                if kp_detected is not None:
                    kp = list(kp_detected)
                else:
                    kp = []
                print(f"SIFT: Found {len(kp)} keypoints.")
            except AttributeError: # SIFT might not be available
                print("SIFT is not available in your OpenCV build. Try 'pip install opencv-contrib-python'.")
                kp = []
            except cv.error as e:
                print(f"OpenCV error with SIFT: {e}")
                kp = []


        elif type_detector == "ORB":
            try:
                orb_n_features = nb_kp if nb_kp is not None and nb_kp > 0 else 500
                orb = cv.ORB_create(nfeatures=orb_n_features)
                kp_detected = orb.detect(gray_img_to_use, None)
                if kp_detected is not None:
                    kp = list(kp_detected)
                else:
                    kp = []
                print(f"ORB: Found {len(kp)} keypoints.")
            except cv.error as e:
                print(f"OpenCV error with ORB: {e}")
                kp = []
            except Exception as e:
                print(f"An error occurred with ORB detector: {e}")
                kp = []
        else:
            print(f"Detector type '{type_detector}' not recognized or supported.")
            kp = []
    else:
        print(f"Cannot apply {type_detector} detector: input image is None.")
        kp = []
    return kp


def feature_extractor(type_feat, gray_img, kp_list): # Changed img to gray_img to be consistent
    desc = None
    # GFTT does not inherently provide descriptors. You'd use another method (e.g., SIFT/ORB compute) on GFTT points.
    if not kp_list: # No keypoints to describe
        print(f"{type_feat}: No keypoints to compute descriptors for.")
        return None, kp_list # Return None for desc and original kp_list

    gray_img_to_use = gray_img # Assume gray_img is already correct type (uint8)
    if gray_img_to_use is not None and gray_img_to_use.dtype != np.uint8:
        # Simplified conversion, assuming it might be float or needs scaling
        if np.max(gray_img_to_use) <= 1.0:
             gray_img_to_use = np.array(gray_img_to_use * 255, dtype=np.uint8)
        else:
             gray_img_to_use = np.array(gray_img_to_use, dtype=np.uint8)

    if type_feat == "SIFT":
        try:
            sift = cv.SIFT_create() # nfeatures is for detection, not compute usually
            kp_updated, desc = sift.compute(gray_img_to_use, kp_list)
            print(f"SIFT: Computed descriptors for {len(kp_updated) if kp_updated is not None else 0} keypoints.")
            return desc, kp_updated
        except AttributeError:
            print("SIFT is not available for descriptor extraction.")
            return None, kp_list
        except cv.error as e:
            print(f"OpenCV error computing SIFT descriptors: {e}")
            return None, kp_list

    elif type_feat == "ORB":
        try:
            orb = cv.ORB_create() # nfeatures is for detection
            kp_updated, desc = orb.compute(gray_img_to_use, kp_list)
            print(f"ORB: Computed descriptors for {len(kp_updated) if kp_updated is not None else 0} keypoints.")
            return desc, kp_updated
        except cv.error as e:
            print(f"OpenCV error computing ORB descriptors: {e}")
            return None, kp_list
    elif type_feat == "GFTT":
        print("GFTT is a detector only. Descriptors need to be computed with another algorithm (e.g., SIFT or ORB).")
        # Example: To compute SIFT descriptors for GFTT points:
        # sift_for_gftt = cv.SIFT_create()
        # kp_updated, desc = sift_for_gftt.compute(gray_img_to_use, kp_list)
        # print(f"Computed SIFT descriptors for {len(kp_updated)} GFTT keypoints.")
        # return desc, kp_updated
        return None, kp_list # GFTT itself doesn't compute descriptors
    else:
        print(f"Descriptor type '{type_feat}' not recognized or supported for extraction.")
        return None, kp_list


def main():
    parser = parse_command_line_arguments()
    args = vars(parser.parse_args())

    print("Load image 1")
    img1, gray1 = load_gray_image(args["image1"])
    print("Load image 2")
    img2, gray2 = load_gray_image(args["image2"])

    if img1 is not None:
        cv.namedWindow("Image 1 Original")
        display_image(img1, "Image 1 Original")
    if img2 is not None:
        cv.namedWindow("Image 2 Original")
        display_image(img2, "Image 2 Original")

    detectors_to_run = ["GFTT", "SIFT", "ORB"]

    for detector_name in detectors_to_run:
        print(f"\n--- Applying {detector_name} detector ---")
        kp1_current, kp2_current = [], []
        desc1_current, desc2_current = None, None

        if gray1 is not None:
            kp1_current = feature_detector(detector_name, gray1.copy(), args["nbKp"])
            if img1 is not None and kp1_current: # Check if keypoints were found
                img_kp1_current = cv.drawKeypoints(img1.copy(), kp1_current, None,
                                                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                window_name_img1 = f"Image 1 {detector_name}"
                cv.namedWindow(window_name_img1)
                display_image(img_kp1_current, window_name_img1)

                if args["descriptor"]:
                    desc1_current, kp1_current = feature_extractor(detector_name, gray1.copy(), kp1_current)
                    if desc1_current is not None:
                        print(f"  Img1 {detector_name}: {len(kp1_current)} keypoints, {desc1_current.shape} descriptors")


        if gray2 is not None:
            kp2_current = feature_detector(detector_name, gray2.copy(), args["nbKp"])
            if img2 is not None and kp2_current: # Check if keypoints were found
                img_kp2_current = cv.drawKeypoints(img2.copy(), kp2_current, None,
                                                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                window_name_img2 = f"Image 2 {detector_name}"
                cv.namedWindow(window_name_img2)
                display_image(img_kp2_current, window_name_img2)

                if args["descriptor"]:
                    desc2_current, kp2_current = feature_extractor(detector_name, gray2.copy(), kp2_current)
                    if desc2_current is not None:
                        print(f"  Img2 {detector_name}: {len(kp2_current)} keypoints, {desc2_current.shape} descriptors")

        # Ici, vous pourriez ajouter la logique de matching si les descripteurs des deux images existent pour ce détecteur


    key = 0
    print("\nPress 'q' or ESC to quit.")
    while key != ESC_KEY and key != Q_KEY:
        key = cv.waitKey(0) # waitKey(0) waits indefinitely for a key stroke

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
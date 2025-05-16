import cv2 as cv
import sys
import numpy as np

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113

MIN_MATCH_COUNT_FOR_HOMOGRAPHY = 10  # Augmenté pour plus de robustesse


def load_gray_image(path):
    if (path != None):
        img = cv.imread(path)
        if img is None:
            print(f"AVERTISSEMENT: Impossible de lire l'image : {path}")
            return None, None
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img = None;
        gray = None
    return img, gray


def display_image(img, image_window_name):
    if img is not None: cv.imshow(image_window_name, img)


def feature_detector(type_detector, gray_img, nb_kp_desired):
    kp = []
    if gray_img is None: print(f"INFO [{type_detector}]: Image grise non fournie pour détection."); return kp
    if gray_img.dtype != np.uint8:
        if gray_img.max() <= 1.0 and (gray_img.dtype == np.float32 or gray_img.dtype == np.float64):
            gray_img = np.array(gray_img * 255, dtype=np.uint8)
        else:
            gray_img = cv.normalize(gray_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if len(gray_img.shape) == 3: gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2GRAY)

    if type_detector == "GFTT":
        params = {'maxCorners': nb_kp_desired if nb_kp_desired else 100, 'qualityLevel': 0.01, 'minDistance': 10}
        corners = cv.goodFeaturesToTrack(gray_img, **params)
        if corners is not None: kp = [cv.KeyPoint(x=c[0, 0], y=c[0, 1], size=20) for c in corners]
    elif type_detector == "SIFT":
        try:
            sift = cv.SIFT_create(nfeatures=(nb_kp_desired if nb_kp_desired else 0))
            kp_detected = sift.detect(gray_img, None)
            if kp_detected: kp = list(kp_detected)
        except Exception as e:
            print(f"ERREUR [Détecteur SIFT]: {e}. opencv-contrib-python requis?"); kp = []
    elif type_detector == "ORB":
        try:
            orb = cv.ORB_create(nfeatures=(nb_kp_desired if nb_kp_desired else 500))
            kp_detected = orb.detect(gray_img, None)
            if kp_detected: kp = list(kp_detected)
        except Exception as e:
            print(f"ERREUR [Détecteur ORB]: {e}"); kp = []
    else:
        print(f"AVERTISSEMENT: Détecteur '{type_detector}' inconnu."); kp = []
    return kp


def feature_extractor(extractor_type, gray_img, keypoints):
    kp_updated = keypoints;
    desc = None
    if gray_img is None: print(
        f"INFO [{extractor_type}]: Image grise non fournie pour l'extraction."); return kp_updated, desc
    if not keypoints: print(f"INFO [{extractor_type}]: Pas de points clés pour l'extraction."); return kp_updated, desc

    if gray_img.dtype != np.uint8:
        if gray_img.max() <= 1.0 and (gray_img.dtype == np.float32 or gray_img.dtype == np.float64):
            gray_img = np.array(gray_img * 255, dtype=np.uint8)
        else:
            gray_img = cv.normalize(gray_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    if extractor_type == "SIFT":
        try:
            sift = cv.SIFT_create()
            kp_updated, desc = sift.compute(gray_img, keypoints)
        except Exception as e:
            print(f"ERREUR [Extracteur SIFT]: {e}")
    elif extractor_type == "ORB":
        try:
            orb = cv.ORB_create()
            kp_updated, desc = orb.compute(gray_img, keypoints)
        except Exception as e:
            print(f"ERREUR [Extracteur ORB]: {e}")
    else:
        print(f"AVERTISSEMENT: Extracteur '{extractor_type}' inconnu.")

    if kp_updated is None: kp_updated = []
    return kp_updated, desc


def calculer_homographie(correspondances, kp1, kp2, methode_ransac, seuil_reproj):
    if not isinstance(kp1, (list, tuple)) or not isinstance(kp2, (list, tuple)): return None, None
    # ... (autres vérifications de types pour kp1, kp2 si nécessaire) ...
    if not correspondances or len(correspondances) < MIN_MATCH_COUNT_FOR_HOMOGRAPHY:
        print(
            f"INFO [Homographie]: Pas assez de correspondances ({len(correspondances)}) pour calculer (min: {MIN_MATCH_COUNT_FOR_HOMOGRAPHY}).")
        return None, None
    try:
        pts_source = np.float32([kp2[m.trainIdx].pt for m in correspondances]).reshape(-1, 1, 2)
        pts_destination = np.float32([kp1[m.queryIdx].pt for m in correspondances]).reshape(-1, 1, 2)
    except IndexError:
        return None, None
    except AttributeError:
        return None, None

    H, masque = cv.findHomography(pts_source, pts_destination, methode_ransac, seuil_reproj)

    if H is None:
        print("INFO [Homographie]: findHomography a retourné None (pas de consensus RANSAC).")
    else:
        print("INFO [Homographie]: Matrice d'homographie calculée.")
        print("DEBUG [Homographie Matrix H]:\n", H)  # Afficher H pour débogage
        num_inliers = np.sum(masque) if masque is not None else 0
        print(f"INFO [Homographie]: Nombre d'inliers RANSAC: {num_inliers} / {len(correspondances)}")
    return H, masque


def creer_panoramique(img_gauche, img_droite, H):
    if H is None:
        print("INFO [Panoramique]: Homographie H est None, impossible de créer le panoramique.")
        return None
    if img_gauche is None or img_droite is None:
        print("INFO [Panoramique]: Une des images est None.")
        return None

    h_gauche, w_gauche = img_gauche.shape[:2]
    h_droite, w_droite = img_droite.shape[:2]

    # Étape 1: Transformer les coins de l'image droite pour déterminer la taille du canevas final
    pts_corners_droite = np.float32(
        [[0, 0], [0, h_droite - 1], [w_droite - 1, h_droite - 1], [w_droite - 1, 0]]).reshape(-1, 1, 2)
    if pts_corners_droite is None or H is None:  # Vérification supplémentaire
        print("ERREUR [Panoramique]: Erreur avec les coins ou H avant perspectiveTransform.")
        return None
    try:
        pts_corners_droite_transformes = cv.perspectiveTransform(pts_corners_droite, H)
    except cv.error as e:
        print(f"ERREUR [Panoramique] cv.error during perspectiveTransform: {e}")
        print("H_matrix:\n", H)
        print("pts_corners_droite:\n", pts_corners_droite)
        return None

    if pts_corners_droite_transformes is None:
        print("ERREUR [Panoramique]: pts_corners_droite_transformes est None après perspectiveTransform.")
        return None

    # Coordonnées de tous les points qui définiront le canevas (coins img_gauche + coins img_droite transformés)
    # np.concatenate prend un tuple d'arrays
    pts_cadrage = np.concatenate(
        (np.float32([[0, 0], [0, h_gauche - 1], [w_gauche - 1, h_gauche - 1], [w_gauche - 1, 0]]).reshape(-1, 1, 2),
         pts_corners_droite_transformes), axis=0)

    # Trouver les min/max pour x et y pour le cadrage
    x_min, y_min = np.int32(pts_cadrage.min(axis=0).ravel() - 0.5)  # -0.5 pour arrondir vers le bas
    x_max, y_max = np.int32(pts_cadrage.max(axis=0).ravel() + 0.5)  # +0.5 pour arrondir vers le haut

    # Translation pour ramener x_min et y_min à 0 (si négatifs)
    # C'est important pour que toute l'image soit visible sur le canevas
    mat_translation = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]], dtype=np.float32)

    # Appliquer cette translation à la matrice d'homographie originale
    # Ainsi, l'image droite sera transformée directement aux bonnes coordonnées sur le canevas final
    H_finale_pour_droite = mat_translation @ H  # Multiplication matricielle

    # Taille du canevas final
    largeur_pano = x_max - x_min
    hauteur_pano = y_max - y_min

    print(
        f"INFO [Panoramique]: Taille canevas final: {largeur_pano}x{hauteur_pano}. Décalage appliqué: x={-x_min}, y={-y_min}")

    # Étape 2: Transformer (warper) l'image droite sur le canevas final
    img_droite_transformee = cv.warpPerspective(img_droite, H_finale_pour_droite, (largeur_pano, hauteur_pano))

    # Étape 3: Créer le panoramique et y placer l'image gauche (translatée aussi)
    # Le canevas est initialisé avec l'image droite transformée (qui pourrait être noire si H est mauvais)
    panoramique = img_droite_transformee.copy()

    # Placer l'image gauche, en tenant compte de la translation
    # img_gauche_transformee_pour_pano = cv.warpPerspective(img_gauche, mat_translation, (largeur_pano, hauteur_pano))
    # Cette ligne n'est correcte que si img_gauche doit être 'warped'. Ici on la place juste.

    # Le plus simple: img_gauche est placée à sa position translatée sur le canevas.
    # panoramique[int(-y_min) : int(-y_min)+h_gauche, int(-x_min) : int(-x_min)+w_gauche] = img_gauche
    # Cette méthode simple d'affectation directe peut poser problème si les régions se chevauchent
    # et que l'une est complètement noire à cause de warpPerspective.

    # Méthode de combinaison plus sûre :
    # Créer une image gauche translatée sur une toile de la taille du pano
    img_gauche_placee = np.zeros_like(panoramique)
    img_gauche_placee[int(-y_min):int(-y_min) + h_gauche, int(-x_min):int(-x_min) + w_gauche] = img_gauche

    # Utiliser un masque pour combiner : là où l'image gauche a du contenu, on la prend.
    # Sinon, on prend l'image droite transformée.
    masque_gauche = cv.cvtColor(img_gauche_placee, cv.COLOR_BGR2GRAY)
    _, masque_gauche = cv.threshold(masque_gauche, 0, 255, cv.THRESH_BINARY_INV)  # Inversé : noir où il y a img_gauche

    panoramique = cv.bitwise_and(panoramique, panoramique, mask=masque_gauche)  # Efface la zone de img_gauche dans pano
    panoramique = cv.add(panoramique, img_gauche_placee)  # Ajoute img_gauche (qui a 0 où elle n'est pas)

    print("INFO [Panoramique]: Panoramique combiné.")
    return panoramique


def main():
    image1_path = "./IMG_1_reduced.jpg"
    image2_path = "./IMG_2_reduced.jpg"
    num_keypoints_desired = 200
    compute_descriptors_flag = True
    alpha_threshold_value = 4.0
    ransac_reproj_threshold = 5.0

    print("INFO: Chargement Image 1...")
    img1, gray1 = load_gray_image(image1_path)
    if img1 is None: print(f"ERREUR: Image 1 ({image1_path}) non chargée. Arrêt."); return
    print("INFO: Chargement Image 2...")
    img2, gray2 = load_gray_image(image2_path)

    cv.namedWindow("Image 1 Original");
    display_image(img1, "Image 1 Original")
    if img2 is not None: cv.namedWindow("Image 2 Original"); display_image(img2, "Image 2 Original")

    resultats_globaux = {}
    extracteur_pour_gftt = "SIFT"
    detecteurs_config = {
        "GFTT-SIFT": {"detecteur": "GFTT", "extracteur": extracteur_pour_gftt},
        "SIFT": {"detecteur": "SIFT", "extracteur": "SIFT"},
        "ORB": {"detecteur": "ORB", "extracteur": "ORB"}
    }
    normes_pour_extracteur = {
        "SIFT": ("NORM_L2", cv.NORM_L2),
        "ORB": ("NORM_HAMMING", cv.NORM_HAMMING)
    }

    for nom_methode, config in detecteurs_config.items():
        print(f"\n--- Traitement avec {nom_methode} ---")
        resultats_globaux[nom_methode] = {'kp1': [], 'desc1': None, 'kp2': [], 'desc2': None, 'H': None, 'matches': []}

        if gray1 is not None:
            resultats_globaux[nom_methode]['kp1'] = feature_detector(config["detecteur"], gray1.copy(),
                                                                     num_keypoints_desired)
        if gray2 is not None:
            resultats_globaux[nom_methode]['kp2'] = feature_detector(config["detecteur"], gray2.copy(),
                                                                     num_keypoints_desired)

        if compute_descriptors_flag:
            if gray1 is not None and resultats_globaux[nom_methode]['kp1']:
                resultats_globaux[nom_methode]['kp1'], resultats_globaux[nom_methode]['desc1'] = \
                    feature_extractor(config["extracteur"], gray1.copy(), resultats_globaux[nom_methode]['kp1'])
            if gray2 is not None and resultats_globaux[nom_methode]['kp2']:
                resultats_globaux[nom_methode]['kp2'], resultats_globaux[nom_methode]['desc2'] = \
                    feature_extractor(config["extracteur"], gray2.copy(), resultats_globaux[nom_methode]['kp2'])

        if img1 is not None and img2 is not None and compute_descriptors_flag and \
                resultats_globaux[nom_methode]['desc1'] is not None and resultats_globaux[nom_methode][
            'desc2'] is not None:

            norm_name_str, actual_norm_cv = normes_pour_extracteur[config["extracteur"]]
            # print(f"INFO [{nom_methode}]: Appariement avec norme {norm_name_str} (Alpha: {alpha_threshold_value})")

            try:
                bf = cv.BFMatcher(normType=actual_norm_cv, crossCheck=True)
                matches = bf.match(resultats_globaux[nom_methode]['desc1'], resultats_globaux[nom_methode]['desc2'])
                if not matches: print(
                    f"INFO [{nom_methode}]: Aucune correspondance initiale (crossCheck=True)."); continue
                matches = sorted(matches, key=lambda x: x.distance)
                min_dist_cc = matches[0].distance
                threshold_cc = alpha_threshold_value * min_dist_cc
                good_matches_alpha = [m for m in matches if m.distance <= (threshold_cc + 1e-9)]
                resultats_globaux[nom_methode]['matches'] = good_matches_alpha

                if good_matches_alpha:
                    kp1_draw = list(resultats_globaux[nom_methode]['kp1'])
                    kp2_draw = list(resultats_globaux[nom_methode]['kp2'])
                    # ... (code d'affichage des correspondances) ...

                if good_matches_alpha:
                    H_calc, _ = calculer_homographie(good_matches_alpha,
                                                     resultats_globaux[nom_methode]['kp1'],
                                                     resultats_globaux[nom_methode]['kp2'],
                                                     cv.RANSAC,
                                                     ransac_reproj_threshold)
                    resultats_globaux[nom_methode]['H'] = H_calc
                    if H_calc is not None:
                        # print(f"INFO [{nom_methode}]: Homographie H calculée.")
                        panorama_img = creer_panoramique(img1, img2, H_calc)
                        if panorama_img is not None:
                            cv.namedWindow(f"Panoramique - {nom_methode}");
                            display_image(panorama_img, f"Panoramique - {nom_methode}")
                else:
                    print(f"INFO [{nom_methode}]: Pas de bonnes correspondances pour homographie/panoramique.")
            except cv.error as e:
                print(f"ERREUR [{nom_methode}] pendant appariement/homographie (norme {norm_name_str}): {e}")
        # ... (autres else pour sauter les étapes) ...

    key = 0
    print("\nAppuyez sur 'q' ou Échap pour quitter.")
    while key != ESC_KEY and key != Q_KEY: key = cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2 as cv
import sys  # Importation standard, bien que sys ne soit pas explicitement utilisé ici.
import numpy as np

# Définition des codes pour les touches du clavier pour la gestion des fenêtres OpenCV
ESC_KEY = 27
Q_KEY = 113

# Seuil minimum de correspondances nécessaires pour tenter de calculer une homographie.
# En dessous de ce seuil, on considère que l'homographie ne serait pas fiable.
MIN_MATCH_COUNT_FOR_HOMOGRAPHY = 10


def load_gray_image(path):
    """
    Charge une image à partir d'un chemin, la convertit en niveaux de gris.
    Retourne l'image couleur originale et sa version en niveaux de gris.
    Gère le cas où l'image ne peut pas être lue.
    """
    if (path != None):
        img = cv.imread(path)  # Lecture de l'image avec OpenCV
        if img is None:
            print(f"AVERTISSEMENT: Impossible de lire l'image : {path}")
            return None, None
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
    else:
        img = None;
        gray = None
    return img, gray


def display_image(img, image_window_name):
    """Affiche une image dans une fenêtre OpenCV si l'image est valide."""
    if img is not None: cv.imshow(image_window_name, img)


def feature_detector(type_detector, gray_img, nb_kp_desired):
    """
    Détecte les points d'intérêt dans une image en niveaux de gris en utilisant l'algorithme spécifié.
    Prend en charge GFTT, SIFT, ORB, et AKAZE.
    Effectue un prétraitement de l'image pour s'assurer qu'elle est au format uint8.
    """
    kp = []  # Initialisation de la liste des points clés
    if gray_img is None: print(f"INFO [{type_detector}]: Image grise non fournie pour détection."); return kp

    # S'assurer que l'image est au format uint8, requis par de nombreux détecteurs OpenCV
    if gray_img.dtype != np.uint8:
        if gray_img.max() <= 1.0 and (gray_img.dtype == np.float32 or gray_img.dtype == np.float64):
            gray_img = np.array(gray_img * 255, dtype=np.uint8)
        else:  # Normalisation pour les autres cas (ex: uint16, int32)
            gray_img = cv.normalize(gray_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if len(gray_img.shape) == 3: gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2GRAY)  # S'assurer qu'elle est bien grise

    # Sélection du détecteur en fonction du type demandé
    if type_detector == "GFTT":  # Good Features To Track - Détecteur de coins de Shi-Tomasi
        params = {'maxCorners': nb_kp_desired if nb_kp_desired else 100, 'qualityLevel': 0.01, 'minDistance': 10}
        corners = cv.goodFeaturesToTrack(gray_img, **params)  # Détection directe des coins
        if corners is not None: kp = [cv.KeyPoint(x=c[0, 0], y=c[0, 1], size=20) for c in
                                      corners]  # Conversion en objets KeyPoint
    elif type_detector == "SIFT":  # Scale-Invariant Feature Transform
        try:
            sift = cv.SIFT_create(nfeatures=(nb_kp_desired if nb_kp_desired else 0))  # 0 = tous les points
            kp_detected = sift.detect(gray_img, None)
            if kp_detected: kp = list(kp_detected)
        except Exception as e:  # SIFT peut ne pas être dispo si opencv-contrib n'est pas installé
            print(f"ERREUR [Détecteur SIFT]: {e}. opencv-contrib-python requis?");
            kp = []
    elif type_detector == "ORB":  # Oriented FAST and Rotated BRIEF
        try:
            orb = cv.ORB_create(nfeatures=(nb_kp_desired if nb_kp_desired else 500))
            kp_detected = orb.detect(gray_img, None)
            if kp_detected: kp = list(kp_detected)
        except Exception as e:
            print(f"ERREUR [Détecteur ORB]: {e}");
            kp = []
    elif type_detector == "AKAZE":  # Accelerated KAZE
        try:
            akaze = cv.AKAZE_create()  # Utilise les paramètres par défaut
            kp_detected = akaze.detect(gray_img, None)
            if kp_detected: kp = list(kp_detected)
        except Exception as e:
            print(f"ERREUR [Détecteur AKAZE]: {e}");
            kp = []
    else:
        print(f"AVERTISSEMENT: Détecteur '{type_detector}' inconnu.");
        kp = []
    # print(f"INFO [{type_detector}]: {len(kp)} points détectés.")
    return kp


def feature_extractor(extractor_type, gray_img, keypoints):
    """
    Calcule les descripteurs pour une liste de points d'intérêt donnée.
    Supporte SIFT, ORB, et AKAZE comme extracteurs.
    Les points clés peuvent être mis à jour (certains peuvent être supprimés si aucun descripteur ne peut être calculé).
    """
    kp_updated = keypoints;  # Par défaut, on garde les mêmes keypoints
    desc = None
    if gray_img is None: print(
        f"INFO [{extractor_type}]: Image grise non fournie pour l'extraction."); return kp_updated, desc
    if not keypoints: print(f"INFO [{extractor_type}]: Pas de points clés pour l'extraction."); return kp_updated, desc

    # Prétraitement de l'image (similaire au détecteur)
    if gray_img.dtype != np.uint8:
        if gray_img.max() <= 1.0 and (gray_img.dtype == np.float32 or gray_img.dtype == np.float64):
            gray_img = np.array(gray_img * 255, dtype=np.uint8)
        else:
            gray_img = cv.normalize(gray_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # print(f"INFO [{extractor_type}]: Calcul des descripteurs pour {len(keypoints)} points...")
    if extractor_type == "SIFT":
        try:
            sift = cv.SIFT_create()  # On recrée un objet SIFT pour le calcul des descripteurs
            kp_updated, desc = sift.compute(gray_img, keypoints)
        except Exception as e:
            print(f"ERREUR [Extracteur SIFT]: {e}")
    elif extractor_type == "ORB":
        try:
            orb = cv.ORB_create()
            kp_updated, desc = orb.compute(gray_img, keypoints)
        except Exception as e:
            print(f"ERREUR [Extracteur ORB]: {e}")
    elif extractor_type == "AKAZE":
        try:
            akaze = cv.AKAZE_create()  # AKAZE est aussi un extracteur
            kp_updated, desc = akaze.compute(gray_img, keypoints)
        except Exception as e:
            print(f"ERREUR [Extracteur AKAZE]: {e}")
    else:
        print(f"AVERTISSEMENT: Extracteur '{extractor_type}' inconnu.")

    if kp_updated is None: kp_updated = []  # S'assurer que kp_updated est toujours une liste
    # if desc is not None: print(f"INFO [{extractor_type}]: {desc.shape[0]} descripteurs calculés.")
    # else: print(f"INFO [{extractor_type}]: Aucun descripteur calculé.")
    return kp_updated, desc


def calculer_homographie(correspondances, kp1, kp2, methode_ransac, seuil_reproj):
    """
    Calcule la matrice d'homographie H entre deux ensembles de points clés appariés.
    Utilise l'algorithme RANSAC pour une estimation robuste.
    H transforme les points de l'image associée à kp2 vers le plan de l'image associée à kp1.
    """
    # Vérifications initiales de validité des entrées
    if not isinstance(kp1, (list, tuple)) or not isinstance(kp2, (list, tuple)): return None, None
    if any(not isinstance(k, cv.KeyPoint) for k_list in (kp1, kp2) for k in k_list if k_list): return None, None

    if not correspondances or len(correspondances) < MIN_MATCH_COUNT_FOR_HOMOGRAPHY:
        # print(f"INFO [Homographie]: Pas assez de correspondances ({len(correspondances)}) pour calculer (min: {MIN_MATCH_COUNT_FOR_HOMOGRAPHY}).")
        return None, None  # Pas assez de correspondances pour une homographie fiable

    try:
        # S'assurer que les listes de keypoints ne sont pas vides et que les indices sont valides
        if not kp1 or not kp2: return None, None

        # Extraction des coordonnées des points appariés
        # pts_source (de l'image 2) et pts_destination (de l'image 1)
        pts_source = np.float32([kp2[m.trainIdx].pt for m in correspondances if m.trainIdx < len(kp2)]).reshape(-1, 1,
                                                                                                                2)
        pts_destination = np.float32([kp1[m.queryIdx].pt for m in correspondances if m.queryIdx < len(kp1)]).reshape(-1,
                                                                                                                     1,
                                                                                                                     2)

        # Vérifier si on a toujours assez de points après filtrage potentiel des indices
        if pts_source.shape[0] < MIN_MATCH_COUNT_FOR_HOMOGRAPHY or pts_destination.shape[
            0] < MIN_MATCH_COUNT_FOR_HOMOGRAPHY or pts_source.shape[0] != pts_destination.shape[0]:
            return None, None
    except Exception as e:  # Gérer les erreurs d'index ou d'attribut
        print(f"ERREUR [Homographie] préparation des points: {e}");
        return None, None

    # Calcul de l'homographie avec RANSAC pour la robustesse aux outliers
    H, masque = cv.findHomography(pts_source, pts_destination, methode_ransac, seuil_reproj)

    if H is None:
        print("INFO [Homographie]: findHomography a retourné None (pas de consensus RANSAC trouvé).")
    # else:
    #     # Pour débogage ou information, on pourrait afficher H et le nombre d'inliers
    #     print("INFO [Homographie]: Matrice d'homographie calculée.")
    #     print("DEBUG [Homographie Matrix H]:\n", H)
    #     num_inliers = np.sum(masque) if masque is not None else 0
    #     print(f"INFO [Homographie]: Nombre d'inliers RANSAC: {num_inliers} / {len(correspondances)}")
    return H, masque


def creer_panoramique(img_gauche, img_droite, H):
    """
    Crée une image panoramique en transformant img_droite pour l'aligner avec img_gauche,
    en utilisant la matrice d'homographie H.
    La taille du canevas est calculée dynamiquement pour englober les deux images.
    """
    if H is None: print("INFO [Panoramique]: H est None."); return None
    if img_gauche is None or img_droite is None: print("INFO [Panoramique]: Image(s) None."); return None

    h_gauche, w_gauche = img_gauche.shape[:2];
    h_droite, w_droite = img_droite.shape[:2]

    # 1. Calculer où les coins de l'image droite atterrissent après transformation par H.
    # Ceci nous aidera à définir la taille du canevas final du panoramique.
    corners_droite_orig = np.float32(
        [[0, 0], [0, h_droite - 1], [w_droite - 1, h_droite - 1], [w_droite - 1, 0]]).reshape(-1, 1, 2)
    try:
        pts_corners_droite_transformes = cv.perspectiveTransform(corners_droite_orig, H)
    except cv.error as e:
        print(f"ERREUR [Panoramique] perspectiveTransform: {e}");
        return None
    if pts_corners_droite_transformes is None: print(
        "ERREUR [Panoramique]: perspectiveTransform a échoué pour les coins."); return None

    # 2. Déterminer le cadre (bounding box) qui englobe l'image gauche et l'image droite transformée.
    corners_gauche = np.float32([[0, 0], [0, h_gauche - 1], [w_gauche - 1, h_gauche - 1], [w_gauche - 1, 0]]).reshape(
        -1, 1, 2)
    tous_les_points = np.concatenate((corners_gauche, pts_corners_droite_transformes), axis=0)

    # Coordonnées min/max pour définir la taille du canevas
    x_min, y_min = np.floor(tous_les_points.min(axis=0).ravel()).astype(np.int32)
    x_max, y_max = np.ceil(tous_les_points.max(axis=0).ravel()).astype(np.int32)

    # 3. Calculer un décalage si les coordonnées min sont négatives (pour tout ramener dans le positif)
    decalage_x = -x_min;
    decalage_y = -y_min

    # Matrice de translation pour ajuster la position sur le canevas final
    mat_translation = np.array([[1, 0, decalage_x], [0, 1, decalage_y], [0, 0, 1]], dtype=np.float32)

    # Nouvelle homographie pour transformer l'image droite directement sur le canevas final
    H_finale_pour_droite = mat_translation @ H

    # Taille du canevas final
    largeur_pano = x_max - x_min;
    hauteur_pano = y_max - y_min

    if largeur_pano <= 0 or hauteur_pano <= 0: print(
        f"ERREUR [Panoramique]: Dimensions canevas ({largeur_pano}x{hauteur_pano}) invalides."); return None
    # print(f"INFO [Panoramique]: Taille canevas: {largeur_pano}x{hauteur_pano}. Décalage: x={decalage_x}, y={decalage_y}")

    # 4. Transformer (warper) l'image droite sur ce canevas final.
    img_droite_transformee = cv.warpPerspective(img_droite, H_finale_pour_droite, (largeur_pano, hauteur_pano))

    # 5. Combiner les deux images sur le canevas.
    # On commence avec l'image droite transformée comme base.
    panoramique = img_droite_transformee.copy()

    # On crée une image temporaire pour placer l'image gauche à sa position translatée.
    img_gauche_placee = np.zeros_like(panoramique)
    y_dest_gauche = decalage_y;
    x_dest_gauche = decalage_x  # Coords du coin sup-gauche de img_gauche sur le pano

    # Tranche de destination sur le panoramique et portion de img_gauche à copier
    slice_pano_y_start = y_dest_gauche;
    slice_pano_y_end = y_dest_gauche + h_gauche
    slice_pano_x_start = x_dest_gauche;
    slice_pano_x_end = x_dest_gauche + w_gauche

    h_copie = min(slice_pano_y_end, hauteur_pano) - slice_pano_y_start
    w_copie = min(slice_pano_x_end, largeur_pano) - slice_pano_x_start

    if h_copie > 0 and w_copie > 0:
        # S'assurer qu'on ne découpe pas plus que la taille originale de img_gauche
        img_gauche_a_copier = img_gauche[0:min(h_copie, h_gauche), 0:min(w_copie, w_gauche)]
        # S'assurer que les dimensions de la source et de la destination de la tranche correspondent
        if img_gauche_a_copier.shape[0] == h_copie and img_gauche_a_copier.shape[1] == w_copie:
            img_gauche_placee[slice_pano_y_start: slice_pano_y_start + h_copie,
            slice_pano_x_start: slice_pano_x_start + w_copie] = img_gauche_a_copier

    # Combinaison: on prend les pixels de img_gauche_placee là où ils existent,
    # et ceux de img_droite_transformee (qui est dans 'panoramique') ailleurs.
    masque_gauche_contenu = cv.cvtColor(img_gauche_placee, cv.COLOR_BGR2GRAY)
    _, masque_gauche_contenu = cv.threshold(masque_gauche_contenu, 0, 255,
                                            cv.THRESH_BINARY)  # Masque où img_gauche a du contenu
    masque_gauche_inv = cv.bitwise_not(masque_gauche_contenu)  # Inversé: où img_gauche N'A PAS de contenu

    panoramique_partie_droite = cv.bitwise_and(panoramique, panoramique,
                                               mask=masque_gauche_inv)  # Garde img_droite là où img_gauche n'est pas
    panoramique_final = cv.add(panoramique_partie_droite, img_gauche_placee)  # Ajoute img_gauche

    # print("INFO [Panoramique]: Panoramique combiné.")
    return panoramique_final


def main():
    """
    Fonction principale qui orchestre le processus de création de panoramique.
    Les paramètres (chemins d'images, choix d'algorithmes) sont définis en dur ici.
    """
    # Définition des paramètres directement dans le code pour simplifier les tests
    image1_path = "./2.jpeg"  # Image de référence (gauche)
    image2_path = "./1.jpeg"  # Image à transformer (droite)

    num_keypoints_desired = 300  # Nombre de points d'intérêt souhaité (si applicable)
    compute_descriptors_flag = True  # Activer/Désactiver le calcul des descripteurs et la suite
    alpha_threshold_value = 4.0  # Coefficient pour le filtrage des correspondances
    ransac_reproj_threshold = 5.0  # Seuil de reprojection pour RANSAC (calcul d'homographie)

    print("INFO: Chargement Image 1...")
    img1, gray1 = load_gray_image(image1_path)
    if img1 is None: print(f"ERREUR: Image 1 ({image1_path}) non chargée. Arrêt."); return

    img2, gray2 = None, None  # Initialisation
    if image2_path:  # On charge l'image 2 seulement si un chemin est fourni
        print("INFO: Chargement Image 2...")
        img2, gray2 = load_gray_image(image2_path)
        if img2 is None: print(
            f"AVERTISSEMENT: Image 2 ({image2_path}) non chargée. Le panoramique ne sera pas possible.")
    else:
        print("INFO: Pas de deuxième image fournie. Le panoramique ne sera pas possible.")

    # Affichage des images originales
    cv.namedWindow("Image 1 Original");
    display_image(img1, "Image 1 Original")
    if img2 is not None: cv.namedWindow("Image 2 Original"); display_image(img2, "Image 2 Original")

    # Dictionnaire pour stocker les résultats (kp, desc, H, matches) pour chaque méthode
    resultats_globaux = {}

    # Configuration des détecteurs/extracteurs à tester
    extracteur_pour_gftt = "SIFT"  # GFTT est un détecteur, on utilise SIFT pour décrire ses points
    detecteurs_config = {
        "GFTT-SIFT": {"detecteur": "GFTT", "extracteur": extracteur_pour_gftt},
        "SIFT": {"detecteur": "SIFT", "extracteur": "SIFT"},
        "ORB": {"detecteur": "ORB", "extracteur": "ORB"},
        "AKAZE": {"detecteur": "AKAZE", "extracteur": "AKAZE"}
    }
    # Normes de distance appropriées pour chaque type de descripteur
    normes_pour_extracteur = {
        "SIFT": ("NORM_L2", cv.NORM_L2),  # Descripteurs SIFT (flottants) -> NORM_L2 (Euclidienne)
        "ORB": ("NORM_HAMMING", cv.NORM_HAMMING),  # Descripteurs ORB (binaires) -> NORM_HAMMING
        "AKAZE": ("NORM_HAMMING", cv.NORM_HAMMING)  # Descripteurs AKAZE (MLDB par défaut, binaires) -> NORM_HAMMING
    }

    # Boucle principale: tester chaque configuration de détecteur/extracteur
    for nom_methode, config in detecteurs_config.items():
        print(f"\n--- Traitement avec {nom_methode} ---")
        resultats_globaux[nom_methode] = {'kp1': [], 'desc1': None, 'kp2': [], 'desc2': None, 'H': None, 'matches': []}

        # Détection des points d'intérêt
        if gray1 is not None:
            resultats_globaux[nom_methode]['kp1'] = feature_detector(config["detecteur"], gray1.copy(),
                                                                     num_keypoints_desired)
        if gray2 is not None:  # Uniquement si l'image 2 a été chargée
            resultats_globaux[nom_methode]['kp2'] = feature_detector(config["detecteur"], gray2.copy(),
                                                                     num_keypoints_desired)

        # Extraction des descripteurs (si activé et points clés trouvés)
        if compute_descriptors_flag:
            if gray1 is not None and resultats_globaux[nom_methode]['kp1']:
                resultats_globaux[nom_methode]['kp1'], resultats_globaux[nom_methode]['desc1'] = \
                    feature_extractor(config["extracteur"], gray1.copy(), resultats_globaux[nom_methode]['kp1'])
            if gray2 is not None and resultats_globaux[nom_methode]['kp2']:
                resultats_globaux[nom_methode]['kp2'], resultats_globaux[nom_methode]['desc2'] = \
                    feature_extractor(config["extracteur"], gray2.copy(), resultats_globaux[nom_methode]['kp2'])

        # Suite du traitement (appariement, homographie, panoramique) uniquement si les deux images et leurs descripteurs sont disponibles
        condition_pour_assemblage = (
                img1 is not None and
                img2 is not None and
                compute_descriptors_flag and
                resultats_globaux[nom_methode]['desc1'] is not None and
                resultats_globaux[nom_methode]['desc2'] is not None
        )

        if condition_pour_assemblage:
            # Choix de la norme de distance
            if config["extracteur"] not in normes_pour_extracteur:
                print(
                    f"AVERTISSEMENT [{nom_methode}]: Norme non définie pour {config['extracteur']}. Appariement sauté.")
                continue
            norm_name_str, actual_norm_cv = normes_pour_extracteur[config["extracteur"]]

            try:
                # Appariement par force brute avec crossCheck pour un premier filtrage robuste
                bf = cv.BFMatcher(normType=actual_norm_cv, crossCheck=True)
                matches = bf.match(resultats_globaux[nom_methode]['desc1'], resultats_globaux[nom_methode]['desc2'])
                if not matches: print(f"INFO [{nom_methode}]: Aucune correspondance (crossCheck=True)."); continue

                # Tri des correspondances par distance
                matches = sorted(matches, key=lambda x: x.distance)
                if not matches: continue  # Sécurité, devrait être déjà couvert

                # Filtrage supplémentaire basé sur alpha * minDist
                min_dist_cc = matches[0].distance
                threshold_cc = alpha_threshold_value * min_dist_cc
                good_matches_alpha = [m for m in matches if
                                      m.distance <= (threshold_cc + 1e-9)]  # Epsilon pour la comparaison flottante
                resultats_globaux[nom_methode]['matches'] = good_matches_alpha
                print(
                    f"INFO [{nom_methode}]: {len(matches)} (crossCheck) -> {len(good_matches_alpha)} (alpha*minDist). MinDist_cc: {min_dist_cc:.2f}")

                # Affichage des correspondances (Question 4)
                if good_matches_alpha:
                    kp1_draw = list(resultats_globaux[nom_methode]['kp1'])
                    kp2_draw = list(resultats_globaux[nom_methode]['kp2'])
                    if kp1_draw and kp2_draw:  # S'assurer que les listes ne sont pas vides
                        matches_a_dessiner = good_matches_alpha[:25]  # Limiter à 25 pour la lisibilité
                        if matches_a_dessiner:  # S'assurer qu'il y a bien des matches à dessiner
                            img_matches_display = cv.drawMatches(img1.copy(), kp1_draw, img2.copy(), kp2_draw,
                                                                 matches_a_dessiner, None,
                                                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                            cv.namedWindow(f"Correspondances {nom_methode}");
                            display_image(img_matches_display, f"Correspondances {nom_methode}")

                # Calcul de l'homographie (Question 5)
                if good_matches_alpha:
                    H_calc, _ = calculer_homographie(good_matches_alpha,
                                                     resultats_globaux[nom_methode]['kp1'],
                                                     resultats_globaux[nom_methode]['kp2'],
                                                     cv.RANSAC,  # Méthode RANSAC pour la robustesse
                                                     ransac_reproj_threshold)
                    resultats_globaux[nom_methode]['H'] = H_calc

                    # Création du Panoramique (Question 6)
                    if H_calc is not None:
                        panorama_img = creer_panoramique(img1, img2, H_calc)
                        if panorama_img is not None:
                            cv.namedWindow(f"Panoramique - {nom_methode}");
                            display_image(panorama_img, f"Panoramique - {nom_methode}")
                else:
                    print(f"INFO [{nom_methode}]: Pas de bonnes correspondances pour homographie/panoramique.")
            except cv.error as e:  # Gérer les erreurs OpenCV spécifiques (ex: taille/type de descripteurs incompatibles avec la norme)
                print(f"ERREUR OpenCV [{nom_methode}] pendant appariement/homographie (norme {norm_name_str}): {e}")
            except Exception as e_gen:  # Gérer les autres erreurs potentielles
                print(f"ERREUR Générique [{nom_methode}] pendant appariement/homographie: {e_gen}")
        else:  # Expliquer pourquoi les étapes d'assemblage sont sautées
            if img1 is None or img2 is None:
                print(f"INFO [{nom_methode}]: Au moins une image manquante, opérations d'assemblage sautées.")
            elif not compute_descriptors_flag:
                print(f"INFO [{nom_methode}]: Calcul des descripteurs désactivé, assemblage sauté.")
            elif not (resultats_globaux[nom_methode]['desc1'] is not None and resultats_globaux[nom_methode][
                'desc2'] is not None):
                print(f"INFO [{nom_methode}]: Descripteurs non disponibles pour les deux images, assemblage sauté.")

    # Boucle d'attente pour l'affichage des fenêtres
    key = 0
    print("\nAppuyez sur 'q' ou Échap dans une fenêtre OpenCV pour quitter.")
    while key != ESC_KEY and key != Q_KEY:
        key = cv.waitKey(0)  # Attente indéfinie d'une touche
    cv.destroyAllWindows()  # Fermeture de toutes les fenêtres OpenCV
    print("INFO: Toutes les fenêtres OpenCV ont été fermées.")


if __name__ == "__main__":
    main()
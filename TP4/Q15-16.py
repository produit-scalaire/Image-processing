import cv2
import numpy as np

# --- Paramètres Globaux et Constantes ---
VIDEO_FILENAME = 'video.avi'
M_FOR_AVG_BACKGROUND = 150
M_FOR_ROAD_STD_DEV = 70
STD_DEV_THRESHOLD_ROAD = 50
MORPH_KERNEL_SIZE_ROAD = 7

# Ajustements pour la sensibilité de détection
BG_SUBTRACTION_THRESHOLD = 38
MORPH_KERNEL_SIZE_CARS = 6
MORPH_DILATE_CARS = 1
MIN_CAR_AREA = 100
MAX_CAR_AREA = 4000

ROI_RECT = (150, 200, 200, 180)  # (x, y, width, height)

# Paramètres pour le comptage par franchissement de ligne horizontale (Q15)
COUNTING_LINE_Y_POS = int(ROI_RECT[1] + 20)  # Ligne de comptage positionnée à Y = ROI_y + 20
CROSSING_MARGIN_Y = 10  # Marge pour valider le franchissement
MAX_DIST_FOR_TRACKING_HEURISTIC = 100

# --- Variables Globales pour le compteur Q15 ---
g_total_cars_counted_q15 = 0
g_previous_frame_centroids_q15 = []
g_tracked_cars_q15 = {}  # Dictionnaire pour le suivi persistant des véhicules


# --- Chargement de toutes les trames de la vidéo ---
def q6_load_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    temp_frames_list = []
    while cap.isOpened():  # Boucle tant que la vidéo est ouverte et lit des trames
        ret, frame = cap.read()
        if not ret:  # Si aucune trame n'est lue (fin de la vidéo ou erreur)
            break
        temp_frames_list.append(frame)
    cap.release()
    video_frames_color_np = np.array(temp_frames_list)
    video_frames_gray_np = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in video_frames_color_np])
    return video_frames_color_np, video_frames_gray_np


# --- Calcul de l'image moyenne sur une série de trames ---
def q7_calculate_average_image(frames_slice_gray):
    return np.mean(frames_slice_gray, axis=0).astype(np.uint8)


# --- Génération du masque de la route basé sur la variance temporelle ---
def q9_generate_road_mask(frames_sequence_gray, m_for_std, threshold_val, kernel_s):
    std_dev_image_float = np.std(frames_sequence_gray[0:m_for_std], axis=0)
    std_dev_image_uint8 = cv2.normalize(std_dev_image_float, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, road_activity_mask = cv2.threshold(std_dev_image_uint8, threshold_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_s, kernel_s))
    road_mask_cleaned = cv2.morphologyEx(road_activity_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return road_mask_cleaned


# --- Création d'un masque binaire pour la Région d'Intérêt (ROI) ---
def q11_create_roi_mask(frame_shape, roi_rectangle):
    roi_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    x, y, w, h = roi_rectangle
    cv2.rectangle(roi_mask, (x, y), (x + w, y + h), 255, -1)
    return roi_mask


# --- Dessin de la ROI et de la ligne de comptage sur une trame ---
def q11_draw_roi_on_frame(frame_to_draw_on, roi_rectangle, draw_counting_line=False):
    x, y, w, h = roi_rectangle
    cv2.rectangle(frame_to_draw_on, (x, y), (x + w, y + h), (0, 255, 255), 2)  # ROI en jaune

    if draw_counting_line:
        # Ligne de comptage HORIZONTALE en bleu
        cv2.line(frame_to_draw_on, (x, COUNTING_LINE_Y_POS), (x + w, COUNTING_LINE_Y_POS), (255, 0, 0), 2)


# --- Traitement des contours pour extraire les véhicules valides et leurs centroïdes ---
def q14_process_contours_and_get_centroids(binary_mask_input, min_area, max_area):
    valid_contours = []
    valid_centroids = []
    mask_processed = binary_mask_input.copy()  # Utilisation directe du masque d'entrée

    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:  # Filtrage par aire pour ne garder que les objets de taille plausible
            valid_contours.append(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:  # Assurer que le dénominateur pour le calcul du centroïde n'est pas nul
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                valid_centroids.append((cx, cy))
    return valid_contours, valid_centroids


# --- Q15: Logique de comptage des véhicules par franchissement de ligne horizontale ---
# Cette fonction actualise le décompte total des véhicules.
# Elle se base sur le franchissement d'une ligne de détection horizontale virtuelle.
# Un mécanisme de suivi inter-trame est employé pour attribuer un ID unique à chaque véhicule détecté
# et s'assurer qu'il n'est comptabilisé qu'une seule fois.
def q15_update_line_crossing_count(current_centroids, previous_centroids_list, line_y_pos, margin_y, max_dist_heuristic,
                                   current_total_count, frame_idx_debug):
    # Utilisation d'une variable globale pour conserver l'état des véhicules suivis entre les appels successifs de la fonction.
    global g_tracked_cars_q15

    newly_counted_this_frame = 0  # Initialisation du compteur pour les véhicules franchissant la ligne dans la trame actuelle.

    # Dictionnaire temporaire pour construire la liste des véhicules suivis pour la trame suivante.
    next_tracked_cars = {}

    # Liste d'indices des centroïdes de la trame actuelle, initialement tous considérés comme non appariés.
    unmatched_current_centroids_indices = list(range(len(current_centroids)))

    # --- Phase 1: Tentative de ré-association des véhicules déjà sous suivi ---
    # Parcours des véhicules qui étaient suivis dans la trame précédente.
    for car_id, prev_data in list(g_tracked_cars_q15.items()):
        px_prev, py_prev = prev_data['centroid']  # Coordonnées du véhicule dans la trame N-1.
        counted_status = prev_data.get('counted', False)  # Statut de comptage du véhicule.

        best_match_curr_idx = -1  # Indice du meilleur centroïde courant correspondant.
        min_dist = max_dist_heuristic  # Distance maximale pour un appariement valide.

        # Recherche de la correspondance la plus proche pour ce véhicule parmi les centroïdes courants non encore affectés.
        for i_curr_idx_val_in_list, actual_curr_idx in enumerate(unmatched_current_centroids_indices):
            cx_curr, cy_curr = current_centroids[actual_curr_idx]
            dist = np.sqrt((cx_curr - px_prev) ** 2 + (cy_curr - py_prev) ** 2)  # Distance euclidienne.
            if dist < min_dist:  # Si une correspondance plus proche est trouvée.
                min_dist = dist
                best_match_curr_idx = actual_curr_idx
                best_match_list_idx = i_curr_idx_val_in_list  # Index dans la liste des non-appariés.

        if best_match_curr_idx != -1:  # Si un appariement a été trouvé.
            cx_curr, cy_curr = current_centroids[best_match_curr_idx]
            # Le véhicule est reporté dans la liste des véhicules suivis pour la prochaine trame, avec son statut de comptage.
            next_tracked_cars[car_id] = {'centroid': (cx_curr, cy_curr), 'counted': counted_status}
            # Le centroïde apparié est retiré de la liste des centroïdes en attente.
            unmatched_current_centroids_indices.pop(best_match_list_idx)

            if not counted_status:  # Si ce véhicule suivi n'avait pas encore été compté.
                # Condition de franchissement : le véhicule était au-dessus de la ligne (py_prev < line_y_pos)
                # et est maintenant sur ou en dessous de la ligne + marge (cy_curr >= (line_y_pos + margin_y)).
                if py_prev < line_y_pos and cy_curr >= (line_y_pos + margin_y):
                    newly_counted_this_frame += 1
                    next_tracked_cars[car_id]['counted'] = True  # Le véhicule est marqué comme compté.

    # --- Phase 2: Traitement des nouveaux centroïdes (non appariés à des véhicules existants) ---
    # Statut d'appariement pour les centroïdes de la trame précédente (utilisé pour les NOUVEAUX objets uniquement).
    previous_centroids_matched_status = [False] * len(previous_centroids_list)

    for curr_idx in unmatched_current_centroids_indices:  # Parcours des centroïdes courants non encore associés.
        cx_curr, cy_curr = current_centroids[curr_idx]

        # Vérification si ce nouveau centroïde est positionné pour un comptage (sous la ligne + marge).
        if cy_curr >= (line_y_pos + margin_y):
            best_prev_match_idx_for_new = -1  # Indice du meilleur "parent" potentiel dans la trame N-1.
            min_dist_for_new = max_dist_heuristic

            # Recherche d'un "parent" dans la liste des centroïdes de la trame N-1.
            # Cela permet de s'assurer d'une certaine continuité de mouvement avant de compter un nouvel objet.
            for idx_prev, (px_prev_glob, py_prev_glob) in enumerate(previous_centroids_list):
                if not previous_centroids_matched_status[
                    idx_prev]:  # Si ce "parent" n'a pas déjà été utilisé pour un autre nouveau.
                    # Le "parent" doit avoir été au-dessus de la ligne de comptage.
                    if py_prev_glob < line_y_pos:
                        dist = np.sqrt((cx_curr - px_prev_glob) ** 2 + (cy_curr - py_prev_glob) ** 2)
                        if dist < min_dist_for_new:  # Si une correspondance plus proche est trouvée.
                            min_dist_for_new = dist
                            best_prev_match_idx_for_new = idx_prev

            if best_prev_match_idx_for_new != -1:  # Si un "parent" valide est trouvé.
                previous_centroids_matched_status[
                    best_prev_match_idx_for_new] = True  # Marquer ce "parent" comme utilisé.

                # Création d'un nouvel ID pour ce véhicule et ajout à la liste des véhicules suivis.
                new_car_id = f"car_{frame_idx_debug}_{curr_idx}"  # ID simple basé sur l'index de trame et de centroïde.
                next_tracked_cars[new_car_id] = {'centroid': (cx_curr, cy_curr),
                                                 'counted': True}  # Compté dès sa validation.
                newly_counted_this_frame += 1

    # Actualisation du dictionnaire global des véhicules suivis.
    g_tracked_cars_q15 = next_tracked_cars

    return current_total_count + newly_counted_this_frame


# --- Exécution Principale ---
if __name__ == '__main__':
    video_frames_color_np, video_frames_gray_np = q6_load_all_frames(VIDEO_FILENAME)

    num_total_frames, H, W, _ = video_frames_color_np.shape
    g_total_cars_counted_q15 = 0
    g_previous_frame_centroids_q15 = []
    g_tracked_cars_q15 = {}

    m_bg_actual = min(num_total_frames, M_FOR_AVG_BACKGROUND)
    background_model_gray = q7_calculate_average_image(video_frames_gray_np[0:m_bg_actual])
    cv2.imshow('Modele de Fond', background_model_gray)  # Visualisation du fond

    m_road_actual = min(num_total_frames, M_FOR_ROAD_STD_DEV)
    road_mask = q9_generate_road_mask(video_frames_gray_np, m_road_actual, STD_DEV_THRESHOLD_ROAD,
                                      MORPH_KERNEL_SIZE_ROAD)

    roi_mask_for_counting = q11_create_roi_mask((H, W), ROI_RECT)
    start_frame_idx_processing = m_bg_actual

    for i in range(start_frame_idx_processing, num_total_frames):
        current_gray_frame = video_frames_gray_np[i]
        current_color_frame_display_main = video_frames_color_np[i].copy()

        # Traitement pour isoler les véhicules
        diff_frame = cv2.absdiff(current_gray_frame, background_model_gray)
        _, fg_mask_raw = cv2.threshold(diff_frame, BG_SUBTRACTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        objects_on_road_mask_q10 = cv2.bitwise_and(fg_mask_raw, fg_mask_raw, mask=road_mask)
        kernel_cars_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (MORPH_KERNEL_SIZE_CARS, MORPH_KERNEL_SIZE_CARS))
        cars_mask_processed_q10 = cv2.morphologyEx(objects_on_road_mask_q10, cv2.MORPH_CLOSE, kernel_cars_close,
                                                   iterations=1)
        if MORPH_DILATE_CARS > 0:
            kernel_cars_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_DILATE_CARS, MORPH_DILATE_CARS))
            cars_mask_processed_q10 = cv2.dilate(cars_mask_processed_q10, kernel_cars_dilate, iterations=1)
        cars_in_roi_mask_final = cv2.bitwise_and(cars_mask_processed_q10, cars_mask_processed_q10,
                                                 mask=roi_mask_for_counting)
        cv2.imshow('Masque pour Comptage (Voitures dans ROI)',
                   cars_in_roi_mask_final)  # Visualisation du masque de détection

        # Comptage par composantes connexes (Q12)
        num_labels_q12, _, stats_q12, _ = cv2.connectedComponentsWithStats(cars_in_roi_mask_final, connectivity=8)
        car_count_q12_per_frame = 0
        for label_idx in range(1, num_labels_q12):
            if MIN_CAR_AREA < stats_q12[label_idx, cv2.CC_STAT_AREA] < MAX_CAR_AREA:
                car_count_q12_per_frame += 1

        # Comptage par contours (Q14) et récupération des centroïdes pour Q15
        valid_contours_q14, current_centroids_for_q15 = q14_process_contours_and_get_centroids(
            cars_in_roi_mask_final, MIN_CAR_AREA, MAX_CAR_AREA)
        car_count_q14_per_frame = len(valid_contours_q14)

        # Mise à jour du compteur total par franchissement de ligne (Q15)
        g_total_cars_counted_q15 = q15_update_line_crossing_count(
            current_centroids_for_q15,
            g_previous_frame_centroids_q15,
            COUNTING_LINE_Y_POS,
            CROSSING_MARGIN_Y,
            MAX_DIST_FOR_TRACKING_HEURISTIC,
            g_total_cars_counted_q15,
            i)
        g_previous_frame_centroids_q15 = list(current_centroids_for_q15)  # Mise à jour pour la trame suivante

        # Visualisations sur la trame couleur
        q11_draw_roi_on_frame(current_color_frame_display_main, ROI_RECT, draw_counting_line=True)
        cv2.putText(current_color_frame_display_main, f"Q12 CC/Fr: {car_count_q12_per_frame}", (10, H - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        cv2.putText(current_color_frame_display_main, f"Q14 Cnt/Fr: {car_count_q14_per_frame}", (10, H - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(current_color_frame_display_main, f"Q15 Total: {g_total_cars_counted_q15}", (10, H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 180, 50), 2)
        cv2.drawContours(current_color_frame_display_main, valid_contours_q14, -1, (0, 255, 0),
                         2)  # Contours valides en vert

        for cx, cy in current_centroids_for_q15:  # Centroïdes détectés pour Q15 en magenta
            cv2.circle(current_color_frame_display_main, (cx, cy), 3, (255, 0, 255), -1)

        for car_id, data in g_tracked_cars_q15.items():  # Visualisation des véhicules suivis pour Q15
            cx_track, cy_track = data['centroid']
            color = (0, 255, 0) if data.get('counted') else (0, 0, 255)  # Vert si compté, Rouge sinon
            cv2.putText(current_color_frame_display_main, str(car_id)[-3:], (cx_track - 10, cy_track - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.circle(current_color_frame_display_main, (cx_track, cy_track), 5, color, 1)

        cv2.imshow('Q15 - Comptage Ligne Horizontale (+Q12/Q14)',
                   current_color_frame_display_main)  # Affichage principal

        key = cv2.waitKey(30) & 0xFF  # Attente de 30ms ou d'une touche
        if key == 27:  # Touche Echap pour quitter
            print("Arrêt demandé par l'utilisateur.")
            break
        elif key == ord('p'):  # Touche 'p' pour mettre en pause/reprendre
            print("Pause. Appuyez sur 'p' pour reprendre ou 'echap' pour quitter.")
            while True:
                key_pause = cv2.waitKey(0) & 0xFF
                if key_pause == ord('p'):
                    print("Reprise.")
                    break
                elif key_pause == 27:
                    print("Arrêt demandé par l'utilisateur pendant la pause.")
                    i = num_total_frames  # Forcer la sortie de la boucle principale
                    break
            if i == num_total_frames:  # Si arrêt demandé pendant la pause
                break

    # Affichage du résultat final et attente d'une touche si la vidéo s'est terminée normalement
    if i >= num_total_frames - 1 and not (cv2.waitKey(1) & 0xFF == 27 and i < num_total_frames - 1):
        print(f"Fin du traitement. Q15 Total Cars: {g_total_cars_counted_q15}. Appuyez sur une touche pour quitter.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("\n--- Fin du programme ---")

import cv2
import numpy as np

# --- Paramètres Globaux et Constantes (Tes valeurs) ---
VIDEO_FILENAME = 'video.avi'
M_FOR_AVG_BACKGROUND = 150
M_FOR_ROAD_STD_DEV = 70
STD_DEV_THRESHOLD_ROAD = 50
MORPH_KERNEL_SIZE_ROAD = 7

# Ajustements pour plus de sensibilité
BG_SUBTRACTION_THRESHOLD = 38  # Diminué (était 50)
MORPH_KERNEL_SIZE_CARS = 6  # Diminué (était 7, puis 5)
MORPH_DILATE_CARS = 1  # Ajout d'une légère dilatation pour fusionner les fragments
MIN_CAR_AREA = 100  # Fortement diminué (était 150-200)
MAX_CAR_AREA = 4000  # Augmenté (était 2500-3000)

ROI_RECT = (100, 150, 700, 180)  # (x, y, width, height)

# Pour Q15: Comptage total par franchissement de ligne HORIZONTALE
COUNTING_LINE_Y_POS = int(ROI_RECT[1] * 1.2)  # Environ 180 si ROI_RECT[1] = 150
CROSSING_MARGIN_Y = 10
MAX_DIST_FOR_TRACKING_HEURISTIC = 100  # Maintenu à une valeur permissive

# --- Variables Globales pour le compteur Q15 ---
g_total_cars_counted_q15 = 0
g_previous_frame_centroids_q15 = []
g_tracked_cars_q15 = {}  # Pour un suivi persistant


# --- Question 6: Chargement de toutes les frames ---
def q6_load_all_frames(video_path):
    print(f"\n--- Exécution Logique Q6: Chargement des frames de '{video_path}' ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur Q6: Impossible d'ouvrir la vidéo {video_path}")
        return None, None
    temp_frames_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        temp_frames_list.append(frame)
    cap.release()
    if not temp_frames_list:
        print("Erreur Q6: Aucune frame lue.")
        return None, None
    video_frames_color_np = np.array(temp_frames_list)
    video_frames_gray_np = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in video_frames_color_np])
    print(
        f"Q6: {len(video_frames_color_np)} frames stockées. Forme couleur: {video_frames_color_np.shape}, Forme gris: {video_frames_gray_np.shape}")
    return video_frames_color_np, video_frames_gray_np


# --- Question 7: Fonction pour calculer l'image moyenne ---
def q7_calculate_average_image(frames_slice_gray):
    if frames_slice_gray is None or len(frames_slice_gray) == 0: return None
    return np.mean(frames_slice_gray, axis=0).astype(np.uint8)


# --- Question 9: Fonction pour générer le masque de la route ---
def q9_generate_road_mask(frames_sequence_gray, m_for_std, threshold_val, kernel_s):
    print(f"\n--- Exécution Logique Q9: Génération Masque de Route (M={m_for_std}) ---")
    if frames_sequence_gray is None or len(frames_sequence_gray) < m_for_std or m_for_std < 2:
        print(f"Erreur Q9: Pas assez d'images pour M={m_for_std}.")
        return None
    std_dev_image_float = np.std(frames_sequence_gray[0:m_for_std], axis=0)
    std_dev_image_uint8 = cv2.normalize(std_dev_image_float, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, road_activity_mask = cv2.threshold(std_dev_image_uint8, threshold_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_s, kernel_s))
    road_mask_cleaned = cv2.morphologyEx(road_activity_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return road_mask_cleaned


# --- Question 11: Fonctions pour la ROI ---
def q11_create_roi_mask(frame_shape, roi_rectangle):
    roi_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    x, y, w, h = roi_rectangle
    cv2.rectangle(roi_mask, (x, y), (x + w, y + h), 255, -1)
    return roi_mask


def q11_draw_roi_on_frame(frame_to_draw_on, roi_rectangle, draw_counting_line=False):
    x, y, w, h = roi_rectangle
    cv2.rectangle(frame_to_draw_on, (x, y), (x + w, y + h), (0, 255, 255), 2)  # ROI en jaune

    if draw_counting_line:
        # Ligne de comptage HORIZONTALE en bleu
        cv2.line(frame_to_draw_on, (x, COUNTING_LINE_Y_POS), (x + w, COUNTING_LINE_Y_POS), (255, 0, 0), 2)
        # Ligne de marge pour visualiser (optionnel)
        # cv2.line(frame_to_draw_on, (x, COUNTING_LINE_Y_POS + CROSSING_MARGIN_Y), (x + w, COUNTING_LINE_Y_POS + CROSSING_MARGIN_Y), (0, 255, 255), 1)


# --- Question 14: Traitement des contours et récupération des centroïdes ---
def q14_process_contours_and_get_centroids(binary_mask_input, min_area, max_area):
    valid_contours = []
    valid_centroids = []

    # Pour une sensibilité maximale aux formes, on n'applique pas d'ouverture ici pour l'instant.
    # Si le bruit devient un problème majeur, on pourra réintroduire :
    # kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # mask_processed = cv2.morphologyEx(binary_mask_input, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask_processed = binary_mask_input.copy()  # Utiliser le masque tel quel pour l'instant

    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:  # Le filtre d'aire est crucial
            valid_contours.append(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                valid_centroids.append((cx, cy))
    return valid_contours, valid_centroids


# --- Q15: Logique de comptage total par franchissement de ligne HORIZONTALE (moins sensible) ---
def q15_update_line_crossing_count(current_centroids, previous_centroids_list, line_y_pos, margin_y, max_dist_heuristic,
                                   current_total_count, frame_idx_debug):
    global g_tracked_cars_q15

    newly_counted_this_frame = 0

    # Dictionnaire pour les voitures suivies dans la frame suivante
    next_tracked_cars = {}

    # Liste des centroïdes actuels qui n'ont pas encore été appariés
    unmatched_current_centroids_indices = list(range(len(current_centroids)))

    # Tenter de ré-associer les voitures déjà suivies
    for car_id, prev_data in list(g_tracked_cars_q15.items()):
        px_prev, py_prev = prev_data['centroid']
        counted_status = prev_data.get('counted', False)

        best_match_curr_idx = -1
        min_dist = max_dist_heuristic

        for i_curr_idx_val_in_list, actual_curr_idx in enumerate(unmatched_current_centroids_indices):
            cx_curr, cy_curr = current_centroids[actual_curr_idx]
            dist = np.sqrt((cx_curr - px_prev) ** 2 + (cy_curr - py_prev) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_match_curr_idx = actual_curr_idx
                best_match_list_idx = i_curr_idx_val_in_list  # Index dans unmatched_current_centroids_indices

        if best_match_curr_idx != -1:
            # Une voiture suivie a été retrouvée
            cx_curr, cy_curr = current_centroids[best_match_curr_idx]
            next_tracked_cars[car_id] = {'centroid': (cx_curr, cy_curr), 'counted': counted_status}
            unmatched_current_centroids_indices.pop(best_match_list_idx)  # Retirer de la liste des non-appariés

            if not counted_status:  # Si pas encore comptée
                # Vérifier le franchissement : était AU-DESSUS et est maintenant EN DESSOUS DE LA MARGE
                if py_prev < line_y_pos and cy_curr >= (line_y_pos + margin_y):
                    newly_counted_this_frame += 1
                    next_tracked_cars[car_id]['counted'] = True
                    # print(f"Frame {frame_idx_debug}: Voiture SUIVIE ID {car_id} comptée. ({px_prev},{py_prev}) -> ({cx_curr},{cy_curr}) vs Ligne {line_y_pos}+{margin_y}")

    # Gérer les nouveaux centroïdes (ceux restants dans unmatched_current_centroids_indices)
    # Pour ces nouveaux, on cherche un parent dans la liste globale des centroïdes de la frame précédente
    # pour établir la condition de franchissement initial.

    # Créer une liste de statuts pour les centroïdes de la frame précédente (pour l'appariement des NOUVEAUX objets)
    previous_centroids_matched_status = [False] * len(previous_centroids_list)

    for curr_idx in unmatched_current_centroids_indices:
        cx_curr, cy_curr = current_centroids[curr_idx]

        # Ce nouveau centroïde est-il en position d'être compté (en dessous de la ligne + marge) ?
        if cy_curr >= (line_y_pos + margin_y):
            best_prev_match_idx_for_new = -1
            min_dist_for_new = max_dist_heuristic

            for idx_prev, (px_prev_glob, py_prev_glob) in enumerate(previous_centroids_list):
                if not previous_centroids_matched_status[
                    idx_prev]:  # Si ce parent global n'a pas été utilisé pour un autre NOUVEL objet
                    # Le parent était-il AU-DESSUS de la ligne ?
                    if py_prev_glob < line_y_pos:
                        dist = np.sqrt((cx_curr - px_prev_glob) ** 2 + (cy_curr - py_prev_glob) ** 2)
                        if dist < min_dist_for_new:
                            min_dist_for_new = dist
                            best_prev_match_idx_for_new = idx_prev

            if best_prev_match_idx_for_new != -1:
                # Un parent adéquat a été trouvé pour ce nouveau centroïde
                previous_centroids_matched_status[best_prev_match_idx_for_new] = True  # Marquer ce parent comme utilisé

                new_car_id = f"car_{frame_idx_debug}_{curr_idx}"  # ID unique simple
                next_tracked_cars[new_car_id] = {'centroid': (cx_curr, cy_curr), 'counted': True}  # Compté directement
                newly_counted_this_frame += 1
                # print(f"Frame {frame_idx_debug}: NOUVELLE voiture ID {new_car_id} comptée. Prev@({previous_centroids_list[best_prev_match_idx_for_new]}) -> Curr@({cx_curr},{cy_curr}) vs Ligne {line_y_pos}+{margin_y}")
            # else:
            # print(f"Frame {frame_idx_debug}: Nouveau centroïde @({cx_curr},{cy_curr}) sous la ligne mais pas de parent valide trouvé.")
        # else:
        # print(f"Frame {frame_idx_debug}: Nouveau centroïde @({cx_curr},{cy_curr}) pas encore assez bas pour comptage.")

    g_tracked_cars_q15 = next_tracked_cars
    return current_total_count + newly_counted_this_frame


# --- Main: Exécution ---
if __name__ == '__main__':
    video_frames_color_np, video_frames_gray_np = q6_load_all_frames(VIDEO_FILENAME)

    if video_frames_color_np is not None:
        num_total_frames, H, W, _ = video_frames_color_np.shape
        g_total_cars_counted_q15 = 0
        g_previous_frame_centroids_q15 = []  # Stocke les centroïdes de la frame N-1
        g_tracked_cars_q15 = {}  # Réinitialiser le dictionnaire de suivi

        print(f"\n--- Préparation (Q7): Calcul du Modèle de Fond ---")
        m_bg_actual = min(num_total_frames, M_FOR_AVG_BACKGROUND)
        background_model_gray = q7_calculate_average_image(
            video_frames_gray_np[0:m_bg_actual]) if m_bg_actual > 0 else None
        if background_model_gray is not None:
            cv2.imshow('Modele de Fond', background_model_gray)
        else:
            print("Erreur: Modèle de fond non généré.")
            exit()

        m_road_actual = min(num_total_frames, M_FOR_ROAD_STD_DEV)
        road_mask = q9_generate_road_mask(video_frames_gray_np, m_road_actual, STD_DEV_THRESHOLD_ROAD,
                                          MORPH_KERNEL_SIZE_ROAD)
        if road_mask is None:
            print("Erreur: Masque de route non généré.")
            exit()

        print(f"\n--- Exécution Logique Q10 à Q15 ---")
        print(f"ROI: {ROI_RECT}, Filtre Taille: Min={MIN_CAR_AREA}, Max={MAX_CAR_AREA}")
        print(
            f"Q15 Ligne Y (horizontale): {COUNTING_LINE_Y_POS}, Marge Y: {CROSSING_MARGIN_Y}, Max Dist Track: {MAX_DIST_FOR_TRACKING_HEURISTIC}")

        roi_mask_for_counting = q11_create_roi_mask((H, W), ROI_RECT)
        start_frame_idx_processing = m_bg_actual  # Commencer après les frames de fond

        for i in range(start_frame_idx_processing, num_total_frames):
            current_gray_frame = video_frames_gray_np[i]
            current_color_frame_display_main = video_frames_color_np[i].copy()

            diff_frame = cv2.absdiff(current_gray_frame, background_model_gray)
            _, fg_mask_raw = cv2.threshold(diff_frame, BG_SUBTRACTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            objects_on_road_mask_q10 = cv2.bitwise_and(fg_mask_raw, fg_mask_raw, mask=road_mask)

            kernel_cars_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          (MORPH_KERNEL_SIZE_CARS, MORPH_KERNEL_SIZE_CARS))
            cars_mask_processed_q10 = cv2.morphologyEx(objects_on_road_mask_q10, cv2.MORPH_CLOSE, kernel_cars_close,
                                                       iterations=1)  # iterations=1 peut être suffisant

            if MORPH_DILATE_CARS > 0:
                kernel_cars_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                               (MORPH_DILATE_CARS, MORPH_DILATE_CARS))
                cars_mask_processed_q10 = cv2.dilate(cars_mask_processed_q10, kernel_cars_dilate, iterations=1)

            cars_in_roi_mask_final = cv2.bitwise_and(cars_mask_processed_q10, cars_mask_processed_q10,
                                                     mask=roi_mask_for_counting)
            cv2.imshow('Masque pour Comptage (Voitures dans ROI)', cars_in_roi_mask_final)

            num_labels_q12, _, stats_q12, _ = cv2.connectedComponentsWithStats(cars_in_roi_mask_final, connectivity=8)
            car_count_q12_per_frame = 0
            for label_idx in range(1, num_labels_q12):  # Ignorer le label 0 (fond)
                if MIN_CAR_AREA < stats_q12[label_idx, cv2.CC_STAT_AREA] < MAX_CAR_AREA:
                    car_count_q12_per_frame += 1

            valid_contours_q14, current_centroids_for_q15 = q14_process_contours_and_get_centroids(
                cars_in_roi_mask_final, MIN_CAR_AREA, MAX_CAR_AREA)
            car_count_q14_per_frame = len(valid_contours_q14)

            g_total_cars_counted_q15 = q15_update_line_crossing_count(
                current_centroids_for_q15,
                g_previous_frame_centroids_q15,  # Passer la liste des centroïdes de la frame N-1
                COUNTING_LINE_Y_POS,
                CROSSING_MARGIN_Y,
                MAX_DIST_FOR_TRACKING_HEURISTIC,
                g_total_cars_counted_q15,
                i)

            # Mettre à jour pour la prochaine itération
            g_previous_frame_centroids_q15 = list(current_centroids_for_q15)

            q11_draw_roi_on_frame(current_color_frame_display_main, ROI_RECT, draw_counting_line=True)
            cv2.putText(current_color_frame_display_main, f"Q12 CC/Fr: {car_count_q12_per_frame}", (10, H - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(current_color_frame_display_main, f"Q14 Cnt/Fr: {car_count_q14_per_frame}", (10, H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(current_color_frame_display_main, f"Q15 Total: {g_total_cars_counted_q15}", (10, H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 180, 50), 2)
            cv2.drawContours(current_color_frame_display_main, valid_contours_q14, -1, (0, 255, 0), 2)

            for cx, cy in current_centroids_for_q15:
                cv2.circle(current_color_frame_display_main, (cx, cy), 3, (255, 0, 255), -1)  # Centroïdes en magenta

            # Visualiser les voitures suivies pour Q15
            for car_id, data in g_tracked_cars_q15.items():
                cx_track, cy_track = data['centroid']
                color = (0, 255, 0) if data.get('counted') else (0, 0, 255)  # Vert si compté, Rouge sinon
                cv2.putText(current_color_frame_display_main, str(car_id)[-3:], (cx_track - 10, cy_track - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.circle(current_color_frame_display_main, (cx_track, cy_track), 5, color, 1)

            cv2.imshow('Q15 - Comptage Ligne Horizontale (+Q12/Q14)', current_color_frame_display_main)

            key = cv2.waitKey(30) & 0xFF  # 30ms pour une lecture fluide
            if key == 27:  # Touche Echap
                print("Arrêt demandé par l'utilisateur.")
                break
            elif key == ord('p'):  # Touche 'p' pour pause/reprise
                print("Pause. Appuyez sur 'p' pour reprendre ou 'echap' pour quitter.")
                while True:
                    key_pause = cv2.waitKey(0) & 0xFF
                    if key_pause == ord('p'):
                        print("Reprise.")
                        break
                    elif key_pause == 27:
                        print("Arrêt demandé par l'utilisateur pendant la pause.")
                        i = num_total_frames  # Pour sortir de la boucle principale
                        break
                if i == num_total_frames:
                    break

        if i >= num_total_frames - 1 and not (cv2.waitKey(1) & 0xFF == 27):  # Si la boucle s'est terminée normalement
            print(
                f"Fin du traitement. Q15 Total Cars: {g_total_cars_counted_q15}. Appuyez sur une touche pour quitter.")
            cv2.waitKey(0)
    else:
        print("Erreur lors du chargement des frames vidéo.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("\n--- Fin du programme ---")

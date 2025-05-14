import cv2
import numpy as np
from matplotlib import pyplot as plt

# Chargement en niveaux de gris
img_gray = cv2.imread('imagesDeTest/monarch.png', cv2.IMREAD_GRAYSCALE)
print(f"Forme (gris) : {img_gray.shape}, Type : {img_gray.dtype}")
# Forme typique : (H, W)
# Chargement en couleur (BGR)
img_color = cv2.imread('imagesDeTest/monarch.png', cv2.IMREAD_COLOR)
print(f"Forme (couleur) : {img_color.shape}, Type : {img_color.dtype}")
# Forme typique : (H, W, 3)
# Chargement d’une image avec canal alpha (par exemple PNG avec transparence)
img_rgba = cv2.imread('imagesDeTest/jaguar_rgba.png', cv2.IMREAD_UNCHANGED)
print(f"Forme : {img_rgba.shape}") # (H, W, 4)
print(f"Type : {img_rgba.dtype}")
# Accès à un pixel (100, 150) :
b, g, r, a = img_rgba[100, 150]
print(f"B: {b}, G: {g}, R: {r}, A: {a}")

### Étape 3 redimmensionnement d'image

# 1 Chargez une image de 256x256 pixels en niveaux de gris, par exemple peppers-256.png
img_gray = cv2.imread('imagesDeTest/peppers-256-RGB.png', cv2.IMREAD_GRAYSCALE)
print(f"Forme (gris) : {img_gray.shape}, Type : {img_gray.dtype}")

# 2 Écrivez une fonction Python qui construit une nouvelle image 64x64
def redimensionner_a_la_main(image_originale, facteur_reduction=4):

    facteur_reduction = 4

    hauteur_originale, largeur_originale = image_originale.shape[:2]

    nouvelle_hauteur = hauteur_originale // facteur_reduction
    nouvelle_largeur = largeur_originale // facteur_reduction

    array_zeros = np.zeros((nouvelle_hauteur, nouvelle_largeur), dtype=image_originale.dtype)

    for i in range(nouvelle_hauteur):
        for j in range(nouvelle_largeur):
            array_zeros[i, j] = 1/2 * img_gray[i * facteur_reduction, j * facteur_reduction] + 1/2 * img_gray[i * facteur_reduction +1, j * facteur_reduction] + 1/2 * img_gray[i * facteur_reduction + 1, j * facteur_reduction + 1] + 1/2 * img_gray[i * facteur_reduction, j * facteur_reduction + 1]

    print(f"Forme (gris) : {array_zeros.shape}, Type : {array_zeros.dtype}")
    return array_zeros

def afficher_image(image, nom_fenetre="Image"):
    cv2.imshow(nom_fenetre, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_redimensionnee = redimensionner_a_la_main(img_gray)
afficher_image(image_redimensionnee, "Image Redimensionnée")

### Étape 4 Application de Filtres de Convolution
img_gray_512 = cv2.imread('imagesDeTest/peppers-512.png', cv2.IMREAD_GRAYSCALE)

def convolution_2d(image_originale, filtre):
    hauteur_originale = image_originale.shape[0]
    largeur_originale = image_originale.shape[1]
    hauteur_filtre = filtre.shape[0]
    largeur_filtre = filtre.shape[1]

    # Calcul du padding nécessaire pour centrer le filtre
    padding_hauteur = hauteur_filtre // 2
    padding_largeur = largeur_filtre // 2

    # Créer une image avec padding pour gérer les bords
    image_paddee = np.pad(image_originale,
                          ((padding_hauteur, padding_hauteur), (padding_largeur, padding_largeur)),
                          mode='constant')

    # Calcul de la taille de l'image résultante (même taille que l'originale)
    image_convoluee = np.zeros_like(image_originale)

    # Parcours de l'image originale (pour le centre du filtre)
    for i in range(hauteur_originale):
        for j in range(largeur_originale):
            somme = 0
            # Parcours du filtre
            for k in range(hauteur_filtre):
                for v in range(largeur_filtre):
                    # Calcul des coordonnées du pixel dans l'image paddee
                    y = i - padding_hauteur + k
                    x = j - padding_largeur + v
                    somme += filtre[k, v] * image_paddee[y, x]
            image_convoluee[i, j] = somme
    return image_convoluee

    print(f"Forme (gris) : {array_zeros.shape}, Type : {array_zeros.dtype}")
    return array_zeros

# Filtre 1
filter1 = (1/16) * np.array([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]])

# Filtre 2
filter2 = (1/9) * np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])

# Filtre 3
filter3 = np.array([[1, -3, 1],
                    [-3, 9, -3],
                    [1, -3, 1]])

# Filtre 4
filter4 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# Filtre 5
filter5 = np.array([[0, -1, -1],
                    [1, 0, -1],
                    [1, 1, 0]])

# Filtre 6
filter6 = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

filters = [filter1, filter2, filter3, filter4, filter5, filter6]

for i, filter in enumerate(filters):
    filtered_image = convolution_2d(img_gray_512, filter)
    afficher_image(filtered_image, f"Image Filtrée {i+1}")

### Étape 5 Morphologie Mathématique

def binarize_image(image, threshold=128):
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 255
    return binary_image

def erosion(image, structuring_element):
    image_height, image_width = image.shape
    se_height, se_width = structuring_element.shape
    pad_height, pad_width = se_height // 2, se_width // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    eroded_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+se_height, j:j+se_width]
            if np.all(region == structuring_element * 255):
                eroded_image[i, j] = 255

    return eroded_image

def dilation(image, structuring_element):
    image_height, image_width = image.shape
    se_height, se_width = structuring_element.shape
    pad_height, pad_width = se_height // 2, se_width // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    dilated_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+se_height, j:j+se_width]
            if np.any(region[structuring_element == 1] == 255):
                dilated_image[i, j] = 255

    return dilated_image

def opening(image, structuring_element):
    eroded = erosion(image, structuring_element)
    opened = dilation(eroded, structuring_element)
    return opened

def closing(image, structuring_element):
    dilated = dilation(image, structuring_element)
    closed = erosion(dilated, structuring_element)
    return closed

# Charger l'image et la binariser
img_gray_512 = cv2.imread('imagesDeTest/peppers-512.png', cv2.IMREAD_GRAYSCALE)
binary_image = binarize_image(img_gray_512)

# Définir l'élément structurant
structuring_element = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

# Appliquer les opérations de morphologie mathématique
eroded_image = erosion(binary_image, structuring_element)
dilated_image = dilation(binary_image, structuring_element)
opened_image = opening(binary_image, structuring_element)
closed_image = closing(binary_image, structuring_element)

# Afficher les résultats
afficher_image(binary_image, "Image Binaire")
afficher_image(eroded_image, "Image Érodée")
afficher_image(dilated_image, "Image Dilatée")
afficher_image(opened_image, "Image Ouverte")
afficher_image(closed_image, "Image Fermée")

structuring_element_cross = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]])

eroded_image_cross = erosion(binary_image, structuring_element_cross)
dilated_image_cross = dilation(binary_image, structuring_element_cross)
opened_image_cross = opening(binary_image, structuring_element_cross)
closed_image_cross = closing(binary_image, structuring_element_cross)

afficher_image(eroded_image_cross, "Image Érodée (Élément Structurant Croix)")
afficher_image(dilated_image_cross, "Image Dilatée (Élément Structurant Croix)")
afficher_image(opened_image_cross, "Image Ouverte (Élément Structurant Croix)")
afficher_image(closed_image_cross, "Image Fermée (Élément Structurant Croix)")

### Étape 6 Transformée de Fourier

def apply_fourier_transform(image):
    # Appliquer la transformée de Fourier
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Calculer le spectre d'amplitude
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

    return magnitude_spectrum

def plot_fourier_transform(original_image, transformed_image, title):
    plt.subplot(121), plt.imshow(original_image, cmap='gray')
    plt.title('Image Originale'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(transformed_image, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

# Charger les images
img_D1r = cv2.imread('imagesDeTest/D1r.pgm', cv2.IMREAD_GRAYSCALE)
img_D11r = cv2.imread('imagesDeTest/D11r.pgm', cv2.IMREAD_GRAYSCALE)
img_D46r = cv2.imread('imagesDeTest/D46r.pgm', cv2.IMREAD_GRAYSCALE)

# Appliquer la transformée de Fourier et afficher les résultats
magnitude_spectrum_D1r = apply_fourier_transform(img_D1r)
plot_fourier_transform(img_D1r, magnitude_spectrum_D1r, 'Spectre d\'Amplitude D1r')

magnitude_spectrum_D11r = apply_fourier_transform(img_D11r)
plot_fourier_transform(img_D11r, magnitude_spectrum_D11r, 'Spectre d\'Amplitude D11r')

magnitude_spectrum_D46r = apply_fourier_transform(img_D46r)
plot_fourier_transform(img_D46r, magnitude_spectrum_D46r, 'Spectre d\'Amplitude D46r')

# Charger l'image peppers-512.png
img_peppers = cv2.imread('imagesDeTest/peppers-512.png', cv2.IMREAD_GRAYSCALE)

# Appliquer la transformée de Fourier et afficher le résultat
magnitude_spectrum_peppers = apply_fourier_transform(img_peppers)
plot_fourier_transform(img_peppers, magnitude_spectrum_peppers, 'Spectre d\'Amplitude Peppers')

# Effectuer une rotation de l'image
angle = 45
rows, cols = img_peppers.shape
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
rotated_img_peppers = cv2.warpAffine(img_peppers, rotation_matrix, (cols, rows))

# Appliquer la transformée de Fourier sur l'image rotée et afficher le résultat
magnitude_spectrum_rotated_peppers = apply_fourier_transform(rotated_img_peppers)
plot_fourier_transform(rotated_img_peppers, magnitude_spectrum_rotated_peppers, 'Spectre d\'Amplitude Peppers ')



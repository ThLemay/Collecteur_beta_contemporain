import cv2
import argparse
from screeninfo import get_monitors # pip install screeninfo

WINDOW_NAME = "Press 'c' to take a measurement"

# Fonction pour calculer la moyenne d'une liste de tuples
def calculer_moyennes(mesures):
    taille_x_moy = sum([m[0] for m in mesures]) / len(mesures)
    taille_y_moy = sum([m[1] for m in mesures]) / len(mesures)
    b_moy = sum([m[2][0] for m in mesures]) / len(mesures)
    g_moy = sum([m[2][1] for m in mesures]) / len(mesures)
    r_moy = sum([m[2][2] for m in mesures]) / len(mesures)
    return taille_x_moy, taille_y_moy, (b_moy, g_moy, r_moy)

def capture_measurements():
    # ID initial
    id = 1  # id (uint8_t) qui commence à 1 et sera incrémenté après chaque 5 mesures

    # Stockage des 5 mesures
    mesures = []
    
    # Obtenir la résolution de l'écran
    monitor = get_monitors()[0]  # Prendre le premier écran (ou ajustez si vous avez plusieurs écrans)
    screen_width = monitor.width
    screen_height = monitor.height
    print(f"screen_width = {screen_width} | screen_height = {screen_height}")
    new_width = screen_width // 2 # division entiere
    new_height = screen_height // 2
    
    # Nommer la fenêtre (cela permet de la manipuler)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # Déplacer la fenêtre à une position spécifique
    cv2.moveWindow(WINDOW_NAME, 100, 100)  # (x, y)

    # Affiche une fenêtre vide (ou ton flux vidéo)
    img = cv2.imread("Accueil.jpg")
    # Redimensionner l'image pour s'adapter à la moitié de l'écran
    img = cv2.resize(img, (new_width, new_height))
    y, x = img.shape[:2]
    print(f"x = {x} | y = {y}")
    croped_img = img[y//4:y-0, 0:x-x//2] # cropped = img[start_row:end_row, start_col:end_col]
    # croped_img = img[250:y-1400, 270:x-350] # cropped = img[start_row:end_row, start_col:end_col]
    
    cv2.imshow(WINDOW_NAME, croped_img)
    # Boucle pour capturer les mesures
    while True:

        
        # Attend 1 milliseconde pour une touche
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Exemple : on génère ici des valeurs aléatoires pour illustrer, mais tu peux mettre tes vraies données
            taille_x = 100.0 + len(mesures) * 10  # Exemple de valeurs
            taille_y = 50.0 + len(mesures) * 5
            couleur_bgr = (len(mesures) * 50, 100, 150)
            
            # Ajouter la mesure à la liste (taille_x, taille_y, couleur BGR)
            mesures.append((taille_x, taille_y, couleur_bgr))
            print(f"Mesure {len(mesures)} ajoutée : Taille X = {taille_x}, Taille Y = {taille_y}, Couleur BGR = {couleur_bgr}")
            
            # Si 5 mesures ont été prises, on calcule la moyenne
            if len(mesures) == 5:
                taille_x_moy, taille_y_moy, couleur_bgr_moy = calculer_moyennes(mesures)
                
                # Écrire la moyenne dans le fichier avec un ID différent à chaque série de 5 mesures
                with open("output.txt", "a") as file:
                    file.write(f"ID: {id}, Moyenne Taille X: {taille_x_moy:.2f}, Moyenne Taille Y: {taille_y_moy:.2f}, Moyenne Couleur BGR: {couleur_bgr_moy}\n")
                print(f"Moyennes écrites dans output.txt avec ID = {id} : Taille X = {taille_x_moy}, Taille Y = {taille_y_moy}, Couleur BGR = {couleur_bgr_moy}")
                
                # Incrémenter l'ID pour la prochaine série de mesures
                id += 1
                
                # Réinitialiser les mesures après avoir écrit dans le fichier
                mesures = []
        
        # Si on presse 'q', on quitte la boucle
        elif key == ord('q'):
            print("Fin de la boucle.")
            break
        
            # Vérifie si une touche est appuyée
        if key == 27:  # 27 correspond à la touche 'ESC'
            break

        # Vérifie si la fenêtre a été fermée
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

ap = argparse.ArgumentParser(description="Demo script for NUT project")
ap.add_argument("-c", "--calibration", action='store_true', required=False, help="Enable the detection calibration")
args = vars(ap.parse_args())

if(args["calibration"] == True):
    capture_measurements()
else:
    print("Mode calibration non activé. Lancer avec '--calibration' pour activer la capture.")

# Ferme les fenêtres
cv2.destroyAllWindows()
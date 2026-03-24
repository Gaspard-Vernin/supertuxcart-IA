import os
import gymnasium as gym
import sys
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import pystk2_gymnasium
import numpy as np
import torch
import math
import supertuxcart_rainbow_dqn as m

chemin_save = 'save_iter_2700.pth'

if __name__ == "__main__":
    frames = []
    tab_map_action = m.tab_map_action
    taille_input = 27
    nb_actions = len(tab_map_action)

    net = m.Net(n=3, nb_actions=nb_actions, taille_etat=taille_input)
    net.load_state_dict(torch.load(chemin_save))
    net.supprimer_bruit()

    # render_mode=None pour désactiver l'affichage graphique
    env = gym.make("supertuxkart/simple-v0", render_mode="human", num_kart=2, max_episode_steps=10000,track="cocoa_temple")

    total_reward = 0
    last_distance_parcourue = 0
    compteur_pas_assez_de_vitesse = 0
    compteur_trop_loin = 0
    seuil_vitesse = 0.5
    last_energie = 0

    etat, _ = env.reset()
    done = False
    print("lesgi")
    a=input()
    while not done:
        etat_forw = m.generer_vector_etat(etat)
        with torch.no_grad():
            forw = net.forward(etat_forw)
            q_values = (forw * net.valeurs_possibles_actions).sum(dim=2)
            action_idx = torch.argmax(q_values, dim=1).item()

        action_tuple = tab_map_action[action_idx]
        action = m.creer_action(action_tuple[0], action_tuple[1], action_tuple[2], action_tuple[3], action_tuple[4])

        etat_suivant, _, terminated, truncated, _ = env.step(action)
        print(etat)
        # Extraction propre de l'image
        if "image" in etat:
            frames.append(etat["image"][0])
        elif "rgb" in etat:
            frames.append(etat["rgb"][0])

        largeur_chemin = etat_suivant["paths_width"][0][0]
        if abs(etat_suivant["center_path_distance"][0]) > largeur_chemin:
            compteur_trop_loin += 1
        else:
            compteur_trop_loin = max(0, compteur_trop_loin - 1)
        if compteur_trop_loin >= 10:
            truncated = True

        vitesse = math.sqrt(etat_suivant["velocity"][0]**2 + etat_suivant["velocity"][1]**2 + etat_suivant["velocity"][2]**2)
        if vitesse < seuil_vitesse:
            compteur_pas_assez_de_vitesse += 1
        else:
            compteur_pas_assez_de_vitesse = max(0, compteur_pas_assez_de_vitesse - 1)
        if compteur_pas_assez_de_vitesse >= 60:
            truncated = True

        energie = etat["energy"][0]
        drift = int(action_tuple[4])
        reward = m.def_reward(
            distance_parcourue=etat_suivant["distance_down_track"][0],
            dist_centre=etat_suivant["center_path_distance"][0],
            last_distance_parcourue=last_distance_parcourue,
            last_energie=last_energie,
            energie=energie,
            drift=drift,
            skeed=etat_suivant["skeed_factor"][0],
            point1=etat_suivant["paths_start"][0],
            point2=etat_suivant["paths_start"][2]
        )

        last_energie = energie
        last_distance_parcourue = etat_suivant["distance_down_track"][0]
        done = terminated or truncated
        total_reward += reward
        etat = etat_suivant

    env.close()

    if len(frames) > 0:
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile("run.mp4")
        print("Vidéo sauvegardée.")
    else:
        print("Erreur : Aucune image trouvée dans l'état.")

    print(f"distance : {etat_suivant['distance_down_track'][0]:.1f}")
    print(f"reward totale : {total_reward:.2f}")
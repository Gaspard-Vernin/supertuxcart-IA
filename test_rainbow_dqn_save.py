import os
import gymnasium as gym
import pystk2_gymnasium
import numpy as np
import torch
import math
import time
import supertuxcart_rainbow_dqn as m

chemin_save = 'save_iter_100.pth'

if __name__ == "__main__":
    tab_map_action = m.tab_map_action
    taille_input = 27
    nb_actions = len(tab_map_action)

    net = m.Net(n=3, nb_actions=nb_actions, taille_etat=taille_input)
    net.load_state_dict(torch.load(chemin_save))
    net.supprimer_bruit()

    env = gym.make("supertuxkart/simple-v0", render_mode="human", num_kart=2, max_episode_steps=10000)

    total_reward = 0
    total_distance = 0
    last_distance_parcourue = 0
    compteur_pas_assez_de_vitesse = 0
    compteur_trop_loin = 0
    seuil_vitesse = 0.5
    last_energie = 0

    print("len de tabmap = ", len(tab_map_action))

    etat, _ = env.reset()
    done = False

    while not done:
        etat_forw = m.generer_vector_etat(etat)
        with torch.no_grad():
            forw = net.forward(etat_forw)
            q_values = (forw * net.valeurs_possibles_actions).sum(dim=2)
            action_idx = torch.argmax(q_values, dim=1).item()

        action_tuple = tab_map_action[action_idx]
        action = m.creer_action(action_tuple[0], action_tuple[1], action_tuple[2], action_tuple[3], action_tuple[4])

        etat_suivant, _, terminated, truncated, _ = env.step(action)

        largeur_chemin = etat_suivant["paths_width"][0][0]
        if abs(etat_suivant["center_path_distance"][0]) > largeur_chemin:
            compteur_trop_loin += 1
        else:
            compteur_trop_loin = max(0, compteur_trop_loin - 1)
        if compteur_trop_loin >= 10:
            truncated = True
            print("run terminée car trop loin")

        vitesse = math.sqrt(etat_suivant["velocity"][0]**2 + etat_suivant["velocity"][1]**2 + etat_suivant["velocity"][2]**2)
        if vitesse < seuil_vitesse:
            compteur_pas_assez_de_vitesse += 1
        else:
            compteur_pas_assez_de_vitesse = max(0, compteur_pas_assez_de_vitesse - 1)
        if compteur_pas_assez_de_vitesse >= 60:
            truncated = True
            print("run terminée car trop lent")

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
        total_distance += etat_suivant["distance_down_track"][0]
        etat = etat_suivant
        time.sleep(0.05)

    print(f"distance : {etat_suivant['distance_down_track'][0]:.1f}")
    print(f"reward totale : {total_reward:.2f}")
    env.close()
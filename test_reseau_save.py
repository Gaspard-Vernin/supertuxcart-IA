import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import time
import math
import mariokart_discrete_version as m
#git update-index --assume-unchanged save_reseau_supertuxcart.pth
#git update-index --no-assume-unchanged nom_de_ton_fichier.pth
chemin_save = 'save_reseau_supertuxcart.pth'
if __name__ == "__main__":
    tab_map_action = m.tab_map_action
    taille_input = m.taille_input
    nb_actions = len(tab_map_action)
    net = m.Dueling_network(taille_input, nb_actions,lr=0)
    net.load_state_dict(torch.load(chemin_save))
    env = gym.make("supertuxkart/simple-v0", render_mode="human", num_kart=2,max_episode_steps=10000)
    total_reward=0
    liste_reward=[]
    liste_distances=[]
    total_distance=0
    best_total_reward = -10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    last_distance_parcourue = 0
    print("len de tabmap = ",len(tab_map_action))
    compteur_nb_ajout = 0
    last_distance_parcourue=0
    etat,_=env.reset()
    done=False 
    compteur_pas_assez_de_vitesse = 0
    seuil_vitesse = 0.5
    compteur_trop_loin = 0
    temps_derapage=0
    while not done:
        etat_forw=m.generer_vector_etat(etat,temps_derapage)
        forw=net.forward(etat_forw)
        #on renvoie l'action avec la plus grosse Q_value
        action=torch.argmax(forw).item()
        action_tuple = tab_map_action[action]
        action = m.creer_action(action_tuple[0],action_tuple[1],action_tuple[2])
        if(action["drift"]==1):
                temps_derapage+=1 
        else : 
            temps_derapage=0
        etat_suivant,reward,terminated,truncated,_=env.step(action)
        
        largeur_chemin = etat_suivant["paths_width"][0][0]
        if(abs(etat_suivant["center_path_distance"][0])>largeur_chemin/2):
            compteur_trop_loin+=1
        else : 
            compteur_trop_loin = max(0,compteur_trop_loin-1)
        if(compteur_trop_loin>=5):
            truncated=True 
            reward-=10 
            print("run terminée car trop loin")
        if(m.norme(etat_suivant["velocity"])<seuil_vitesse):
            compteur_pas_assez_de_vitesse+=1
        else : 
            #on le baisse progressivement pour repartir plus vite si on a avancer despi mais on est tjr coincés
            compteur_pas_assez_de_vitesse = max(0,compteur_pas_assez_de_vitesse-1)
        if(compteur_pas_assez_de_vitesse>=60):
            truncated = True 
            reward -= 10
            print("run terminée car trop lent")
        drift = 0 if etat["skeed_factor"][0] == 1 else 1
        reward = m.def_reward(distance_parcourue = etat_suivant["distance_down_track"][0],
                            dist_centre = etat_suivant["center_path_distance"][0] , 
                            norme_vitesse=m.norme(etat_suivant["velocity"]),last_distance_parcourue=last_distance_parcourue
                            ,drift = drift
                                ,chemin = etat_suivant["paths_start"])
        last_distance_parcourue = etat_suivant["distance_down_track"][0]
        #la partie est finie si on a gagné ou si on est sorti
        done = terminated or truncated
        #print(etat_suivant["paths_start"],"\n\n\n",etat_suivant["paths_end"],"\n\n\n",etat_suivant["center_path"],"\n\n\n")
        total_reward+=reward 
        etat=etat_suivant 
        total_distance+=(etat_suivant["distance_down_track"][0])
        time.sleep(0.05)
        if(done):
            print("distance : ",etat_suivant["distance_down_track"][0],"\n")
    env.close()
import os
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["EGL_PLATFORM"] = "surfaceless"
import numpy as np 
import pystk2_gymnasium
import torch.nn as nn
import torch.optim as optim
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque 
import random
import signal
import math
epsilon=1
gamma=0.99
max_size_buffer=10000
class Noisy_linear_layer(nn.Module):
    def __init__(self,taille_entree,taille_sortie):
        super().__init__()
        self.taille_entree=taille_entree 
        self.taille_sortie=taille_sortie 
        self.esperance_poids = nn.Parameter(nn.init.uniform_(torch.empty(taille_sortie,taille_entree),-1/math.sqrt(taille_entree),1/math.sqrt(taille_entree)))
        self.esperance_biais = nn.Parameter(nn.init.uniform_(torch.empty(taille_sortie),-1/math.sqrt(taille_entree),1/math.sqrt(taille_entree)))
        self.variance_poids= nn.Parameter(nn.init.constant_(torch.empty(taille_sortie,taille_entree),0.5/math.sqrt(taille_entree)))
        self.variance_biais=nn.Parameter(nn.init.constant_(torch.empty(taille_sortie),0.5/math.sqrt(taille_sortie)))
    
        #on init les bruits a 0 car quand on test un reseau pré-entrainé, on veut aucun bruit et donc tt a zero.
        self.noise_in=torch.zeros(taille_entree)
        self.noise_out=torch.zeros(taille_sortie)
        self.matrice_bruit_poids =torch.zeros(taille_sortie,taille_entree)
    def supprimer_bruit(self):
        self.noise_in=torch.zeros(self.taille_entree)
        self.noise_out=torch.zeros(self.taille_sortie)
        self.matrice_bruit_poids =torch.zeros(self.taille_sortie,self.taille_entree)
    def init_noise_in(self):
        bruit= torch.randn(self.taille_entree)
        return torch.sign(bruit)*torch.sqrt(torch.abs(bruit))
    def init_noise_out(self):
        bruit= torch.randn(self.taille_sortie)
        return torch.sign(bruit)*torch.sqrt(torch.abs(bruit))
    def forward(self,input):    
        poids = (self.esperance_poids+self.variance_poids*self.matrice_bruit_poids)
        biais = (self.esperance_biais+self.variance_biais*self.noise_out)
        output = torch.nn.functional.linear(input,poids,biais)
        return output
    def reset_bruit(self):
        self.noise_in=self.init_noise_in()
        self.noise_out=self.init_noise_out()
        self.matrice_bruit_poids = torch.outer(self.noise_out,self.noise_in)
        return
class Net (nn.Module):
    def __init__(self,n,nb_actions,taille_etat,N=51):
        super(Net,self).__init__()
        self.nb_actions = nb_actions
        self.taille_etat=taille_etat
        self.fc1=Noisy_linear_layer(self.taille_etat,512)
        self.fc2=Noisy_linear_layer(512,256)
        self.fc3A=Noisy_linear_layer(256,self.nb_actions*N)
        self.fc3V=Noisy_linear_layer(256,1*N)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.v_min=-10
        self.v_max=10
        self.valeurs_possibles_actions=torch.linspace(self.v_min,self.v_max,N)
        self.delta=(self.v_max-self.v_min)/(N-1)
        self.N=N
        self.n=n
    def reset_bruit(self):
        self.fc1.reset_bruit()
        self.fc2.reset_bruit()
        self.fc3A.reset_bruit()
        self.fc3V.reset_bruit()
    def supprimer_bruit(self):
        self.fc1.supprimer_bruit()
        self.fc2.supprimer_bruit()
        self.fc3A.supprimer_bruit()
        self.fc3V.supprimer_bruit()
    def forward(self,input):
        if input.dim() == 1:
            input = input.unsqueeze(0)
        c1=torch.relu(self.fc1(input))
        c2=torch.relu(self.fc2(c1))
        a=self.fc3A(c2)
        v=self.fc3V(c2)
        #on reshape les actions pour un tab de la forme (batch,actions,N) pour action et (batch,1,N) pour values
        #-1 déduit directement la dimension restante, ici on sait pas c quoi pr batch donc on met -1
        v=v.view(-1,1,self.N)
        a=a.view(-1,self.nb_actions,self.N)
        distributions_actions=v+(a-a.mean(dim=1,keepdim=True))
        #on normalise chaque distribution pour chaque action
        return torch.softmax(distributions_actions,dim=2)
    def backprop(self,output,target,compensations,buffer,indices):
        cross_entropy=-torch.sum(target*torch.log(output+1e-8),dim=1)
        loss = (compensations*cross_entropy).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        buffer.update(indices,cross_entropy)
    def train(self,input,target):
        output=net(input)
        net.backprop(torch.tensor(target),output)
    def train_batches(self,buffer,goal_net,gamma):
        if buffer.sumtree.taille_actuelle<buffer.batch_size:
            #le buffer n'est pas plein, on ignore l'appel
            return
        #recuperer les transitions du batch du buffer
        transitions, indices, compensations = buffer.tirage()

        rewards=torch.tensor([transition.reward for transition in transitions]).unsqueeze(1)
        dones=torch.tensor([transition.done for transition in transitions]).unsqueeze(1)
        states=torch.tensor(np.array(np.stack([transition.state for transition in transitions]),dtype=np.float32))
        next_states=torch.tensor(np.array(np.stack([transition.prochaine_etat for transition in transitions]),dtype=np.float32))
        actions=torch.tensor([transition.action for transition in transitions])
        #on feed les transitions
        feed_state = self.forward(torch.tensor(states))
        pred_actions_jouees = feed_state[torch.arange(len(transitions)), actions, :]
        #calcul des target pour chaque transitions
        with torch.no_grad():
            #on forward les prochaines actions 
            
            forw_net = self.forward(next_states)
            #on calcule les Q values qui sont donc l'esperances des lois trouvées pr chaque action
            q_values_next_action_net = (forw_net*self.valeurs_possibles_actions).sum(dim=2)
            #on choisit la plus grande Q value pour l'action suivante
            next_actions = torch.argmax(q_values_next_action_net,dim=1)

            #on calcule la distribution associée a la next action mais dans le goalnet
            forw_goal_net=goal_net.forward(next_states)
            prochaine_distributions=forw_goal_net[torch.arange(len(transitions)),next_actions,:].float()

            target = rewards + (gamma**self.n)*self.valeurs_possibles_actions*(~dones)
            if target.min()==self.v_min or target.max() ==self.v_max : print("target trop grande : ",target)

            target=torch.clamp(target,self.v_min,self.v_max)
            target_normalisee=(target-self.v_min)/self.delta 
            atome_gauche = (target_normalisee.floor()).long()
            atome_droit = (target_normalisee.ceil()).long()

            #traitement du cas chiant où on tombe pile sur un atome
            atome_droit[(atome_droit == atome_gauche) & (atome_droit < self.N - 1)] += 1
            atome_gauche[(atome_droit == atome_gauche) & (atome_gauche > 0)] -= 1

            distrib_finale=torch.zeros(target_normalisee.shape)
            repartition_gauche=prochaine_distributions*(atome_droit-target_normalisee).float()
            repartition_droite=prochaine_distributions*(target_normalisee-atome_gauche).float()

            distrib_finale.scatter_add_(1,atome_gauche,repartition_gauche)
            distrib_finale.scatter_add_(1,atome_droit,repartition_droite)
        self.backprop(pred_actions_jouees,distrib_finale,compensations,buffer,indices)
class Agent():
    def __init__(self, net,goal_net,buffer,gamma):
        self.net=net 
        self.goal_net=goal_net
        self.buffer=buffer
        self.gamma=gamma
    def fusionner_poids(self):
        self.goal_net.load_state_dict(self.net.state_dict())
    def fusion_douce_poids(self):
        tau=0.005
        with torch.no_grad():
            for poid_target, poid_actuel in zip(self.goal_net.parameters(), self.net.parameters()):
                poid_target.data.copy_(tau * poid_actuel.data + (1.0 - tau) * poid_target.data)
    def train(self):
        self.net.train_batches(self.buffer,self.goal_net,self.gamma)
        self.fusion_douce_poids()
class Transition():
    def __init__(self,state,action,reward,prochaine_etat,done):
        self.state=state
        self.action=action 
        self.reward=reward 
        self.prochaine_etat=prochaine_etat
        self.done=done
class Sumtree:
    def __init__(self,taille):
        self.taille=taille 
        #un arbre parfait qui a x feuilles à 2x-1 noeuds 
        self.tree = np.zeros(taille*2-1)
        #data[i] représente la transition indexée par i
        self.data = np.empty(taille,dtype=Transition)
        #pointeur pointe sur la prochaine case sur laquelle on va écrire
        self.pointeur=0
        self.taille_actuelle=0
        #on initialise max_priorité à 1, pour que la première valeur ajoutée dans le tree ait cette priorité
        #et pas une priorité nulle ainsi elle sera tirée et sa vraie priorité sera mise à la place de 1
        self.max_priorité = 1
    def i_data_to_i_arbre(self,x):
        """renvoie l'indice de l'arbre associé à l'indice de data"""
        return self.taille-1+x
    def i_arbre_to_i_data(self,x):
        """renvoie l'indice de data associé à l'indice de l'arbre"""
        return -(self.taille-1)+x
    def add(self,priorité,transi):
        """ajoute transi/priorité et supprime la plus vieilles transi du sumtree"""
        #on stocke la transition a la bonne case
        self.data[self.pointeur]=transi 
        #on update la priorité dans l'arbre 
        self.update(self.i_data_to_i_arbre(self.pointeur),priorité)
        #on avance write, si il dépasse taille, il repart à 0
        self.pointeur=(self.pointeur+1)%self.taille
        if(self.taille_actuelle<self.taille):
            self.taille_actuelle+=1
    def update(self,indice_arbre,priorité):
        """prend l'indice dans l'arbre a changer et une priorité et fait les updates nécessaires"""
        if(priorité>self.max_priorité):
            self.max_priorité=priorité
        ancienne_priorité = self.tree[indice_arbre]
        #quand indice_arbre=0, il est ensuite update a -1 donc on quitte la boucle pile après avoir finit toutes les modif
        while(indice_arbre>=0):
            self.tree[indice_arbre]+=(priorité-ancienne_priorité)
            if indice_arbre==0:
                break
            #le parent d'un noeud d'indexe i dans un arbre parfait est a l'index partie entière de(i-1)//2
            indice_arbre = (indice_arbre-1)//2
    def total(self):
        """renvoie la somme des priorités stockées"""
        return self.tree[0]
    def get(self, priorité):
        """renvoie (transition,indice_data) en fonction de la priorité donnée"""
        i=0
        while True:
            #si c'est une feuille
            if(i>=self.taille-1):
                indice_data=self.i_arbre_to_i_data(i)
                return (self.data[indice_data],indice_data)
            i_enfant_gauche=2*i+1
            #si la transi cherchée est à gauche, le fils gauche de i est a l'index 2i+1
            if priorité <=self.tree[i_enfant_gauche]:
                i=i_enfant_gauche
            #si la transi est à droite, le fils droit de i est a l'index 2i+2, 
            #il faut soustraire à la priorité la priorité de gauche car elle est comptée dans priorité
            else :
                priorité-=self.tree[2*i+1]
                i=i_enfant_gauche+1
class Buffer:
    def __init__(self,capacity,alpha,batch_size,beta):
        self.sumtree=Sumtree(capacity)
        self.alpha=alpha 
        self.batch_size=batch_size
        self.beta=beta
    def add(self,transi):
        """ajoute la transition dans le buffer avec la plus grande priorité du buffer existante,
            la priorité sera update quand on aura train au moins une fois dessus et donc qu'on update
            ainsi toutes les priorités (à priori) seront jouées au moins une fois"""
        self.sumtree.add(self.sumtree.max_priorité,transi)
    def tirage(self):
        """renvoie un tirage de taille self.buffer_size respectant les priorités du sumtree sous la forme (transitions,indices,compensations)"""
        #pour respecter les priorités au mieux, on va diviser le "segment" [0,sumtree.total()] en 
        #batch_size segments côte à côte. on tire une transi dans chaque segment. Ainsi on force le
        #tirage à s'étaler sur tout le spectre des priorités et donc on réduit les cas où la faute à
        #pas de chance, on a tiré que des transition nulles
        taille_segment = self.sumtree.total()/self.batch_size 
        """
        version claire de la version vectoriése ci dessous<
        tirage = []
        for i in range(taille_segment):
            #on tire la position sur l'interval, puis on décale jusqu'à l'interval
            prio_tirée = (np.random.random()*taille_segment)+(i*taille_segment)
            tirage.append(self.sumtree.get(prio_tirée))
        """
        #on tire les priorités sur les segments et on extrait transitions et indices
        priorités = [(np.random.random()*taille_segment)+(i*taille_segment)for i in range(self.batch_size)]
        tirage = [self.sumtree.get(priorités[i]) for i in range(self.batch_size) ]
        transitions = [tirage[i][0] for i in range(self.batch_size)]
        indices= [tirage[i][1] for i in range(self.batch_size)]
        #on calcule la compensation à ajouter aux TD_errors
        priorités = torch.tensor([self.sumtree.tree[self.sumtree.i_data_to_i_arbre(idx)] for idx in indices], dtype=torch.float32)
        priorités_normalisées = priorités / self.sumtree.total()
        compensations = torch.pow(self.sumtree.taille_actuelle*priorités_normalisées,-self.beta)
        #on normalise la compensation pour plus de stabilité, N peut être très grand...
        compensations/=compensations.max()
        return transitions,indices,compensations
    def update(self,indices,td_errors):
        td_errors=td_errors.detach().cpu().numpy()
        for indice,td_error in zip(indices,td_errors):
            poid=abs(td_error)**self.alpha+1e-5
            self.sumtree.update(self.sumtree.i_data_to_i_arbre(indice),poid)
def map_action(pas_steer,pas_accel):
    steer=[round((i+1)/pas_steer,2) for i in range((-pas_steer-1),(pas_steer))]
    accel=[round((i+1)/pas_accel,2) for i in range(pas_accel)]
    nitro=[0,1]
    fire=[0,1]
    drift=[0,1]
    action_map=[]
    for s in steer:
        for a in accel:
            for nn_ in nitro:
                for f in fire:
                    for d in drift:
                        action_map.append((a,s,nn_,f,d))
    return action_map
def creer_action(accel,steer,nitro,fire,drift):
    return {
        'acceleration': np.float32(accel),
        'steer': np.float32(steer),
        'brake': np.int64(0),
        'drift': np.int64(drift),
        'fire': np.int64(fire),
        'nitro': np.int64(nitro),
        'rescue': np.int64(0)
    }
def num_action(action):
    global tab_map_action
    t=(round(action["acceleration"],2),round(action["steer"],2),action["nitro"],action["fire"],action["drift"])
    t=convert_random_action_to_legal_action(t)
    for i in range(len(tab_map_action)):
        if tab_map_action[i]==t:
            return i
    raise Exception("l'action n'est pas dans les actions possibles")
def generer_vector_etat(etat):
    global obj1x,obj1z,is_obj1_boost,obj2x,obj2z,is_obj2_boost,obj3x,obj3z,is_obj3_boost,bananex,bananez,is_banane_a_banane
    center_path=etat["center_path"]
    center_path_distance=etat["center_path_distance"]
    velocity=etat["velocity"]
    chemin_a_suivre=etat["paths_start"]
    point_tres_proche=chemin_a_suivre[0]
    point_proche=chemin_a_suivre[1]
    point_moyen_distance=chemin_a_suivre[2]
    point_loin=chemin_a_suivre[3]
    energy=etat["energy"][0]
    charge_drift=etat["skeed_factor"][0]
    indices_items=[]
    indice_banane=-1
    for i in range(len(etat["items_type"])):
        if etat['items_type'][i] in (0,2,3) and len(indices_items)<3 and 0<etat['items_position'][i][2]<80:
            indices_items.append(i)
        elif etat['items_type'][i] in (1,4) and indice_banane==-1 and etat['items_position'][i][2]<80:
            indice_banane=i
    match len(indices_items):
        case 0:
            obj1x=0;obj1z=1;is_obj1_boost=0
            obj2x=0;obj2z=1;is_obj2_boost=0
            obj3x=0;obj3z=1;is_obj3_boost=0
        case 1:
            obj1x=etat['items_position'][indices_items[0]][0]/40;obj1z=etat['items_position'][indices_items[0]][2]/40;is_obj1_boost=1
            obj2x=0;obj2z=1;is_obj2_boost=0
            obj3x=0;obj3z=1;is_obj3_boost=0
        case 2:
            obj1x=etat['items_position'][indices_items[0]][0]/40;obj1z=etat['items_position'][indices_items[0]][2]/40;is_obj1_boost=1
            obj2x=etat['items_position'][indices_items[1]][0]/40;obj2z=etat['items_position'][indices_items[1]][2]/40;is_obj2_boost=1
            obj3x=0;obj3z=1;is_obj3_boost=0
        case 3:
            obj1x=etat['items_position'][indices_items[0]][0]/40;obj1z=etat['items_position'][indices_items[0]][2]/40;is_obj1_boost=1
            obj2x=etat['items_position'][indices_items[1]][0]/40;obj2z=etat['items_position'][indices_items[1]][2]/40;is_obj2_boost=1
            obj3x=etat['items_position'][indices_items[2]][0]/40;obj3z=etat['items_position'][indices_items[2]][2]/40;is_obj3_boost=1
        case _:
            raise Exception("la taille de indice_items a dépassée 3")
    if indice_banane==-1:
        bananex=0;bananez=1;is_banane_a_banane=0
    else:
        bananex=etat['items_position'][indice_banane][0]/40
        bananez=etat['items_position'][indice_banane][2]/40
        is_banane_a_banane=1
    return torch.from_numpy(np.array([
        center_path[0].item()/180, center_path[2].item()/180,
        center_path_distance[0]/180,
        velocity[0]/40, velocity[2]/40,
        point_tres_proche[0]/300, point_tres_proche[2]/300,
        point_proche[0]/300, point_proche[2]/300,
        point_moyen_distance[0]/300, point_moyen_distance[2]/300,
        point_loin[0]/300, point_loin[2]/300,
        energy/10,
        charge_drift,
        obj1x, obj1z, is_obj1_boost,
        obj2x, obj2z, is_obj2_boost,
        obj3x, obj3z, is_obj3_boost,
        bananex, bananez, is_banane_a_banane
    ], dtype=np.float32))
def convert_random_action_to_legal_action(random_action):
    global tab_map_action
    return min(tab_map_action, key=lambda couple: abs(couple[0]-random_action[0])+abs(couple[1]-random_action[1])+abs(couple[2]-random_action[2])+abs(couple[3]-random_action[3])+abs(couple[4]-random_action[4]))
def def_reward(distance_parcourue, dist_centre, last_distance_parcourue,
               last_energie, energie, drift, skeed, point1, point2):

    # signal principal : avancement (~0 à 1.0 par step)
    delta_dist = distance_parcourue - last_distance_parcourue

    # pénalité centre : dist_centre va jusqu'à ~18, on normalise par 10
    # résultat : entre 0 et -0.5 environ
    penalite_centre = -0.5 * abs(dist_centre) / 10.0

    # bonus boost
    recompense_boost = 0
    if energie - last_energie > 0:
        recompense_boost = min(0.5, 20 * (energie - last_energie))

    # punition banane
    reward_banane = 0
    if math.sqrt(bananex**2 + bananez**2) < 0.13 and bananex != 0:
        reward_banane = -1.0

    # reward dérapage
    reward_drift = 0
    angle_a_tourner = math.atan2(point2[0] - point1[0], point2[2] - point1[2])
    if drift == 1:
        if abs(angle_a_tourner) > math.pi / 5:
            reward_drift = min(0.5, (1 + 2 * skeed) * abs(angle_a_tourner - math.pi / 5))
        else:
            reward_drift = -0.3

    return delta_dist + penalite_centre + recompense_boost + reward_banane + reward_drift
def signal_handler(sig, frame):
    global env
    if env is not None:
        try:
            env.close()
        except Exception as e:
            print(f"Erreur fermeture : {e}")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
tab_map_action=map_action(3,3)
obj1x,obj1z,is_obj1_boost=0,0,0
obj2x,obj2z,is_obj2_boost=0,0,0
obj3x,obj3z,is_obj3_boost=0,0,0
bananex,bananez,is_banane_a_banane=0,0,0
env=None
last_energie = 0
taille_etat = 27
 
if __name__ == "__main__":
    import signal, sys
 

    n = 3
    
    nb_actions = len(tab_map_action)
    print(f"nb actions : {nb_actions}, taille etat : {taille_etat}")
 
    net = Net(n, nb_actions, taille_etat)
    goal_net = Net(n, nb_actions, taille_etat)
    buffer = Buffer(capacity=50000, alpha=0.6, batch_size=128, beta=0.4)
    player = Agent(net, goal_net, buffer, gamma)
    player.fusionner_poids()
 
    env = gym.make("supertuxkart/simple-v0", num_kart=2, max_episode_steps=10000)
 
    total_reward = 0
    liste_total_reward = []
    sauvegarde_etats_reward = []
    sauvegarde_transi = []
    mode_vision = True
    is_visualisation = False
    seuil_vitesse = 0.5
 
    for iter in range(5000):
        if iter != 0 and iter % 100 == 0:
            liste_total_reward.append(total_reward / 100)
            print(f"iter : {iter} score : {total_reward/100} taille buffer : {player.buffer.sumtree.taille_actuelle}")
            torch.save(player.net.state_dict(), f"save_iter_{iter}.pth")
            total_reward = 0
 
        if mode_vision:
            if iter % 25 == 0 and iter != 0:
                env.close()
                env = gym.make("supertuxkart/simple-v0", num_kart=2, max_episode_steps=10000, render_mode="human")
                is_visualisation = True
            if (iter - 1) % 25 == 0 and iter != 1:
                env.close()
                env = gym.make("supertuxkart/simple-v0", num_kart=2, max_episode_steps=10000)
                is_visualisation = False
 
        etat, _ = env.reset()
        player.net.reset_bruit()
        done = False
        sauvegarde_transi = []
        sauvegarde_etats_reward = []
        compteur_pas_assez_de_vitesse = 0
        compteur_trop_loin = 0
        last_energie = 0                         
        last_distance_parcourue = 0
 
        while not done:
            etat_forw = generer_vector_etat(etat)
            with torch.no_grad():
                forw = player.net.forward(etat_forw)
                q_values = (forw * player.net.valeurs_possibles_actions).sum(dim=2)
                action_idx = torch.argmax(q_values, dim=1).item()
 
            action_tuple = tab_map_action[action_idx]
            action = creer_action(action_tuple[0], action_tuple[1], action_tuple[2], action_tuple[3], action_tuple[4])
 
            etat_suivant, _, terminated, truncated, _ = env.step(action)
 
            vector_etat = generer_vector_etat(etat)
 
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
 
            done = terminated or truncated
 
            penalite_fin = 0
            if compteur_trop_loin >= 10:
                penalite_fin = -5.0
            if compteur_pas_assez_de_vitesse >= 60:
                penalite_fin = -5.0
 
            energie = etat["energy"][0]
            drift = int(action_tuple[4])
            recompense = def_reward(
                distance_parcourue=etat_suivant["distance_down_track"][0],
                dist_centre=etat_suivant["center_path_distance"][0],
                last_distance_parcourue=last_distance_parcourue,
                last_energie=last_energie,
                energie=energie,
                drift=drift,
                skeed=etat_suivant["skeed_factor"][0],
                point1=etat_suivant["paths_start"][0],
                point2=etat_suivant["paths_start"][2]
            ) + penalite_fin
 
            last_energie = energie
            last_distance_parcourue = etat_suivant["distance_down_track"][0]
 
            transi = Transition(vector_etat, action_idx, 0, None, None)
            sauvegarde_transi.append(transi)
            sauvegarde_etats_reward.append((vector_etat, recompense))
            total_reward += recompense
            player.net.reset_bruit()
            etat = etat_suivant
            nb_step_game = len(sauvegarde_transi)
 
        vector_etat_final = generer_vector_etat(etat_suivant)
        sauvegarde_etats_reward.append((vector_etat_final, 0))
 
        for i in range(nb_step_game):
            if i >= nb_step_game - n:
                sauvegarde_transi[i].done = True
                sauvegarde_transi[i].prochaine_etat = sauvegarde_etats_reward[-1][0]
            else:
                sauvegarde_transi[i].done = False
                sauvegarde_transi[i].prochaine_etat = sauvegarde_etats_reward[i + n][0]
            for j in range(min(n, nb_step_game - i)):
                sauvegarde_transi[i].reward += (gamma**j) * sauvegarde_etats_reward[i + j][1]
            player.buffer.add(sauvegarde_transi[i])
 
        for i in range(nb_step_game):
            if i % 4 == 0:
                player.train()
                player.net.reset_bruit()
 
        print(f"épisode {iter} — {nb_step_game} steps — distance : {etat_suivant['distance_down_track'][0]:.1f}")
 
    plt.figure(figsize=(12, 6))
    plt.plot(liste_total_reward)
    plt.title("Reward moyenne par 100 épisodes")
    plt.show()
 
    env.close()
    env = gym.make("supertuxkart/simple-v0", num_kart=2, max_episode_steps=10000, render_mode="human")
    player.net.supprimer_bruit()
 
    for i in range(3):
        etat, _ = env.reset()
        done = False
        points_episode = 0
        while not done:
            etat_forw = generer_vector_etat(etat)
            forw = player.net.forward(etat_forw)
            q_values = (forw * player.net.valeurs_possibles_actions).sum(dim=2)
            action_idx = torch.argmax(q_values, dim=1).item()
            action_tuple = tab_map_action[action_idx]
            action = creer_action(action_tuple[0], action_tuple[1], action_tuple[2], action_tuple[3], action_tuple[4])
            etat_suivant, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            points_episode += r  
            etat = etat_suivant
        print(f"points episode : {points_episode:.1f} — distance : {etat_suivant['distance_down_track'][0]:.1f}")
 
    env.close()
 
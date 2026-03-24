import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque 
import random
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
        poids = self.esperance_poids+self.variance_poids*self.matrice_bruit_poids
        biais = self.esperance_biais+self.variance_biais*self.noise_out
        output = torch.nn.functional.linear(input,poids,biais)
        return output
    def reset_bruit(self):
        self.noise_in=self.init_noise_in()
        self.noise_out=self.init_noise_out()
        self.matrice_bruit_poids = torch.outer(self.noise_out,self.noise_in)
        return
class Net (nn.Module):
    def __init__(self,n):
        super(Net,self).__init__()
        self.fc1=Noisy_linear_layer(4,128)
        self.fc2=Noisy_linear_layer(128,128)
        self.fc3A=Noisy_linear_layer(128,2)
        self.fc3V=Noisy_linear_layer(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
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
        return v+(a-a.mean(dim=1,keepdim=True))
    def backprop(self,target,q_value,compensations):
        td_errors = target - q_value
        loss = (compensations * td_errors**2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def train(self,input,target):
        output=net(input)
        net.backprop(torch.tensor(target),output)
    def train_batches(self,buffer,goal_net):
        if buffer.sumtree.taille_actuelle<buffer.batch_size:
            #le buffer n'est pas plein, on ignore l'appel
            return
        #recuperer les transitions du batch du buffer
        transitions, indices, compensations = buffer.tirage()
        #on feed les transitions
        states=np.array([transition.state for transition in transitions],dtype=np.float32)
        feed_state = self.forward(torch.tensor(states))
        #stockage des valeur de Q pour chaque transitions
        q_value=[]
        for i in range(len(transitions)) : 
            q_value.append(feed_state[i,transitions[i].action])
        #on transforme la liste de tensor(Q value) en un tenseur de Q value
        q_value=torch.stack(q_value)
        #calcul des target pour chaque transitions
        with torch.no_grad():
            tab_transi=torch.tensor([transition.prochaine_etat for transition in transitions])
            next_actions = self.forward(tab_transi).argmax(dim=1)
            q_values_goal_net=goal_net.forward(tab_transi)
            next_max_q_value =q_values_goal_net[torch.arange(len(transitions)),next_actions]
            targets = [transitions[i].reward+(1-transitions[i].done)*next_max_q_value[i]*gamma**self.n
                       for i in range(len(transitions))]
            targets=torch.stack(targets).float()
        self.backprop(targets,q_value,compensations)
        with torch.no_grad():
            td_errors = (targets - q_value)
            buffer.update(indices, td_errors)
class Agent():
    def __init__(self, net,goal_net,buffer):
        self.net=net 
        self.goal_net=goal_net
        self.buffer=buffer
    def fusionner_poids(self):
        self.goal_net.load_state_dict(self.net.state_dict())
    def fusion_douce_poids(self):
        tau=0.005
        with torch.no_grad():
            for poid_target, poid_actuel in zip(self.goal_net.parameters(), self.net.parameters()):
                poid_target.data.copy_(tau * poid_actuel.data + (1.0 - tau) * poid_target.data)
    def train(self):
        self.net.train_batches(self.buffer,self.goal_net)
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


n = 1
net=Net(n)
goal_net=Net(n)
buffer=Buffer(capacity=2000,alpha=0.6,batch_size=64,beta=0.4)
player=Agent(net,goal_net,buffer)
player.fusionner_poids()
env=gym.make("CartPole-v1")
total_reward=0
liste_total_reward=[]
sauvegarde_etats_reward=[]
sauvegarde_transi=[]
gamma=0.9
for iter in range(100):
    if(iter!=0 and iter%20==0):
        liste_total_reward.append(total_reward)
        print(f"iter : {iter} score : {total_reward/20} eps : {epsilon} taille buffer : {player.buffer.sumtree.taille_actuelle}")
        total_reward=0
    etat,_=env.reset()
    
    done = False 
    while not done:
        with torch.no_grad():
            forw = player.net.forward(torch.tensor(etat,dtype=torch.float32))
            #.item transforme [x] en x
            action = torch.argmax(forw).item()
        etat_suivant, recompense, terminated, truncated, _ = env.step(action)
        done=terminated or truncated
        transi = Transition(etat,action,0,None,None)
        sauvegarde_transi.append(transi)
        sauvegarde_etats_reward.append((etat,recompense))
        total_reward+=recompense
        player.train()
        player.net.reset_bruit()
        etat=etat_suivant
        nb_step_game=len(sauvegarde_transi)
    for i in range(nb_step_game):
        if i >= nb_step_game-n : 
            sauvegarde_transi[i].done=True
            sauvegarde_transi[i].prochaine_etat=sauvegarde_etats_reward[-1][0]
        else:
            sauvegarde_transi[i].done=False
            sauvegarde_transi[i].prochaine_etat=sauvegarde_etats_reward[i+n][0]
        for j in range(min(n,nb_step_game-i)):
            sauvegarde_transi[i].reward+=(gamma**j)*sauvegarde_etats_reward[i+j][1]
        player.buffer.add(sauvegarde_transi[i])
    sauvegarde_transi = []
    sauvegarde_etats_reward = []
plt.figure(figsize=(12,6))
plt.plot(liste_total_reward)
plt.show()
env=gym.make("CartPole-v1",render_mode="human")
player.net.supprimer_bruit()
for i in range(3):
    etat,_=env.reset()
    done = False 
    points_episode=0
    while not done:
        forw = player.net.forward(torch.tensor(etat))
        action = torch.argmax(forw).item()
        etat_suivant, recompense, terminated, truncated, info = env.step(action)
        done=terminated or truncated 
        tenseur_etat=torch.tensor(etat,dtype=torch.float32)
        #on force le dtype car etat est un np array donc en float64 et les poids sont en float32
        tenseur_prochain_etat=torch.tensor(etat_suivant,dtype=torch.float32)
        etat=etat_suivant
        points_episode+=recompense
    print(f"points episodes : {points_episode}\n")
    points_episode=0
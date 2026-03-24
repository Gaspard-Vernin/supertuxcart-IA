import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import sys
import signal
import time
import math

#pour éviter l'erreur EOF
import os

os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
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
class Transi:
    def __init__(self,etat,action,reward,done,etat_plus_n):
        self.etat=etat 
        self.action=action 
        self.reward=reward 
        self.done_plus_n=done 
        self.etat_plus_n = etat_plus_n
    def __eq__(self, t1):
        assert(isinstance(t1,Transi))
        return (self.etat==t1.etat) and (self.action==t1.action) and (self.reward==t1.reward)  and (self.done==t1.done) and (self.etat_plus_n==t1.etat_plus_n)
class Sumtree:
    def __init__(self,taille):
        self.taille=taille 
        #un arbre parfait qui a x feuilles à 2x-1 noeuds 
        self.tree = np.zeros(taille*2-1)
        #data[i] représente la transition indexée par i
        self.data = np.empty(taille,dtype=Transi)
        #pointeur pointe sur la prochaine case sur laquelle on va écrire
        self.pointeur=0
        self.taille_actuele=0
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
        if(self.taille_actuele<self.taille):
            self.taille_actuele+=1
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
        """renvoie (transition,indice_data,valeur_du_noeud) en fonction de la priorité donnée"""
        i=0
        while True:
            #si c'est une feuille
            if(i>=self.taille-1):
                indice_data=self.i_arbre_to_i_data(i)
                return (self.data[indice_data],indice_data,self.tree[i])
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
        priorities=[tirage[i][2] for i in range(self.batch_size)]
        #on calcule la compensation à ajouter aux TD_errors
        priorités_normalisées = torch.tensor(priorities)/self.sumtree.total()
        compensations = torch.pow(self.sumtree.taille_actuele*priorités_normalisées,-self.beta)
        #on normalise la compensation pour plus de stabilité, N peut être très grand...
        compensations/=compensations.max()
        return transitions,indices,torch.tensor(compensations)
    def update(self,indices,td_errors):
        td_errors=td_errors.detach().cpu().numpy()
        for indice,td_error in zip(indices,td_errors):
            poid=abs(td_error)**self.alpha+1e-5
            self.sumtree.update(self.sumtree.i_data_to_i_arbre(indice),poid)
class Dueling_network(nn.Module):
    def __init__(self,taille_etat,nb_actions,lr):
        super().__init__()
        #!!! si on ajoute des couches ne pas oublier de changer reset_bruit !!!
        self.fc1=Noisy_linear_layer(taille_etat,64)
        self.fc3v=Noisy_linear_layer(64,64)
        self.fcV=Noisy_linear_layer(64,1)
        self.fc3a=Noisy_linear_layer(64,64)
        self.fcA=Noisy_linear_layer(64,nb_actions)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
    def forward(self,input):
        """effectue le feedforward en utilisant le dueling"""
        # Si l'input n'a qu'une dimension on ajoute la dimension 
        if input.dim() == 1:
            input = input.unsqueeze(0) # Transforme (8) en (1, 8)
        x=torch.relu(self.fc1.forward(input))
        v=torch.relu(self.fc3v.forward(x))
        v=self.fcV(v)
        x=torch.relu(self.fc3a.forward(x))
        a=self.fcA(x)
        #a représente a quel point l'état est intéressant et v donne a quel point une action est bonne
        #on lui soustrait sa moyenne (dim=1 car il est de taille batchs_sizexnb_actions et qu'on
        #veut le moyenne sur les actions. Ceci nous renvoie un vecteur de taille (64,0) or on veut un 
        #truc de la taille de v, donc on met keepdim a True
        return v+(a-a.mean(dim=1,keepdim=True))
    def reset_bruit(self):
        self.fc1.reset_bruit()
        self.fc3v.reset_bruit()
        self.fcV.reset_bruit()
        self.fc3a.reset_bruit()
        self.fcA.reset_bruit()
class Agent:
    def __init__(self,state_size,action_size,buffer_capacity,batch_size,alpha,beta,eps,gamma,lr,alphaupdate,tau):
        self.net = Dueling_network(state_size,action_size,lr)
        self.goal_net = Dueling_network(state_size,action_size,lr)
        #on copie les données de net dans goal_net
        self.goal_net.load_state_dict(self.net.state_dict())
        self.buffer = Buffer(buffer_capacity,alpha,batch_size,beta)
        self.eps=eps 
        self.gamma=gamma 
        self.lr=lr 
        self.nb_train=0
        self.alphaupdate=alphaupdate
        self.tau=tau
    def train(self):
        if self.buffer.sumtree.taille_actuele<self.buffer.batch_size : 
            return
        """entraine le réseau sur un minibatch extrait du buffer"""
        self.net.reset_bruit()
        self.goal_net.reset_bruit()
        transitions,indices,compensations=self.buffer.tirage()
        """Plan : 
            1-forward les actions sur etats
            2-caculer la target avec le goalnet
            3-backprop
            4-update les values
        """
        # 1-forward les actions sur etats
        etats = torch.tensor(np.array([transitions[i].etat for i in range(self.buffer.batch_size)]))
        actions= torch.tensor(np.array([transitions[i].action for i in range(self.buffer.batch_size)]))
        forws = self.net.forward(etats)
        q_values = forws[torch.arange(self.buffer.batch_size),actions]
        # 2-caculer la target avec le goalnet
        #pour calculer les targets, on ne veut pas que les gradients soient modifiés
        with torch.no_grad():
            #on calcule les actions suivantes selon le net actuel mais on calcule leur q_values avec le goalnet pour plus de stabilité
            etats_n_step_suivant= torch.tensor(np.array([transitions[i].etat_plus_n for i in range(self.buffer.batch_size)]))
            forws_suivant = self.net.forward(etats_n_step_suivant)
            #forws_suivant de dim batch_sizexactions donc on veut argmax selon les actions
            next_actions = forws_suivant.argmax(dim=1)
            #on calcule les q values sur le goal net selon les actions choisies par le net normal
            forws_suivant_goal = self.goal_net.forward(etats_n_step_suivant)
            q_values_max_goal = forws_suivant_goal[torch.arange(self.buffer.batch_size),next_actions]
            #on vectorise tout pour calculer les target plus vite
            rewards=torch.tensor(np.array([transitions[i].reward for i in range(self.buffer.batch_size)]))
            dones=torch.tensor(np.array([transitions[i].done_plus_n for i in range(self.buffer.batch_size)]),dtype=torch.float32)
            targets = rewards+(self.gamma**n_val)*(1-dones)*q_values_max_goal
        td_errors = targets-q_values 
        # 3-backprop
        loss = (compensations * torch.nn.functional.smooth_l1_loss(q_values, targets, reduction='none')).mean()
        self.net.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),10)
        self.net.optimizer.step()
        # 4-update les values
        self.buffer.update(indices,td_errors)
        with torch.no_grad():
            for target_param, local_param in zip(self.goal_net.parameters(), self.net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        self.nb_train+=1
def map_action(pas_steer,pas_accel):
    return [(0,), (1,)]
def creer_action(a,*args):
    """CartPole attend directement un int"""
    return int(a)
def num_action(action):
    global tab_map_action
    if isinstance(action, int):
        return action
    if isinstance(action, tuple):
        return action[0]
    return int(action)
def generer_vector_etat(etat):
    """Etat déjà vectoriel pour CartPole"""
    return torch.tensor(etat, dtype=torch.float32)
def convert_random_action_to_legal_action(random_action):
    return random_action
def norme(vecteur):
    return math.sqrt(vecteur[0]**2+vecteur[1]**2+vecteur[2]**2)
def signal_handler(sig, frame):
    global env
    if env is not None:
        try:
            print("Environnement fermé avec succès.")
        except Exception as e:
            print(f"Erreur lors de la fermeture (normale avec Ctrl+C): {e}")
    sys.exit(0)
    """
def def_reward(distance_parcourue,dist_centre,norme_vitesse,last_distance_parcourue,last_energie,energie,drift,skeed,point1,point2):
    global last_distance,obj1x,obj1z,is_obj1_boost,obj2x,obj2z,is_obj2_boost,obj3x,obj3z,obj3t,bananex,bananez,is_banane_a_banane

    recompense_boost=0
    if(energie-last_energie>0):
        #plus on prend un gros boost plus la reward est grosse
        recompense_boost = 20*(energie-last_energie)


    reward_banane=0
    if(math.sqrt(bananex**2+bananez**2)<0.13 and (bananex!=0)):
        print("banane")
        #reward fix parceque jsp trop comment la fixer, si ca marche pas je change
        reward_banane = -200

    reward_drift=0
    angle_a_tourner=math.atan2(point2[0]-point1[0],point2[2]-point1[2])
    print("angle_a_tourner : ",angle_a_tourner,"\n")
    if drift==1 :
        if abs(angle_a_tourner)>math.pi/5:
            #si skeed est grand ie si on a bcp déraper on veut prendre un max de boost donc on continue jusqua la fin du virage
            reward_drift=(1+2*skeed)* abs(angle_a_tourner-math.pi/5)
        else:
            #on dérape sans être dans un virage
            reward_drift=-60
    print(f"boost:{recompense_boost}---banane:{reward_banane}---drift:{reward_drift}\nvitesse:{norme_vitesse}---dist_centre:{-30*abs(dist_centre)}---delta_dist:{100*(distance_parcourue-last_distance_parcourue)}")
    reward= (recompense_boost+reward_banane+reward_drift-30*abs(dist_centre)+100*(distance_parcourue-last_distance_parcourue))
    print("reward totale = ",reward)
    return reward/100
    """
def def_reward(distance_parcourue, dist_centre, norme_vitesse, last_distance_parcourue,
               last_energie, energie, drift, skeed, point1, point2):
    
    # ── 1. Signal principal : progression sur le circuit ──────────────────────
    # C'est le signal le plus important et le plus dense
    delta_distance = distance_parcourue - last_distance_parcourue
    # On clip pour éviter les téléportations/resets qui faussent le signal
    delta_distance = max(-5.0, min(5.0, delta_distance))
    reward_progression = 10 * delta_distance

    # ── 2. Rester au centre de la piste ───────────────────────────────────────
    # Quadratique : tolérant au centre, sévère sur les bords
    # dist_centre est en mètres, typiquement [-largeur/2, largeur/2]
    reward_centre = -0.3 * (dist_centre ** 2)

    # ── 3. Maintenir une bonne vitesse ────────────────────────────────────────
    # Encourage à accélérer, plafonné pour ne pas dominer
    # norme_vitesse max experimentale ~23
    reward_vitesse = 2*min(norme_vitesse, 20.0)

    # ── 4. Ramasser des boosts ────────────────────────────────────────────────
    # Signal sparse mais important, on le garde car très lisible
    delta_energie = energie - last_energie
    reward_boost = 3.0 * max(0.0, delta_energie)   # seulement si on gagne de l'énergie

    # ── 5. Éviter les bananes ─────────────────────────────────────────────────
    # On punit si la banane est très proche (bananex/z normalisés par 40)
    # distance < 0.15 ≈ 6m en vrai
    dist_banane = math.sqrt(bananex**2 + bananez**2)
    reward_banane = -2.0 if (dist_banane < 0.15 and is_banane_a_banane == 1) else 0.0

    # ── Assemblage ────────────────────────────────────────────────────────────
    reward = reward_progression + reward_centre + reward_vitesse + reward_boost + reward_banane

    # Debug lisible
    print(f"prog:{reward_progression:+.2f} | centre:{reward_centre:+.2f} | "
          f"vitesse:{reward_vitesse:+.2f} | boost:{reward_boost:+.2f} | "
          f"banane:{reward_banane:+.2f} | TOTAL:{reward:+.2f}")

    # On normalise légèrement pour que les gradients restent stables
    return reward / 10.0
# Enregistrez le gestionnaire de signal
signal.signal(signal.SIGINT, signal_handler)
taille_input=0
tab_map_action = map_action(3,3)
temps_derapage = 0
mode_vision = True
drift_fini = False
last_distance = 0
last_energie=0
energie=0
is_visualisation = False
last_distance,obj1x,obj1z,is_obj1_boost,obj2x,obj2z,is_obj2_boost,obj3x,obj3z,is_obj3_boost,bananex,bananez,is_banane_a_banane=0,0,0,0,0,0,0,0,0,0,0,0,0
if __name__=="__main__":
    #initialisation des variables et classes
    env_princ = gym.make("CartPole-v1")
    env_visu = gym.make("CartPole-v1", render_mode="human")
    n_val=3
    env = env_princ
    etat,_ = env.reset()
    taille_input = len(generer_vector_etat(etat))
    agent = Agent(
        taille_input,
        len(tab_map_action),
        16384,
        128,
        0.6,
        0.4,
        0.99,
        0.99,
        1e-4,
        0.5,
        0.01
    )
    total_reward=0
    liste_reward=[]
    freq_print=10
    save_mode=False

    for iter in range(0,2000):
        

        #on decay le beta qui sert a pondérer le biais de sélection dans le PER
        agent.buffer.beta=min(1,1.001*agent.buffer.beta)

        if iter % 50 == 0 and iter != 0:
            env = env_visu
        else:
            env = env_princ
        #reinitialisation du bruit pour cette itérations
        agent.net.reset_bruit()
        done=False
        

        if(iter%freq_print==0):
            print(f"iter : {iter}, reward : {total_reward/freq_print}")
            liste_reward.append(total_reward/freq_print)
            total_reward=0
            if save_mode:
                nom_fichier = f"save_iter_{iter}.pth"
                torch.save(agent.goal_net.state_dict(), nom_fichier)
            
        nbframe=0
        sauvegarde_etats_reward=[]
        sauvegarde_transi=[]
        etat,_ = env.reset()
        while not done:
            nbframe+=1
            #on choisit l'action
            etat_forw=generer_vector_etat(etat)
            with torch.no_grad():
                forw=agent.net.forward(etat_forw)
            action=torch.argmax(forw).item()
            action_tuple = tab_map_action[action]
            action = creer_action(action_tuple[0])
            
            #on joue l'action
            etat_suivant,reward,terminated,truncated,_=env.step(action)

            #on récupere les infos, on crée la transi et on save
            vector_etat = generer_vector_etat(etat)
            done = terminated or truncated

            transi = Transi(vector_etat,num_action(action),0,None,None)
            sauvegarde_etats_reward.append((vector_etat,reward))
            sauvegarde_transi.append(transi)
            total_reward+=reward
            #on entraine le réseau et on passe a l'état suivant
            agent.train()
            etat=etat_suivant
        
        #le dernier code ne traite pas le dernier etat, on le fait ici
        vector_etat_final = generer_vector_etat(etat_suivant)
        sauvegarde_etats_reward.append((vector_etat_final, 0))
        #on calcule la vrai reward sur les n_val premiers etats a chaque fois 
        nb_step_game=len(sauvegarde_transi)
        for i in range(nb_step_game):
            if i >= nb_step_game-n_val:
                sauvegarde_transi[i].done_plus_n=True
                sauvegarde_transi[i].etat_plus_n=sauvegarde_etats_reward[-1][0]
            else:
                sauvegarde_transi[i].done_plus_n=False
                sauvegarde_transi[i].etat_plus_n=sauvegarde_etats_reward[i+n_val][0]
            for j in range(min(n_val,nb_step_game-i)):
                sauvegarde_transi[i].reward += (agent.gamma**j)*sauvegarde_etats_reward[i+j][1]
            #on rajoute la transition au buffer
            agent.buffer.add(sauvegarde_transi[i])

    plt.plot(liste_reward)
    plt.show()
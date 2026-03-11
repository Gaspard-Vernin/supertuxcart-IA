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
n_val=3
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
class Transi:
    def __init__(self,etat,action,reward,done,etat_plus_n):
        self.etat=etat 
        self.action=action 
        self.reward=reward 
        self.done_plus_n=done 
        self.etat_plus_n = etat_plus_n
    def __eq__(self, t1):
        assert(isinstance(t1,Transi))
        return (self.etat==t1.etat) and (self.action==t1.action) and (self.reward==t1.reward) and (self.done==t1.done) and (self.etat_plus_n==t1.etat_plus_n)
class Sumtree:
    def __init__(self,taille):
        self.taille=taille 
        #un arbre parfait qui a x feuilles à 2x-1 noeuds 
        self.tree = np.zeros(taille*2-1)
        #data[i] représente la transition indexée par i
        self.data = np.zeros(taille,dtype=Transi)
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
        version claire de la version vectoriése ci dessous
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
        priorités_normalisées = torch.tensor(priorités)/self.sumtree.total()
        compensations = torch.pow(self.sumtree.taille_actuele*priorités_normalisées,-self.beta)
        #on normalise la compensation pour plus de stabilité, N peut être très grand...
        compensations/=compensations.max()
        return transitions,indices,compensations
    def update(self,indices,td_errors):
        td_errors=td_errors.detach().cpu().numpy()
        for indice,td_error in zip(indices,td_errors):
            poid=abs(td_error)**self.alpha+1e-5
            self.sumtree.update(self.sumtree.i_data_to_i_arbre(indice),poid)
class Dueling_network(nn.Module):
    def __init__(self,taille_etat,nb_actions,lr):
        super(Dueling_network,self).__init__()
        self.fc1=nn.Linear(taille_etat,512)
        self.fc2=nn.Linear(512,256)
        self.fc3v=nn.Linear(256,256)
        self.fcV=nn.Linear(256,1)
        self.fc3a=nn.Linear(256,256)
        self.fcA=nn.Linear(256,nb_actions)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
    def forward(self,input):
        """effectue le feedforward en utilisant le dueling"""
        # Si l'input n'a qu'une dimension on ajoute la dimension 
        if input.dim() == 1:
            input = input.unsqueeze(0) # Transforme (8) en (1, 8)
        x=torch.relu(self.fc1(input))
        x=torch.relu(self.fc2(x))
        v=torch.relu(self.fc3v(x))
        v=self.fcV(v)
        x=torch.relu(self.fc3a(x))
        a=self.fcA(x)
        #a représente a quel point l'état est intéressant et v donne a quel point une action est bonne
        #on lui soustrait sa moyenne (dim=1 car il est de taille batchs_sizexnb_actions et qu'on
        #veut le moyenne sur les actions. Ceci nous renvoie un vecteur de taille (64,0) or on veut un 
        #truc de la taille de v, donc on met keepdim a True
        return v+(a-a.mean(dim=1,keepdim=True))
class Agent:
    def __init__(self,state_size,action_size,buffer_capacity,batch_size,alpha,beta,eps,gamma,lr,alphaupdate,tau,temperature):
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
        self.temperature = temperature
    def get_action_boltzman(self,etat):
        etat_forw = generer_vector_etat(etat)
        with torch.no_grad():
            q_values = self.net.forward(etat_forw).squeeze()
        q_values = q_values.cpu().numpy()
        exp_q_values = np.exp((q_values-q_values.max())/self.temperature)
        probs = exp_q_values/exp_q_values.sum()
        return np.random.choice(len(probs),p=probs)
    def train(self):
        if self.buffer.sumtree.taille_actuele<self.buffer.batch_size : 
            return
        """entraine le réseau sur un minibatch extrait du buffer"""
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
        loss = (compensations * F.smooth_l1_loss(q_values, self.alphaupdate*targets, reduction='none')).mean()
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
    """cette fonction prend en entier deux int representant le nombre de segments par lesquels on découpe le steer et l'accel
        et renvoie une liste (accel,steer,derapage) contenant le produit carthesien des valeur possibles d'accel de steer et de derapage"""
    steer = [round((i+1)/pas_steer,2) for i in range((-pas_steer-1),(pas_steer))]
    accel = [round((i+1)/pas_accel,2) for i in range(pas_accel)]
    nitro = [0,1]
    fire = [0,1]
    drift=[0,1]
    action_map = []
    for s in steer : 
        for a in accel:
            for n in nitro:
                for f in fire:
                    for d in drift:
                        action_map.append((a,s,n,f,d))
    return action_map
def creer_action(accel,steer,nitro,fire,drift):
    """renvoie une action formatée sous la forme d'un dictionnaire avec que accel et steer"""
    return {
    'acceleration': np.array([accel],dtype=np.float32),
    'steer': np.array([steer],dtype=np.float32),
    'brake': np.int64(0), 
    'drift': np.int64(drift), 
    'fire': np.int64(fire),  
    'nitro': np.int64(nitro), 
    'rescue': np.int64(0) 
    }
def num_action(action):
    """renvoie le numéro associé à l'action donnée"""
    global tab_map_action
    #on commence par convertir l'action en une action valide si elle a été tirée aléatoirement

    t = (round(action["acceleration"][0],2),round(action["steer"][0],2),action["nitro"],action["fire"],action["drift"])
    t = convert_random_action_to_legal_action(t)
    for i in range(len(tab_map_action)):
        if(tab_map_action[i]==t):
            return i 
    raise Exception("l'action n'est pas dans les actions possibles")
def generer_vector_etat(etat):
    """génerer un vecteur avec les infos d'état jugées interessantes pour cette version"""
    global obj1x,obj1z,is_obj1_boost,obj2x,obj2z,is_obj2_boost,obj3x,obj3z,is_obj3_boost,bananex,bananez,is_banane_a_banane
    center_path = etat["center_path"]
    center_path_distance = etat["center_path_distance"]
    velocity = etat["velocity"]
    chemin_a_suivre = etat["paths_start"]
    point_tres_proche = chemin_a_suivre[0]
    point_proche = chemin_a_suivre[1]
    point_moyen_distance = chemin_a_suivre[2]
    point_loin = chemin_a_suivre[3]
    point_tres_loin = chemin_a_suivre[4]
    energy = etat["energy"][0]
    charge_drift = etat["skeed_factor"][0]
    #on stocke la position de la banane/chewingum la plus proche et des trois items les plus proches
    #si pas assez d'items proche ou bien items a plus de 40m selon z, on met le type a 0, et x = z = 1
    #autrement, on normalise par 80 selon z et par 30 selon x comme ca on est loin de 1 et l'IA peut bien
    #faire la différence entre une vraie distance et une distance 1,1.
    indices_items=[]
    indice_banane=-1
    for i in range(len(etat["items_type"])):
        if etat['items_type'][i] in (0,2,3) and len(indices_items)<3 and 0<etat['items_position'][i][2]<80:
            indices_items.append(i)
        elif etat['items_type'][i] in (1,4) and indice_banane==-1 and etat['items_position'][i][2]<80:
            indice_banane=i
    #si on a pas 3 nitros a assez proches : 
    match len(indices_items):
        #si on a rien a portée
        case 0 : 
            obj1x=0
            obj1z=1
            is_obj1_boost=0
            obj2x=0
            obj2z=1
            is_obj2_boost=0
            obj3x=0
            obj3z=1
            is_obj3_boost=0
        case 1 :
            obj1x=etat['items_position'][indices_items[0]][0]/40
            obj1z=etat['items_position'][indices_items[0]][2]/40
            is_obj1_boost=1
            obj2x=0
            obj2z=1
            is_obj2_boost=0
            obj3x=0
            obj3z=1
            is_obj3_boost=0
        case 2 :
            obj1x=etat['items_position'][indices_items[0]][0]/40
            obj1z=etat['items_position'][indices_items[0]][2]/40
            is_obj1_boost=1
            obj2x=etat['items_position'][indices_items[1]][0]/40
            obj2z=etat['items_position'][indices_items[1]][2]/40
            is_obj2_boost=1
            obj3x=0
            obj3z=1
            is_obj3_boost=0
        case 3 :
            obj1x=etat['items_position'][indices_items[0]][0]/40
            obj1z=etat['items_position'][indices_items[0]][2]/40
            is_obj1_boost=1
            obj2x=etat['items_position'][indices_items[1]][0]/40
            obj2z=etat['items_position'][indices_items[1]][2]/40
            is_obj2_boost=1
            obj3x=etat['items_position'][indices_items[2]][0]/40
            obj3z=etat['items_position'][indices_items[2]][2]/40
            is_obj3_boost=1
        case _ :
            raise Exception("la taille de indice_items a dépassée 3")
    if indice_banane == -1 :
        bananex=0
        bananez=1
        is_banane_a_banane=0
    else : 
        bananex=etat['items_position'][indice_banane][0]/40
        bananez=etat['items_position'][indice_banane][2]/40
        is_banane_a_banane=1
    #experimentalement, on mesure max_norme_velocity = 23, max_center_path[2]=10
    #les points loin selon z sont à moins de 40 mètres experimentalement,
    #les points loin selon y sont a moins de 4 mètres experimentalement,
    #les points loin selon x sont a moins de 3 mètres experimentalement

    return torch.from_numpy(np.array([center_path[0].item()/180,center_path[2].item()/180,
                         center_path_distance[0]/180,
                         velocity[0]/40,velocity[2]/40,
                         point_tres_proche[0]/300,point_tres_proche[2]/300,
                         point_proche[0]/300,point_proche[2]/300,
                         point_moyen_distance[0]/300,point_moyen_distance[2]/300,
                         point_loin[0]/300,point_loin[2]/300,
                         energy/10,
                         charge_drift,
                         obj1x,obj1z,is_obj1_boost,
                         obj2x,obj2z,is_obj2_boost,
                         obj3x,obj3z,is_obj3_boost,
                         bananex,bananez,is_banane_a_banane
                        ],dtype = np.float32))
def convert_random_action_to_legal_action(random_action):
    """renvoie un tuple (accel,steer) correspondant au tuple le plus proche de random_action dans tab_map_action"""
    global tab_map_action
    #comme tab_map_action est un produit carthésien de l'ensemble des accel par l'ensemble des steer, le point le plus proche
    #de random_action dans tab_map_action est le plus proche coordonnées par coordonnées, donc on calcule juste le min de la
    #somme des distances selon les deux coordonnées
    return min(tab_map_action, key = lambda couple : abs(couple[0]-random_action[0])+abs(couple[1]-random_action[1])+abs(couple[2]-random_action[2])+abs(couple[3]-random_action[3])+abs(couple[4]-random_action[4]))
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

def def_reward(distance_parcourue,dist_centre,norme_vitesse,last_distance_parcourue,last_energie,energie,drift,skeed,x,z):
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
    angle_a_tourner=math.atan2(x,z)
    if drift==1 :
        if abs(angle_a_tourner)>math.pi/4:
            #si skeed est grand ie si on a bcp déraper on veut prendre un max de boost donc on continue jusqua la fin du virage
            reward_drift=(1+2*skeed)**2 * 10
        else:
            #on dérape sans être dans un virage
            reward_drift=-30
    print(f"boost:{recompense_boost}---banane:{reward_banane}---drift:{reward_drift}\nvitesse:{norme_vitesse}---dist_centre:{-30*abs(dist_centre)}---delta_dist:{100*(distance_parcourue-last_distance_parcourue)}")
    reward= (recompense_boost+reward_banane+reward_drift-10*abs(dist_centre)+100*(distance_parcourue-last_distance_parcourue))
    print("reward totale = ",reward)
    return reward/100

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
    #on récupère la taille de l'input en testant la sortie de la fonction generer vector etat
    env=gym.make("supertuxkart/simple-v0" ,num_kart=2,max_episode_steps=10000)
    etat_test = env.observation_space.sample()
    taille_input = len(generer_vector_etat(etat_test))
    agent = Agent(taille_input,len(tab_map_action),16384,128,0.6,0.4,0.99,0.999,1e-4,0.5,0.01,2.5)
    total_reward=0
    liste_reward=[]
    liste_distances=[]
    total_distance=0
    best_total_reward = -100
    last_distance_parcourue = 0
    print("len de tabmap = ",len(tab_map_action))
    compteur_nb_ajout = 0
    for iter in range(0,5000):
        agent.buffer.beta=min(1,1.001*agent.buffer.beta)
        last_distance_parcourue=0
        print("iter = ",iter,"\n")
        if mode_vision:
            if(iter%50 == 0 and iter!=0):
                env.close()
                env=gym.make("supertuxkart/simple-v0" ,num_kart=2,max_episode_steps=10000,render_mode="human")
                is_visualisation = True
            if((iter-1)%50==0):
                env.close()
                env=gym.make("supertuxkart/simple-v0",num_kart=2)
                is_visualisation = False
        etat,_=env.reset()
        done=False 
        temps_derapage = 0
        if(iter%100==0):
            print(f"iter : {iter}, reward : {total_reward/100}, epsilon : {agent.eps}")
            liste_reward.append(total_reward/100)
            liste_distances.append(total_distance/100)
            nom_fichier = f"save_iter_{iter}.pth"
            torch.save(agent.goal_net.state_dict(), nom_fichier)
            total_reward=0 
            total_distance=0
        compteur_nb_ajout=0
        compteur_pas_assez_de_vitesse = 0
        seuil_vitesse = 0.5
        compteur_trop_loin = 0
        # Avant la boucle while not done:
        nbframe=0
        sauvegarde_etats_reward=[]
        sauvegarde_transi=[]
        while not done:
            nbframe+=1
            compteur_nb_ajout+=1
            #on tire une action 
            if is_visualisation: 
                #on forward l'état
                etat_forw=generer_vector_etat(etat)
                forw=agent.net.forward(etat_forw)
                #on renvoie l'action avec la plus grosse Q_value
                action=torch.argmax(forw).item()
                action_tuple = tab_map_action[action]
                action = creer_action(action_tuple[0],action_tuple[1],action_tuple[2],action_tuple[3],action_tuple[4])
            else:
                action=agent.get_action_boltzman(etat)
                action_tuple = tab_map_action[action]
                action = creer_action(action_tuple[0],action_tuple[1],action_tuple[2],action_tuple[3],action_tuple[4])
            etat_suivant,reward,terminated,truncated,_=env.step(action)
            
            largeur_chemin = etat_suivant["paths_width"][0][0]
            if(abs(etat_suivant["center_path_distance"][0])>largeur_chemin):
                compteur_trop_loin+=1
            else : 
                compteur_trop_loin = max(0,compteur_trop_loin-1)
            if(compteur_trop_loin>=10):
                truncated=True 
                reward-=500
                print("run terminée car trop loin")
            if(norme(etat_suivant["velocity"])<seuil_vitesse):
                compteur_pas_assez_de_vitesse+=1
            else : 
                #on le baisse progressivement pour repartir plus vite si on a avancer despi mais on est tjr coincés
                compteur_pas_assez_de_vitesse = max(0,compteur_pas_assez_de_vitesse-1)
            if(compteur_pas_assez_de_vitesse>=60):
                truncated = True 
                reward -= 500
                print("run terminée car trop lent")
            vector_etat = generer_vector_etat(etat)
            drift = 1 if(action["drift"]==1) else 0
            energie = etat['energy'][0]
            reward = def_reward(distance_parcourue = etat_suivant["distance_down_track"][0],
                                dist_centre = etat_suivant["center_path_distance"][0] , 
                                norme_vitesse=norme(etat_suivant["velocity"]),last_distance_parcourue=last_distance_parcourue,
                                last_energie=last_energie,energie=energie,
                                drift=drift,skeed=etat_suivant["skeed_factor"][0],x=etat_suivant["paths_start"][1][0],
                                z=etat_suivant["paths_start"][1][2])
            last_energie=energie
            last_distance_parcourue = etat_suivant["distance_down_track"][0]
            #la partie est finie si on a gagné ou si on est sorti
            done = terminated or truncated
            transi = Transi(vector_etat,num_action(action),0,None,None)
            #print(etat_suivant["paths_start"],"\n\n\n",etat_suivant["paths_end"],"\n\n\n",etat_suivant["center_path"],"\n\n\n")
            
            #pour le multi step learning, on sauvegarde toutes les transis et couples etat,done et on fait le calcul de reward et l'ajout au buffer a la fin de chaque épisode
            sauvegarde_etats_reward.append((vector_etat,reward))
            sauvegarde_transi.append(transi)
            #agent.buffer.add(transi)
            total_reward+=reward 
            agent.train()
            etat=etat_suivant 
            total_distance+=(etat_suivant["distance_down_track"][0])
            if(done):
                print("distance : ",etat_suivant["distance_down_track"][0],"\n")
        nb_step_game=len(sauvegarde_transi)
        for i in range(nb_step_game):
            if i >= nb_step_game-n_val : 
                sauvegarde_transi[i].done_plus_n=True
            else:
                sauvegarde_transi[i].done_plus_n=False
            for j in range(min(n_val,nb_step_game-i)):
                sauvegarde_transi[i].reward+=(agent.gamma**j)*sauvegarde_etats_reward[i+j][1]
            agent.buffer.add(sauvegarde_transi[i])
        agent.eps=max(0.15,agent.eps*0.9985)
        agent.temperature = max(0.3, agent.temperature*0.995)
        print("nb frame :",nbframe)
    plt.plot(liste_reward)
    plt.show()
    env=gym.make("supertuxkart/simple-v0",track="zengarden",render_mode="human",num_kart=2,max_episode_steps=10000)
    for iter in range(5):
        etat,_=env.reset()
        done=False 
        total_reward=0
        while not done:
            #on forward l'état
            etat_forw=generer_vector_etat(etat)
            forw=agent.net.forward(etat_forw)
            #on renvoie l'action avec la plus grosse Q_value
            action=torch.argmax(forw).item()
            action_tuple = tab_map_action[action]
            action = creer_action(action_tuple[0],action_tuple[1],action_tuple[2])
            #on execute l'action
            etat_suivant,reward,terminated,truncated,_=env.step(action)
            #la partie est finie si on a gagné ou si on est sorti
            done = terminated or truncated
            total_reward+=reward 
            etat=etat_suivant 
        print(f"la reward totale de cette game est de {total_reward}")
        print("distance parcourue : ",etat_suivant["distance_down_track"][0],"\n")

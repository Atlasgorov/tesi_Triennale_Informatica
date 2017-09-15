from networkit import *
from numpy import random
import numpy as np
from numpy import mean,std
import pylab as pl
from math import ceil
import collections
from leader_ActionTable import *

#classe per gestire la propagazione dei vari metodi trattati quali :
#Degree , Closeness, Betweenness Vertex Centrality per nodi
#Betweenness Edge centrality e Spanning edge centrality per gli archi
#Meto di assegnazione random dei nodi iniziali
#Individuazione dei leadere attraverso l'utilizzo di una tabella delle azioni
class Diffusione():
	def __init__(self,Graph = None,mod_flag = True ) :
		self.perc_ICM = [5,10,15,20,25,30,35]
		self.fileName = 'PGPgiantcompo.graph'		#nome del file da prendere in input
		self.num_iteration = 100				#Nunero di iteazioni per ogni metodo per poi fare la media dei risultati ottenuti
		self.range = range(self.num_iteration)
		self.action_table = {}
		self.thresholdNOdes = {}				#Threshold per ogno nodo
		self.edge_to_weight_dict = {} 			#Ogni arco pesato secondo la prazione 1/degree( nodo Target )
		self.node_to_incoming_edge_dict = {}    #dizionario per mappare nodo e nome degli archi che escono verso gli altri nodi adiacenti
		self.activeNodes = {}					#Dizionario contenente i nodi attivi e per ogniuno i suoi vicini ancora non attivi nel momento in cui verrà inserito
		#per ogni metodo tengo un array di nodi cosi da non calcolare ogni volta per tutte le 100 volte gli insiemi da caso, dal momento che una volta fissata la percentuale l'insieme sarà sempre uguale ma cambierà la threshold
		self.bvc = []
		self.cvc = []
		self.dvc = []
		self.leader_nodes = []
		self.random_nodes = []
		self.bec = []
		if Graph == None:
			#generators.ErdosRenyiGenerator(6000,0.005).generate()
			#generators.BarabasiAlbertGenerator(n0= 1e3,k= 20,nMax = 6e3).generate()
			#readGraph('./network/'+self.fileName,Format.METIS,directed=False)
			self.G = readGraph('./network/'+self.fileName,Format.METIS,directed=False)	#Costruisce il grapho non diretto
		else:
			self.G = Graph
		self.G.indexEdges()
		self.n = self.G.numberOfNodes()
		self.E = self.G.edges()
		self.perc = 1/(self.num_iteration)
		self.num_action = 10						#numero di azioni nella tabella delle azioni per il metodo dei leader
		self.mod_flag = mod_flag
		self.numInf = ceil(self.n*self.perc)	#numero di nodi influenzati inizialmente		
		if mod_flag:
			self.check = self.is_activable
		else:
			self.check = self.is_activable1
			#self.numInf = self.perc_ICM[1]
		print('numero nodi ',self.n)
		print('numero archi ',self.G.numberOfEdges())
		self.nodes_list = self.G.nodes()				#lista di tutti i nodi del grafo	
		self.pre_processing()							#inizializza tutte le threshold dei vari nodi del grafo
		
	
	def iniz_x_y(self):
		self.x = []								
		self.y_m = {}
		self.y_sd = {}
		self.y_m['rand'] = []
		self.y_sd['rand'] = []
		self.y_m['leader'] = []
		self.y_sd['leader'] = []
		self.y_m['bc'] = []
		self.y_sd['bc'] = []
		self.y_m['cc'] = []
		self.y_sd['cc'] = []
		self.y_m['dc'] = []
		self.y_sd['dc'] = []
		self.y_m['bec'] = []
		self.y_sd['bec'] = []
		self.y_m['sec'] = []
		self.y_sd['sec'] = []

	def run_main(self,mod_flag):
		#self.print_edge_weight()
		#x e y sono gli array che mi servono per poi visualizzare il grafico alla fine 
		#x contiene i valore del numero di nodi per la percentuale considerata
		#y per ogni metodo contiene il numero di nodi dopo la diffusione del messaggio in media (self.num_iterazioni)
		self.iniz_x_y()
		self.mod_flag = mod_flag
		if self.mod_flag:
			self.check = self.is_activable
			self.modello = 'LTM'
		else:
			self.check = self.is_activable1
			self.modello = 'ICM'
		#Main principale contenente le chimate a tutte le funzioni 
		#for per iterare le percentuali cosi è da 1% a 5%
		for i in range(6):
			self.iterations(i)		#contiene tutte le chiamate alle funzioni per iniziare tutti i vari metodi di individuazione
		#utili per visualizzare grafico finale, dove x (numeodo di nodi iniziali) e y (numero di nodi finali dopo la diffuzione)
		#self.fileName.split('_')[0]
		#'BA rand.'
		#'ER rand.'
		var = self.fileName.split('.')[0]
		#media
		pl.plot(self.x, self.y_m['rand'], 'r', label='scelta random') 
		pl.plot(self.x, self.y_m['leader'], 'g', label='scelta algoritmo leader')
		pl.plot(self.x, self.y_m['bc'], 'b', label='scelta BC')
		#pl.plot(self.x, self.y_m['cc'], 'black', label='scelta CC')
		pl.plot(self.x, self.y_m['dc'], 'orange', label='scelta DC')
		pl.plot(self.x, self.y_m['bec'], 'yellow', label='scelta BEC')
		pl.plot(self.x, self.y_m['sec'], 'violet', label='scelta SEC')
		pl.title('Diffusione del messaggio tramite '+self.modello+', avrg. grafico. Filename '+var+', G=('+str(self.n)+','+str(self.G.numberOfEdges())+')')
		pl.xlabel('nodi influenzati inizialmente')				# Nomi degli assi
		pl.ylabel('nodi influenzati dopo la propagazione')
		pl.xlim(min(self.x),max(self.x))		# Imposta limiti degli assi
		pl.legend()
		pl.show()
		pl.close()
		#standard deviation
		pl.plot(self.x, self.y_sd['rand'], 'r', label='scelta random') 
		pl.plot(self.x, self.y_sd['leader'], 'g', label='scelta algoritmo leader')
		pl.plot(self.x, self.y_sd['bc'], 'b', label='scelta BC')
		#pl.plot(self.x, self.y_sd['cc'], 'black', label='scelta CC')
		pl.plot(self.x, self.y_sd['dc'], 'orange', label='scelta DC')
		pl.plot(self.x, self.y_sd['bec'], 'yellow', label='scelta BEC')
		pl.plot(self.x, self.y_sd['sec'], 'violet', label='scelta SEC')
		pl.title('Diffusione del messaggio tramite '+self.modello+', dev. st. grafico. Filename '+var+', G=('+str(self.n)+','+str(self.G.numberOfEdges())+')')
		pl.xlabel('nodi influenzati inizialmente')				# Nomi degli assi
		pl.ylabel('nodi influenzati dopo la propagazione')
		pl.xlim(min(self.x),max(self.x))		# Imposta limiti degli assi
		pl.legend()
		pl.show()
		pl.close()


	def iterations(self,i):
		if i == 0:
			self.x.append(0)
			for x in self.y_m:
				self.y_m[x].append(0)
				self.y_sd[x].append(0)
		else:
			#if self.mod_flag:
			self.perc = i/(self.num_iteration)		#individua la percentuale dei nodi iniziai
			self.numInf = ceil(self.n*self.perc)  ##self.perc_ICM[i-1]  #numero di nodi influenzati
			#else:
			#	self.numInf = self.perc_ICM[i-1]
			#azzero i precedenti insiemi per tutti i metodi
			self.bvc = []
			self.cvc = []
			self.dvc = []
			self.bec = []
			self.sec = []
			self.leader_nodes = []
			self.random_nodes = []
			self.x.append(self.numInf)
			print('percentuale: ',self.perc)
			#chimata alle funzioni per calcolare la media del numero di nodi finale influenzato per ciascun metodo
			self.average_run_leaders()
			print('leader')
			self.average_run_random()
			print('random')
			self.average_run_centrality_BC()
			print('BC')
			self.average_run_centrality_BEC()
			print('BEC')
			#self.average_run_centrality_CC()
			#print('CC')
			self.average_run_centrality_DC()
			print('DC')
			self.average_run_centrality_SEC()
			print('SEC')
			print('x',self.x)
			print('y_m',self.y_m)

	def average_run_random(self):
		values = []
		for i in self.range:
			self.activeNodes = {}
			self.thresholdNOdes = {}
			self.threshold()
			self.randomInitialNode()
			values.append(self.run())
		self.y_m['rand'].append(mean(values))
		self.y_sd['rand'].append(std(values))
	# per ognuno del seguenti average è semplicemente un ricalcolare per il numero delle iterazioni desiderate il numero di nodi inflenzati,
	# azzerando ogni volta le threshold e l'insieme dei nodi inflenzati inizialmente
	def average_run_centrality_BC(self):
		values = []
		for i in self.range:
			self.activeNodes = {}
			self.thresholdNOdes = {}
			self.threshold()
			self.bcNodes()
			values.append(self.run())
		self.y_m['bc'].append(mean(values))
		self.y_sd['bc'].append(std(values))
		#print('num attivi in media :',num_active/self.num_iteration)
	def average_run_centrality_BEC(self):
		values = []
		for i in self.range:
			self.activeNodes = {}
			self.thresholdNOdes = {}
			self.threshold()
			self.bcEdges()
			values.append(self.run())
		self.y_m['bec'].append(mean(values))
		self.y_sd['bec'].append(std(values))
	def average_run_centrality_SEC(self):
		values = []
		for i in self.range:
			self.activeNodes = {}
			self.thresholdNOdes = {}
			self.threshold()
			self.spanning_edge_centrality()
			values.append(self.run())
		self.y_m['sec'].append(mean(values))
		self.y_sd['sec'].append(std(values))
	def average_run_centrality_CC(self):
		values = []
		for i in self.range:
			self.activeNodes = {}
			self.thresholdNOdes = {}
			self.threshold()
			self.ccNodes()
			values.append(self.run())
		self.y_m['cc'].append(mean(values))
		self.y_sd['cc'].append(std(values))
		#print('num attivi in media :',num_active/self.num_iteration)
	def average_run_centrality_DC(self):
		values = []
		for i in self.range:
			self.activeNodes = {}
			self.thresholdNOdes = {}
			self.threshold()
			self.dcNodes()
			values.append(self.run())
		self.y_m['dc'].append(mean(values))
		self.y_sd['dc'].append(std(values))
		#print('num attivi in media :',num_active/self.num_iteration)
	def average_run_leaders(self):
		values = []
		for i in self.range:
			self.activeNodes = {}
			self.thresholdNOdes = {}
			self.threshold()
			self.leaderNodes()
			values.append(self.run())
		self.y_m['leader'].append(mean(values))
		self.y_sd['leader'].append(std(values))
		#print('num attivi in media :',num_active/self.num_iteration)
	def print_edge_weight(self):
		for ele in self.node_to_incoming_edge_dict:
			for s in self.node_to_incoming_edge_dict[ele]:
				print(str(ele)+' :',s,self.edge_to_weight_dict[s])
	
	#le seguenti due funzioni sono utili per calcolare la tabella delle azioni
	# la prima itera per il numero di azioni desiderato
	def more_action(self):
		for i in range(self.num_action):
			self.run_action_table(i)

	#permette il calcolo di una singola azione individuando inizialmente un numero di nodi iniziali tramite il metodo random 
	#con una percentuale di nodi iniziali del 1% ogni volta
	def run_action_table(self,action):
		self.activeNodes = {}
		self.thresholdNOdes = {}
		self.threshold()
		self.randomInitialNode()
		self.action_table[action] = {}
		self.run_table(action)

	#per ogni nodo calcola una treshold tra [0,1]
	def threshold(self):
		for node in self.nodes_list:
			self.thresholdNOdes[node] = random.uniform(0,1)
			#print('threshold ',node,self.thresholdNOdes[node])
		#print(self.thresholdNOdes)

	# PER LE PROSSIME 7 FUNZIONI IL RAGIONAMENTO è ANALOGO
	# SE L'INSIEME DEI NODI NON è VUOLTO ALLORA ITERO QUESTO INSIEME E INSERISCO TUTTI I NODI TRA I NODI ATTIVI
	# ALTRIMENTI RISPETTO AL METODO CONSIDERATO SI CALCOLA IL METODO, ad esempio il BVC per ogni nodo e 
	# poi li inserisco sia tra i nodi attivi sia nell'insieme del rispettivo metodo cosi da non calcolarlo 
	# poi per tutte le altre rispettive iterazioni che sarebbe identico
	# (se guarda anche solo uo di questi credo capisca cosa intendo)
	def leaderNodes(self):
		if self.leader_nodes != []:
			for i in self.leader_nodes:
				self.insInactiveNodes(i)
		else:
			leaders = CompLeader(self.G,self.mod_flag)
			leader_nodes = list(leaders.leader)
			for node in leader_nodes[:self.numInf]:
				self.leader_nodes.append(node)
				self.insInactiveNodes(node)

	def bcEdges(self):
		for i in self.bec:
			self.insInactiveNodes(i)

	def bcNodes(self):
		#print('self.numInf',self.numInf)
		if self.bvc != []:
			for i in self.bvc:
				self.insInactiveNodes(i)
		else:
			bc = centrality.Betweenness(self.G,computeEdgeCentrality=True)
			bc.run()
			for node in bc.ranking()[:self.numInf]:
				#print(node, node[0])
				self.bvc.append(node[0])
				self.insInactiveNodes(node[0])
			if self.bec == []:
				arr = np.array(bc.edgeScores())
				ind = np.argpartition(arr, -self.numInf*2)[-self.numInf*2:]
				ind = ind[np.argsort(arr[ind])][::-1]
				for i in list(ind):
					if len(self.bec) >= self.numInf:
						break
					e = self.E[i]
					if int(e[0]) not in  self.bec:
						self.bec.append(int(e[0])) 
					if int(e[1]) not in  self.bec:
						self.bec.append(int(e[1]))

	def spanning_edge_centrality(self):
		if self.sec != []:
			for i in self.sec:
				self.insInactiveNodes(i)
		else:
			sec = centrality.SpanningEdgeCentrality(self.G,tol=0.1)
			sec.runParallelApproximation()
			arr = np.array(sec.scores())
			ind = np.argpartition(arr, -self.numInf*2)[-self.numInf*2:]
			ind = ind[np.argsort(arr[ind])][::-1]
			for i in list(ind):
				if len(self.sec) >= self.numInf:
					break
				e = self.E[i]
				if int(e[0]) not in  self.sec:
					self.sec.append(int(e[0])) 
					self.insInactiveNodes(int(e[0]))
				if int(e[1]) not in  self.sec:
					self.sec.append(int(e[1]))
					self.insInactiveNodes(int(e[1]))

	def ccNodes(self):
		#print('self.numInf',self.numInf)
		if self.cvc != []:
			for i in self.cvc:
				self.insInactiveNodes(i)
		else:
			cc = centrality.Closeness(self.G)
			cc.run()

			for node in cc.ranking()[:self.numInf]:
				#print(node, node[0])
				self.cvc.append(node[0])
				self.insInactiveNodes(node[0])

	def dcNodes(self):
		#print('self.numInf',self.numInf)
		if self.dvc != []:
			for i in self.dvc:
				self.insInactiveNodes(i)
		else:
			dc = centrality.DegreeCentrality(self.G)
			dc.run()

			for node in dc.ranking()[:self.numInf]:
				#print(node, node[0])
				self.dvc.append(node[0])
				self.insInactiveNodes(node[0])

	def randomInitialNode(self):
		#Ogni nodo dell'insime random iniziale di size 5% è scelto tra 1 e n (numero dei nodi del grafo)
		#print('self.numInf',self.numInf)
		if self.random_nodes != []:
			for i in self.random_nodes:
				self.insInactiveNodes(i)
		else:
			i = 0
			while i < self.numInf:
				node = random.randint(1,self.n)
				if not node in self.activeNodes:
					self.random_nodes.append(node)
					self.insInactiveNodes(node)
					i += 1

	#inserisce il nodo in activeNode con valore l'array contenente tutti i nodi vicini che fino a quel momento non sono stati ancora attivati
	def insInactiveNodes(self,node):
		self.activeNodes[node] = []
		neighbors = self.G.neighbors(node)
		for neighbor in neighbors:
			if neighbor not in self.activeNodes:
				self.activeNodes[node].append(neighbor)
	#assegna o ogi arco un suo peso b(u,v) con il valore di (percorsi paralleli tra u e v) / (degree di v)
	def pre_processing(self):
		for node in self.nodes_list:
			neighbors_list = self.G.neighbors(node)
			if len(neighbors_list) == 0:
				continue
			edge_weight = 1/float(len(neighbors_list))
			for neighbor in neighbors_list:
				edge_name = str(neighbor)+"TO"+str(node)
				if edge_name in self.edge_to_weight_dict:
					self.edge_to_weight_dict[edge_name] += edge_weight
					print('parallel edge:' ,edge_name)
				else:
					self.edge_to_weight_dict[edge_name] = edge_weight
					self.node_to_incoming_edge_dict.setdefault(neighbor, []).append(edge_name)

	# FUNZIONE RUN scorre tutti i nodi attivi e cerca tramite il metodo LTM
	# di attivare tutti quelli che è possibile restituendo poi in output 
	# il numero di nodi attivati quando non se ne può attivare più nessuno
	def run(self):
		actives = list(self.activeNodes)
		tot = 0
		att = 0
		#print(actives)
		for node in actives:
			#print('\tesamino node attivo',node)
			for inactive in self.activeNodes[node]:
				if inactive not in self.activeNodes:
					#print('esamino nodo inactive',inactive)
					tot += 1
					if self.check(inactive):
						#print('\nTrue')
						att += 1
						self.insInactiveNodes(inactive)
						actives.extend([inactive])
		#print('lunghezza nodi attivi :',len(self.activeNodes))
		print('run',att,'/',tot)
		return len(self.activeNodes)
	# CREA LA TABELLA DELLE AZIONI 
	#inizializzando i nodi di partenza con il tempo 0
	# e poi tramite il modello ltm costrusce la tabella assegnado a ogni nodo un tempo di attivazione
	# questo tempo di attivazione è dato dal time del max nodi vicino attivo più random tra 1 e 10
	# una vota finita la diffusione ordino la tabella in ordine crescente
	def run_table(self,action):
		actives = list(self.activeNodes)
		tot = 0
		att = 0
		for node in actives:
			 self.action_table[action][node] = 0
		for node in actives:
			#print('\tesamino node attivo',node)
			for inactive in self.activeNodes[node]:
				#print('esamino nodo inactive',inactive)
				if inactive not in self.activeNodes:
					tot += 1
					if self.check(inactive):
						att += 1
						self.insInactiveNodes(inactive)
						time = self.time_active(inactive,action)
						self.action_table[action][inactive] = time + random.randint(1,10)
						actives.extend([inactive])
		print('table',att,'/',tot)
		self.action_table[action] = collections.OrderedDict(sorted(self.action_table[action].items(), key=lambda t: t[1], reverse=True ))

	def time_active(self,node,action):
		actives = []
		neighbors_list = self.G.neighbors(node)
		for neighbor in neighbors_list:
				if neighbor in self.activeNodes:
					actives.append(neighbor)
		if random.uniform(0,1.1)> 0.7:
			nodo = random.choice(actives)
			return self.action_table[action][nodo]
		else:
			return self.max_time_active(actives,action)
			

	def max_time_active(self,actives,action):
			max = 0
			for node in actives:
				if max < self.action_table[action][node]:
					max = self.action_table[action][node]
			return max

	#verifica se il nodo passato è attivabile secondo il modello LTM
	def is_activable(self,node):
		#print('verifico se è attivabile')
		total_edge_weight = 0
		incoming_edge_list = self.node_to_incoming_edge_dict.get(node)
		if not incoming_edge_list:
			return False
		for edge in incoming_edge_list:
			w = edge.split("TO")[1]
			if int(w) in self.activeNodes:
				edge_weight = self.edge_to_weight_dict.get(edge)
				#print('preso in considerazione l arco ',edge, 'con peso',edge_weight)
				if not edge_weight:
					continue
				#print('vicino attivo ',w,'peso arco ',edge_weight)
				total_edge_weight += edge_weight
		#print('threshold del nodo :',node,'è',self.thresholdNOdes[node])
		#print('peso dei nodi attivi vicini è ',total_edge_weight)
		if total_edge_weight >= self.thresholdNOdes[node]:
			return True
		return False

	def is_activable1(self,node):
		if random.randint(0,50) < 1:
			return True
		return False

if __name__ == "__main__":
	dif_obj = Diffusione()
	#dif_obj.run_main(True)
	dif_obj.run_main(False)
	'''
	ltm_obj.more_action()
	for action in ltm_obj.action_table:
		print('action',action)
		for k,v in ltm_obj.action_table[action].items():
			print(k,v)
	'''

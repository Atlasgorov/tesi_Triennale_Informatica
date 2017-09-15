from diffusione import  *
import BitVector

class CompLeader:
	def __init__(self,Graph,mod_flag):
		self.dif =Diffusione(Graph,mod_flag)
		self.window = []
		self.influence_vector = {}
		self.influence_matrix = {}
		self.conteiner = {}
		self.dif.more_action()
		self.leader = {}
		for action in self.dif.action_table:
			print('action',action)
			inverse = [(value, key) for key, value in self.dif.action_table[action].items()]
			time_max = max(inverse)[0]
			self.pi_g = ceil(time_max*(1/3)) #tempo di dimensione della window
			if self.pi_g == 0:
				self.pi_g = 1
			print('tempo max ',time_max)
			print('pigreco ',self.pi_g)
			print('lunghezza action Table',len(self.dif.action_table[action]))
			self.len =ceil(len(self.dif.action_table[action])*(3/3))
			print('lunghezza Bit vector',self.len)
			self.window = []
			self.influence_vector = {}
			self.conteiner = Container(ceil(self.len))
			self.influence_matrix[action] = {}

			self.nodes_table = list(self.dif.action_table[action])
			while self.nodes_table != [] and self.dif.action_table[action][self.nodes_table[0]] >=  time_max-self.pi_g :
				self.window.append(self.nodes_table[0])
				pos = self.conteiner.position(self.nodes_table[0])
				self.influence_vector[self.nodes_table[0]] = BitVector.BitVector(size = ceil(self.len))
				self.influence_vector[self.nodes_table[0]][pos] = 1
				neighbors_list = self.dif.G.neighbors(self.nodes_table[0])
				#print('node',self.nodes_table[0],self.influence_vector[self.nodes_table[0]])
				for neighbor in neighbors_list:
					if neighbor in self.window:
						#print('vicino',neighbor,self.influence_vector[neighbor])
						self.influence_vector[self.nodes_table[0]] = self.influence_vector[self.nodes_table[0]].__or__(self.influence_vector[neighbor])
				self.influence_matrix[action][self.nodes_table[0]] = self.influence_vector[self.nodes_table[0]].count_bits() -1
				self.nodes_table.remove(self.nodes_table[0])
			
			while self.nodes_table != []:
				#print('window',self.window)
				#print('action_table',self.dif.action_table[action])
				self.move_window(5,action)
				#print('window',self.window)
			'''for k,v in self.influence_matrix[action].items():
				if v != 0:
					print(k,v)'''
		self.leader_ranking()
	
	def leader_ranking(self):
		utenti_influenzati =  {} #utenti influnzati per ogni azione 
		for action in self.influence_matrix:
			for node in self.influence_matrix[action]:
				utenti_influenzati.setdefault(node,[]).append(self.influence_matrix[action][node])
		for node in utenti_influenzati:
			self.leader[node] =ceil(sum(utenti_influenzati[node])/len(utenti_influenzati[node]))
		self.leader = collections.OrderedDict(sorted(self.leader.items(), key=lambda t: t[1], reverse=True ))
		#print(self.leader)

	def move_window(self,size,action):
		time_max_window = self.dif.action_table[action][self.window[0]]
		#print('tempo max window ',time_max_window)
		time_min_window =  self.dif.action_table[action][self.window[-1]]
		#print('tempo min window ',time_min_window)
		#print('update')
		while self.window != [] and self.dif.action_table[action][self.window[0]] >= time_max_window-size:
			self.update(self.window[0],action)
			self.window.remove(self.window[0])
		#print('propagate')
		while self.nodes_table != [] and self.dif.action_table[action][self.nodes_table[0]] >=  time_min_window - size :
			self.propagate(self.nodes_table[0],action)
			self.influence_matrix[action][self.nodes_table[0]] = self.influence_vector[self.nodes_table[0]].count_bits() -1
			self.nodes_table.remove(self.nodes_table[0])
		
		if self.window == [] and self.nodes_table != []:
			#print('window empty')
			self.propagate(self.nodes_table[0],action)
			self.influence_matrix[action][self.nodes_table[0]] = self.influence_vector[self.nodes_table[0]].count_bits() -1
			self.nodes_table.remove(self.nodes_table[0])




	def update(self,node,action):
		pos = self.conteiner.delete(node)
		for ele in self.conteiner.posizione:
			self.influence_vector[ele][pos] = 0
	def propagate(self,node,action):
		self.window.append(node)
		pos = self.conteiner.position(node)
		self.influence_vector[node] = BitVector.BitVector(size = ceil(self.len))
		self.influence_vector[node][pos] = 1
		for ele in self.conteiner.posizione:
			if self.dif.G.hasEdge(node,ele):
				self.influence_vector[node] = self.influence_vector[node].__or__(self.influence_vector[ele])
		
#classe per gestire l'assegnazione/rimozione del bit per ogni vertice
class Container:
	"""docstring for Container"""
	def __init__(self,numEle):
		self.contenitore = []
		self.posizione = {}
		self.num_element = numEle
		for i in range(self.num_element):
			self.contenitore.append(i)

	def position(self,number):
		if number in self.posizione:
			return self.posizione[number]
		else:
			if self.contenitore == []:
				print('bit non assegnabile')
			pos = random.choice(self.contenitore)
			self.contenitore.remove(pos)
			self.posizione[number] = pos
			return pos
	def delete(self,number):
		pos = self.posizione[number]
		del(self.posizione[number])
		self.contenitore.append(pos)
		return pos 
		

'''
if __name__ == "__main__":
	CL = CompLeader()
'''
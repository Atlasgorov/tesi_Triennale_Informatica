import csv
import collections
V = 0
E = 0
n = 0
vertices = []
metis = {}
data = ""
nomefile = 'Y2H'
with open('./network/'+nomefile+'.edgeList') as csvfile:
	reader = csv.reader(csvfile, delimiter='	')
	
	for row in reader:
		node_0 = int(row[0]) + n
		node_1 = int(row[1]) + n
		for i in [node_0,node_1]:
			if i not in vertices:
				vertices.append(i)
				metis[i] = []
				V += 1
		#if node_0 not in metis[node_1]:
		metis[node_1].append(node_0)
		#if node_1 not in metis[node_0]:
		metis[node_0].append(node_1)
		E += 1
metis =collections.OrderedDict(sorted(metis.items(), key=lambda t: t[0]))
print('len(metis)',len(metis))
data = str(V)+' '+str(E)+'\n'
for i in metis:
	#print(i)
	for ele in metis[i][:-1]:
		data += str(ele)+' '
	data += str(metis[i][-1])
	data += '\n'
data = data[:-1]
with open(nomefile+'.graph', 'w') as output:
	output.write(data)
	output.close()

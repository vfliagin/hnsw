import mmap
import numpy as np
import pickle
import random
from collections import deque
from tqdm import tqdm
from hnsw import HNSW
from hnsw import l2_distance, heuristic

def read_fbin(filename, start_idx=0, chunk_size=None):
	""" Read *.fbin file that contains float32 vectors
	Args:
		:param filename (str): path to *.fbin file
		:param start_idx (int): start reading vectors from this index
		:param chunk_size (int): number of vectors to read. 
								 If None, read all vectors
	Returns:
		Array of float32 vectors (numpy.ndarray)
	"""
	with open(filename, "rb") as f:
		nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
		nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
		arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
						  offset=start_idx * 4 * dim)
	return arr.reshape(nvecs, dim)

def bfs(layer: dict):
	"""
	Проход графа в ширину, подсчет компонент свзяности
	"""

	visited = {}
	vertex_queue = deque()
	components_cnt = 0

	if len(layer.keys()) == 0:
		return component_cnt

	while len(visited.keys()) < len(layer.keys()):

		eligeble_vertices = [vert for vert in layer.keys() if vert not in visited.keys()]
		start = random.choice(eligeble_vertices)
		vertex_queue.append(start)
		while vertex_queue:
			current_vertex = vertex_queue.popleft()
			visited[current_vertex] = True
			for neighbour in [row[0] for row in layer[current_vertex]]:
				if not visited.get(neighbour, False):
					vertex_queue.append(neighbour)

		components_cnt += 1

	return components_cnt

def count_components(hnsw):
	"""
	Подсчет компонент связности для каждого слоя hnsw
	"""

	print('layer count:', len(hnsw._graphs))

	for i, layer in enumerate(hnsw._graphs):
		print(f'layer {i} has {bfs(layer)} components')


if __name__ == '__main__':

	load_graph_one = False

	if not load_graph_one:
		train_data = read_fbin('base.10M.fbin')

		hnsw_one = HNSW(distance_func=l2_distance, m=32, m0=64, ef=10, ef_construction=64, neighborhood_construction = heuristic)
		# Add data to HNSW
		for x in tqdm(train_data):
			hnsw_one.add(x)

		with open('hnsw_one.pickle', 'wb') as f:
			pickle.dump(hnsw_one, f)
	else:

		with open('hnsw_one.pickle', 'rb') as f:
			hnsw_one = pickle.load(f)

	print('hnsw_one results:')
	count_components(hnsw_one)

	load_graph_two = False

	if not load_graph_two:
		train_data = read_fbin('base.10M.fbin')

		hnsw_two = HNSW(distance_func=l2_distance, m=16, m0=32, ef=10, ef_construction=64, neighborhood_construction = heuristic)
		# Add data to HNSW
		for x in tqdm(train_data):
			hnsw_two.add(x)

		with open('hnsw_two.pickle', 'wb') as f:
			pickle.dump(hnsw_two, f)
	else:

		with open('hnsw_two.pickle', 'rb') as f:
			hnsw_two = pickle.load(f)

	print('hnsw_two results:')
	count_components(hnsw_two)

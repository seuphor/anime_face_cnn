import pickle
import os

pickle_path = 'pickle/'

def saver(path, name):
	with open(pickle_path + path, 'wb') as f:
		pickle.dump(name, f)
	print('Saved in %s' %(os.getcwd() + pickle_path))

def loader(path):
	with open(pickle_path + path, 'rb') as f:
		loaded = pickle.load(f)
	print('Loaded from %s' %(os.getcwd() + pickle_path))
	return loaded
import numpy as np
import re
import itertools
from collections import Counter
import os
from gensim.models import KeyedVectors
import imdb_data_helpers as idh
import argparse
import progressbar
import caffe
import lmdb
import time
import shutil
from imdb import IMDb
import create_dataset as cd

def title_to_movie(ia, title):
	movies = ia.search_movie(title)
	for m in movies[:3]:
		y = m['year'] if m.has_key('year') else "????"
		resp = raw_input("Did you mean \"" + m['title'] + " (" + str(y) + ")\"? (y/n) :  ")
		if resp.lower()=='y':
			return m
	return None

def main(movie, vecs_file='data/GoogleNews-vectors-negative300.bin', padding='</s>', word_size=100):
	word2vec = cd.load_word2vec(vecs_file)
	movie_data = [None,None,None,None]
	movie_keys = ['genre', 'director', 'writer', 'editor', 'cast']
	for k in movie_keys:
		if movie.has_key(k):
			movie_data.append([str(p).decode('utf-8') for p in movie[k]])
		else:
			movie_data.append([])
	movie_data = movie_data + [[], movie['plot'][0].decode('utf-8') if movie.has_key('plot') else ""]
	label,vector = cd.get_labels_vectors([[None, idh.to_text(movie_data)]], word2vec, word_size, padding)

	folder_name = 'temp_' + idh.clean_str(movie['title']).replace(' ', '_')

	#lmdb
	if not os.path.exists('lmdbs'):
		os.makedirs('lmdbs')

	cd.create_dataset_x(vector, folder_name)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Fetch movie from imdb and create singular dataset from with Vectors')
	parser.add_argument('-t', help='fetch movie from title', default=False, action="store_true")
	parser.add_argument('-i', help='fetch movie from imdb_id', default=False, action="store_true")
	parser.add_argument('movie', help='title or id', action="store")
	parser.add_argument('--vecs', help='Vector binary file', default='data/GoogleNews-vectors-negative300.bin', action="store")
	parser.add_argument('--padd', help='Padding string', default='</s>', action="store")
	parser.add_argument('--words', help='Number of words', type=int, default=100, action="store")
	args = vars(parser.parse_args())

	if not os.path.exists('lmdbs'):
		os.makedirs('lmdbs')
	folders = os.listdir('lmdbs')
	to_remove = []
	for f in folders:
		if f[:5] == 'temp_':
			to_remove.append(f)
	if len(to_remove)>0:
		approve_remove = raw_input("Remove old temporary files " + str(to_remove) + " ? (y/n) :  ")
		if approve_remove.lower() == 'y':
			for f in to_remove:
				shutil.rmtree('lmdbs/' + f)

	ia = IMDb()

	if args['t']:
		movie = title_to_movie(ia, args['movie'])
	elif args['i']:
		movie = ia.get_movie(args['movie'])
	else:
		print("NO OPTION CHOSEN")
		movie = None
	if movie is not None:
		start_time = time.time()
		ia.update(movie)
		main(movie, args['vecs'],args['padd'],args['words'])
		print("Rating: " + str(movie['rating'] if movie.has_key('rating') else "No Rating"))
		print('Done after %s seconds' % (time.time() - start_time))





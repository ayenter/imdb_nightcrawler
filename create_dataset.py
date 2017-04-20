import numpy as np
import re
import itertools
from collections import Counter
import os
from gensim.models import KeyedVectors
import imdb_data_helpers as idh
import argparse


def load_google_word2vec(path):
	model = KeyedVectors.load_word2vec_format(path, binary=True)
	return model

def get_labels_vectors(movies, word2vec, info_size=100, padding='</s>'):
	labels = []
	vectors = []
	for m in movies:
		labels.append(m[0])
		vector = []
		for word in m[1].split():
			if word in word2vec:
				vector.append(word2vec[word])
			if len(vector) == info_size:
				break
		while len(vector) < info_size:
			vector.append(word2vec['</s>'])
		vectors.append(vector)
	vectors = np.array(vectors)
	labels = np.array(labels)
	return [labels, vectors]


def _write_batch_to_lmdb(db, batch):
    """
    Write a batch of (key,value) to db
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for key, datum in batch:
                lmdb_txn.put(key, datum.SerializeToString())
    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit*2
        try:
            db.set_mapsize(new_limit) # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0,87):
                raise ImportError('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
            else:
                raise e
        # try again
        _write_batch_to_lmdb(db, batch)

#def create_dataset()

def main(data_file='data/movies.csv', vecs_file='data/GoogleNews-vectors-negative300.bin', padding='</s>', word_size=100):
	word2vec = load_google_word2vec(data_file)
	train,test = idh.get_processed_movies(data_file)
	train_y,train_x = get_labels_vectors(train, word2vec, word_size, padding)
	test_y,test_x = get_labels_vectors(test, word2vec, word_size, padding)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create Dataset from CSV and Vectors')
	parser.add_argument('--data', help='Movie CSV file', default='data/movies.csv', action="store")
	parser.add_argument('--vecs', help='Vector binary file', default='data/GoogleNews-vectors-negative300.bin', action="store")
	parser.add_argument('--padd', help='Padding string', default='</s>', action="store")
	parser.add_argument('--words', help='Number of words', type=int, default=100, action="store")
	args = vars(parser.parse_args())

	if not os.path.exists('lmdbs'):
		os.makedirs('lmdbs')

	start_time = time.time()

	main(args['data'],args['vecs'],args['padd'],args['words'])

	print 'Done after %s seconds' % (time.time() - start_time)





import csv
from imdb import IMDb
import progressbar
from random import shuffle
import os.path
from zipfile import ZipFile
import re
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_movies(file = "data/movies.csv"):
	if not os.path.isfile(file):
		if os.path.isfile(file+".zip"):
			print("EXTRACTING " + file + ".zip")
			ZipFile(file+".zip").extractall(os.path.split(file)[0])
		else:
			print("NO FILE \"" + file + "\" OR \"" + file  + ".zip\"")
			raise OSError(2, 'No such file or directory', file)
	movies = []
	with open(file, 'rb') as f:
		reader = csv.reader(f, delimiter=',', quotechar='"', escapechar='\\')
		bar = progressbar.ProgressBar()
		print("Grabbing Movies")
		for row in bar(reader):
			movies.append([
				int(row[0]), #id
				int(row[1]), #year
				int(row[2]), #votes
				float(row[3]), #rating
				row[4].split('|'), #genres
				[" ".join(n.split(", ")[::-1]) for n in row[5].split('|')], #director
				[" ".join(n.split(", ")[::-1]) for n in row[6].split('|')], #writer
				[" ".join(n.split(", ")[::-1]) for n in row[7].split('|')], #editor
				[" ".join(n.split(", ")[::-1]) for n in row[8].split('|')], #cast
				[" ".join(n.split(", ")[::-1]) for n in row[9].split('|')] if row[9]!=('N') else [], #cast_extra
				row[10]
				])
	print("# movies :  " + str(len(movies)))
	print("")
	return movies

def filter_min_votes(movies, min_votes, index=2):
	qualify = []
	bar = progressbar.ProgressBar()
	print("Filtering Movies " + " Over " + str(min_votes) + " Votes")
	for m in bar(movies):
		if m[index] >= min_votes:
			qualify.append(m)
	print("# votes over " + str(min_votes) + " :  " + str(len(qualify)))
	print("")
	return qualify

def whole_round(x, base=5):
    return int(base * round(float(x)/base))

def get_train_test(movies, batch_size=200, train_split=.8, sort_index=1):
	size = len(movies)
	if sort_index >= 0:
		sorted_movies = sorted(movies,key=lambda x: x[sort_index])
	else:
		sorted_movies = shuffle(movies)
	num_train = int(size*train_split)
	num_train = whole_round(num_train, batch_size)
	size = size - (size%batch_size)
	num_test = size - num_train
	print("Train:" +str(round(float(num_train)/size, 4)*100) + "% / Test:" + str(round(float(num_test)/size, 4)*100) + "%")
	print("Train:" + str(num_train) + " / Test:" + str(num_test))
	print("")
	return [sorted_movies[:num_train], sorted_movies[num_train:size]]

def remove_articles(text):
	articles = ['a', 'an', 'and', 'the']
	return " ".join([w for w in text.split() if w.lower() not in articles])


def to_text(m, cast_limit=10, text_limit=-1):
	cast = m[8]+m[9]
	text = " ".join(m[4]+m[5]+m[6]+m[7]+cast[:cast_limit]) + " " + m[10] + " " + " ".join(cast[cast_limit:])
	#text = remove_articles(text)
	text = clean_str(text)
	if text_limit >= 0:
		return " ".join(text.split()[:text_limit])
	else:
		return text

def all_to_text(movies, cast_limit=10, text_limit=-1):
	bar = progressbar.ProgressBar()
	print("Simplifying Movies to Text")
	results = [[m[3],to_text(m, cast_limit, text_limit)] for m in bar(movies)]
	print("")
	return results

def normalize_labels(movies, index=3):
	ratings = [m[index] for m in movies]
	r = np.asarray(ratings)
	n = (r-np.mean(r))/np.std(r)
	for i in range(len(movies)):
		movies[i][index] = n[i]
	return movies

def labels_to_classes(movies):
	for m in movies:
		m[3] = int(m[3]*2)+1
	return movies


def get_processed_movies(file="data/movies.csv", min_votes=50, batch_size=200, train_split=.8, sort_index=1, cast_limit=10, text_limit=-1, normalize=False, classes=False):
	movies = filter_min_votes(get_movies(file), min_votes)
	if classes:
		movies = labels_to_classes(movies)
	elif normalize:
		movies = normalize_labels(movies)
	train, test = get_train_test(movies, batch_size, train_split, sort_index)
	shuffle(train)
	shuffle(test)
	return [all_to_text(train, cast_limit, text_limit), all_to_text(test, cast_limit, text_limit)]

"""
import imdb_data_helpers as idh
train,test = idh.get_processed_movies()
"""

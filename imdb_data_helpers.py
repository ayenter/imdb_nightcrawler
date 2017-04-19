import csv
from imdb import IMDb
import progressbar
from random import shuffle

# p9.movie_id, p9.year, p9.votes, p9.rating, p9.genres, p7.director, p7.writer, p9.editor, p7.cast, p7.cast_extra, p9.plot

def get_movies(file = "data.csv"):
	movies = []
	with open(file, 'rb') as f:
		reader = csv.reader(f, delimiter=',', quotechar='"', escapechar='\\')
		for row in reader:

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
	print "# movies :  " + str(len(movies))
	return movies

def filter_min_votes(movies, min_votes, index=2):
	qualify = []
	for m in movies:
		if m[index] >= min_votes:
			qualify.append(m)
	print "# votes over " + str(min_votes) + " :  " + str(len(qualify))
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
	num_test = size - num_train
	print "Train:" +str(round(float(num_train)/size, 4)*100) + "% / Test:" + str(round(float(num_test)/size, 4)*100) + "%"
	print "Train:" + str(num_train) + " / Test:" + str(num_test)
	return [sorted_movies[:num_train], sorted_movies[num_train:]]

def remove_articles(text):
	articles = ['a', 'an', 'and', 'the']
	return " ".join([w for w in text.split() if w.lower() not in articles])


def to_text(m, cast_limit=10, text_limit=100):
	cast = m[8]+m[9]
	text = " ".join(m[4]+m[5]+m[6]+m[7]+cast[:cast_limit]) + " " + m[10] + " " + " ".join(cast[cast_limit:])
	#text = remove_articles(text)
	return " ".join(text.split()[:text_limit])

def all_to_text(movies, cast_limit=10, text_limit=100):
	return [to_text(m, cast_limit, text_limit) for m in movies]

def process_movies(file="data.csv", min_votes=50, batch_size=200, train_split=.8, sort_index=1, cast_limit=10, text_limit=100):
	train, test = get_train_test(filter_min_votes(get_movies(file), min_votes), batch_size, train_split, sort_index)
	shuffle(train)
	shuffle(test)
	return [all_to_text(train, cast_limit, text_limit), all_to_text(test, cast_limit, text_limit)]


"""
import imdb_data_helpers as idh
movies_50 = idh.filter_min_votes(idh.get_movies(),50)
train, test = idh.get_train_test(movies_50)
"""
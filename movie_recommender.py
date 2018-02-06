# movie recommender -ansu-
import numpy as np 
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch the data
data = fetch_movielens(min_rating=4.5)

#print training data
print(repr(data["train"]))
print(repr(data["test"]))

# using WARP to mmisimise the error - weighted approximated rank pairwise - content absed + collabrative -> hybrid system
model = LightFM(loss="warp")

#train model
model.fit(data["train"], epochs=20, num_threads=2)

def movie_recommender(model, data, user_ids):

	# get the number of users and number of movies in training data
	n_users, n_items = data["train"].shape

	# generate recommendations for each user 

	for user_id in user_ids:

		#movies they already like
		known_positives = data["item_labels"][data["train"].tocsr()[user_id].indices]

		#movies predicted by our model
		scores = model.predict(user_id, np.arange(n_items))

		#rank them inorder of their score in descending order - ranking them from the mot liked to the least liked
		top_items = data["item_labels"][np.argsort(-scores)]


		print("  User %s : " % user_id)
		print("   Top 5 movies highly rated by the user:")

		for x in known_positives[:5]:
			print("    %s" % x)

		print("   Our top 5 Recomended movies:")

		for x in top_items[:5]:
			print("    %s" % x)

# testing out the movie recommender with some user ids
movie_recommender(model, data, [4, 32, 220])			
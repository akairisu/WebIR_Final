import csv
import numpy as np
from collections import defaultdict
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

id_dict = defaultdict(lambda : " ")
title_dict = defaultdict(lambda : -1)

predict = np.load('pearson_predict.npz')['arr_0']

with open('books.csv', encoding="utf-8", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for book in reader:
        id_dict[int(book['book_id'])] = book['title']
        title_dict[book['title']] = int(book['book_id'])

original_data = sparse.load_npz('rating_matrix.npz')
rating = original_data.tocsc()
ISBN_mean = np.array(rating.mean(axis=0)).flatten()
nonzeros = []
for i in range(rating.shape[1]):
    nonzeros.append(len(rating[:, i].data))
nonzeros = np.array(nonzeros)
ISBN_mean = ISBN_mean * rating.shape[0] / nonzeros
rating = rating.tocsr()

while(True):
	print("Please input userid:(0 for new user)")
	userid = int(input())
	if userid > 53424 or userid < 0:
		print("Invalid user!")
	print("Query or Recommendation?(q/r)")
	method = input()
	if method == 'q':
		print("Please input query:")
		query = input()
		print(query)
		#predict with language model
	else:
		if userid == 0:
			book_list = []
			print("Please enter favorite books (enter book id, book title or press enter to skip):")
			query = input()
			while len(query) != 0:
				if query.isdigit():
					if id_dict[int(query)] != " ":
						print(query, id_dict[int(query)])
						book_list.append(int(query))
					else:
						print("Invalid book id!")
				else:
					if title_dict[query] != -1:
						print(title_dict[query], query)
						book_list.append(title_dict[query])
					else:
						print("Invalid book title!")
				print("Continue?(enter book id, book title or press enter to skip):")
				query = input()
			if len(book_list) == 0:
				recommendation_list = (-ISBN_mean).argsort()[:20]
				print("Recommended:")
				for books in recommendation_list:
					print(books + 1, id_dict[books + 1])
			else:
				personal_rating = np.zeros((1, rating.shape[1]))
				for books in book_list:
					personal_rating[0][books - 1] = 5
				cosine_sim = cosine_similarity(personal_rating, original_data).squeeze(0)
				top_sim = (-cosine_sim).argsort()[:20]
				top_rating = predict[top_sim].squeeze(0)
				top_mean = top_rating.mean(axis=0)
				recommendation_list = (-top_mean).argsort()[:20]
				print("Recommended:")
				for books in recommendation_list:
					print(books + 1, id_dict[books + 1])
		else:
			#Predict with BPR
			pass
	print("Do you want to exit?(y/n)")
	exit = input()
	if exit == "y":
		break
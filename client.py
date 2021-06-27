import csv
import numpy as np
from collections import defaultdict
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

id_dict = defaultdict(lambda : "")
title_dict = defaultdict(lambda : -1)

with open('books.csv', encoding="utf-8", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for book in reader:
        id_dict[int(book['book_id'])] = book['title']
        title_dict[book['title']] = int(book['book_id'])

original_data = sparse.load_npz('rating_matrix.npz')
num_users = original_data.shape[0]
num_books = original_data.shape[1]

ground_truth = [[] for _ in range(num_users)]
with open('to_read.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for data in reader:
        ground_truth[int(data['user_id']) - 1].append(int(data['book_id']) - 1)

predict_matrix = np.load('predict_matrix.npy')

while(True):
    userid = int(input("Please input userid (0 for new user): "))
    if userid > 53424 or userid < 0:
        print("Invalid user!")
    method = input("Query or Recommendation?(q/r): ")
    if method == 'q':
        query = input("Please input query: ")
        print(query)
        #predict with language model
    else:
        if userid == 0:
            book_list = []
            query = input("Please enter favorite books (enter book id, book title or press enter to skip): ")
            while len(query) != 0:
                if query.isdigit():
                    if id_dict[int(query)] != " ":
                        print(query, id_dict[int(query)])
                        book_list.append(int(query) - 1)
                    else:
                        print("Invalid book id!")
                else:
                    if title_dict[query] != -1:
                        print(title_dict[query], query)
                        book_list.append(title_dict[query] - 1)
                    else:
                        print("Invalid book title!")
                query = input("Continue?(enter book id, book title or press enter to skip): ")
            if len(book_list) == 0:
                score = predict_matrix.sum(axis=0)
                recommendation_list = (-score).argsort()[:10]
                print("Recommended:")
                for books in recommendation_list:
                    print('    ', books + 1, id_dict[books + 1])
            else:
                new_user = np.zeros(num_books)
                new_user[book_list] = 5
                new_user = new_user.reshape(1, -1)
                sim = np.array(cosine_similarity(new_user, original_data)).flatten()
                top_users = (-sim).argsort()[:20]
                pred = predict_matrix[top_users]
                for i in range(pred.shape[0]):
                    pred[i] = pred[i] * sim[top_users[i]]
                score = pred.sum(axis=0)
                rank = (-score).argsort()
                recommendation_list = []
                for i in range(len(rank)):
                    if rank[i] not in book_list:
                        recommendation_list.append(rank[i] + 1)
                        if len(recommendation_list) == 10:
                            break
                print("Recommended:")
                for book in recommendation_list:
                    print('    ', book, id_dict[book])
        else:
            #Predict with BPR
            ranking = (-predict_matrix[userid - 1]).argsort()
            count = 0
            recommendation_list = []
            for j in range(len(ranking)):
                if ranking[j] not in original_data[userid - 1].indices:
                    recommendation_list.append(ranking[j] + 1)
                    count += 1
                    if count == 10:
                        break
            print("Recommended:")
            for book in recommendation_list:
                print('    ', book, id_dict[book])
            print("To-read:")
            for i in ground_truth[userid - 1]:
                if i + 1 in recommendation_list:
                    print('    ', 'V', i + 1, id_dict[i + 1])
                else:
                    print('    ', 'X', i + 1, id_dict[i + 1])
            
    exit = input("Do you want to exit?(y/n): ")
    if exit == "y":
        break
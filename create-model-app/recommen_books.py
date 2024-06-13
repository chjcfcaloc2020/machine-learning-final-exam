import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

#load data
books = pd.read_csv("D:/VKU_2_3/Shiny/books.csv", sep=";", on_bad_lines='skip', low_memory=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv("D:/VKU_2_3/Shiny/users.csv", sep=';', on_bad_lines='skip', low_memory=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv("D:/VKU_2_3/Shiny/ratings.csv", sep=';', on_bad_lines='skip', low_memory=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

#data processing
books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'], axis=1, inplace=True)
books = books[(books.yearOfPublication != 'DK Publishing Inc') & (books.yearOfPublication != 'Gallimard')]
books.yearOfPublication = books.yearOfPublication.astype('int32')
books = books.dropna(subset=['publisher'])

users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
users.Age = users.Age.fillna(users.Age.mean())
users.Age = users.Age.astype(np.int32)

#prepare data
n_users = users.shape[0]
n_books = books.shape[0]

ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
ratings_explicit = ratings_new[ratings_new.bookRating != 0]
ratings_implicit = ratings_new[ratings_new.bookRating == 0]

# Collaborative Filtering Based Recommendation Systems
counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]

# Generate matrix table from explicit ratings table
ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating').fillna(0)
userID = ratings_matrix.index
ISBN = ratings_matrix.columns
# print(userID)

U, sigma, Vt = svds(ratings_matrix.to_numpy(), k=50)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns)

# Take a particular user_id
user_id = 1
userID = ratings_matrix.iloc[user_id-1, :].name

sorted_user_predictions = preds_df.iloc[user_id].sort_values(ascending=False)

# Get all user interacted books
user_data = ratings_explicit[ratings_explicit.userID == (userID)]
book_data = books[books.ISBN.isin(user_data.ISBN)]

# Merge
user_full_info = user_data.merge(book_data)
print('User {0} has already rated {1} books.'.format(userID, user_full_info.shape[0]))

recommendations = (books[~books['ISBN'].isin(user_full_info['ISBN'])].
                   merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', left_on = 'ISBN'
                         ,right_on = 'ISBN')).rename(columns = {user_id: 'Predictions'})

# print(recommendations.sort_values('Predictions', ascending = False).iloc[:10, :])
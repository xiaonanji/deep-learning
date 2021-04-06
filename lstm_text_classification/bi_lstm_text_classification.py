'''
Use pytorch's bi-lstm to classify text

Use imdb review data as training sample. The review is a piece of English text and the label is
either positive or negative. This can be viewed as a binary classification problem.
'''

import pandas as pd
from sklearn import model_selection

import config
import imdb_dataset
import lstm

# Read imdb.csv from local file. You can download the data from
# https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
df = pd.read_csv('imdb.csv')
# Overwrite the sentiment column of the dataset by replacing positive with 1 and negative with 0
df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)

# We use k-fold cross validation. So first we create a kfold column to hold the fold index:
# fold index is i for i-th fold, where i starts from 0
df['kfold'] = -1

# This is to shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Even though the positive and negative are relatively balanced, we still use stratified k-fold
y = df.sentiment.values
kf = model_selection.StratifiedKFold(n_splits=5) # We use 5-fold cross validation

# For stratified cross validation, we need to supply both X and y, then sklearn's StratifiedKFold
# will return corresponding fold index
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

def train(data_loader, model, optimizer, device):
    model.train()
    total = len(data_loader)
    c = 0
    
    # Iterate over batches
    for data in data_loader:
        c += 1
        if c % 100 == 0:
            print(f'batch {c}/{total}')
        
        reviews = data['review']
        targets = data['target']
        
        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        
        predictions = model(reviews)
        
        # The loss function is essentially a logistic loss
        loss = nn.BCEWithLogitsLoss()(
            predictions,
            targets.view(-1, 1)
        )
        
        loss.backward()
        optimizer.step()

def evaluate(data_loader, model, device):
    final_predictions = []
    final_targets = []
    
    model.eval()
    total = len(data_loader)
    c = 0
    
    with torch.no_grad():
        for data in data_loader:
            c += 1
            if c % 100 == 0:
                print(f'batch {c}/{total}')
            
            reviews = data['review']
            targets = data['target']
            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            
            predictions = model(reviews)
            predictions = predictions.cpu().numpy().tolist()
            targets = data['target'].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)
            
        return final_predictions, final_targets

# Below is to load embedding vectors from FastText
import io
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn import metrics

def load_vectors(fname):
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        
    return data

def create_embedding_matrix(word_index, embedding_dict):
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
            
    return embedding_matrix

def run(df, fold):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    print('Tokenizing...')
    tokenizer.fit_on_texts(df.review.values.tolist())
    print('Tokenized')
    
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)
    
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=config.MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=config.MAX_LEN)
    
    train_dataset = imdb_dataset.IMDBDataset(
        reviews=xtrain,
        targets=train_df.sentiment.values
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
    )
    
    valid_dataset = imdb_dataset.IMDBDataset(
        reviews=xtest,
        targets=valid_df.sentiment.values
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VALID_BATCH_SIZE,
    )
    
    print('Loading embedding vectors...')
    embedding_dict = load_vectors('wiki-news-300d-1M.vec')
    embedding_matrix = create_embedding_matrix(
        tokenizer.word_index, embedding_dict
    )
    print('Embedding vectors loaded')
    
    device = torch.device('cpu')
    model = lstm.LSTM(embedding_matrix)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_accuracy = 0
    early_stopping_counter = 0
    for epoch in range(config.EPOCHS):
        train(train_data_loader, model, optimizer, device)
        outputs, targets = evaluate(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f'Fold:{fold}, Epoch:{epoch}, Accuracy Score={accuracy}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter > 2:
            break
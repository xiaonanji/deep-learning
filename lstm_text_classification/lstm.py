'''
Define the bi-lstm model structure. In pytorch this is done by creating a sub-class of nn.Module.
The key method to overwrite is forward.

The model we have is essentially a two-layer NN. The first layer is embedding layer, which turns
the input word sequence into sequence of embeddings. The embedding we use is FastText's pre-trained
embedding vectors: https://fasttext.cc/docs/en/english-vectors.html. You can use either the wiki based
or the crawl based vectors. They are both 300d.
'''

import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTM, self).__init__()
        num_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        
        # For embedding layer, there is always two dimensions, first dimension is the total number
        # of words and the second dimension is the embedding vector's dimension.
        # The embedding layer, if provided by the embedding matrix, is essentially a lookup layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim
        )
        
        # Notice that we are not training the embeddings ourselvies so we set the weights of the
        # embedding layer to given embedding matrix and we also need to set the requires_grad to
        # False so this layer will not be trained at all
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            )
        )
        
        self.embedding.weight.requires_grad = False
        
        # The second layer is a bi-lstm, we set 128 lstm cells so for bi-directional lstm will
        # output 256d
        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional=True,
            batch_first=True
        )
        
        # The output layer is a dense layer, it takes 512d because later we will use max and average
        # pooling results. Given that our problem is a binary classification problem, the output 
        # dimension of this layer will be 1
        self.out = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x) # Here dimension of x will be (batch_size, seq_max_len, 128*2)
        avg_pool = torch.mean(x, 1) # Dimension of avg_pool will be (batch_size, 128*2)
        max_pool, _ = torch.max(x, 1) # Dimension of max_pool will be (batch_size, 128*2)
        
        # This is a typical way to getting sequences of lstm outputs into one vector. We get the
        # average and max of all sequential outputs and concatinate them together
        out = torch.cat((avg_pool, max_pool), 1) # Dimension of out will be (batch_size, 128*2*2)
        out = self.out(out) # Finally dimension of out will be (batch_size, 1)
        
        return out
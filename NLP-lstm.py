import numpy as np
from string import punctuation
from collections import Counter
# import Word2Vec loading capabilities
from gensim.models import KeyedVectors
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import datetime
from string import punctuation

# read data from text files

with open('data/train_review_slim.csv', 'r', encoding="mac_roman") as f:
    reviews = f.read()
with open('data/train_label_slim.csv', 'r', encoding="mac_roman") as f:
    labels = f.read()

# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])

# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()

# 1=positive, 0=negative label conversion
labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == '1' else 0 for label in labels_split])

# outlier review stats
review_lens = Counter([len(x.split()) for x in reviews_split])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


print('Number of reviews before removing outliers: ', len(reviews_split))

## remove any reviews/labels with zero length from the reviews_ints list.

# get indices of any reviews with length 0
non_zero_idx = [ii for ii, review in enumerate(reviews_split) if len(review.split()) != 0]

# remove 0-length reviews and their labels
reviews_split = [reviews_split[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers: ', len(reviews_split))

# Creating the model
embed_lookup = KeyedVectors.load_word2vec_format('word2vec_model/GoogleNews-vectors-negative300-SLIM.bin',
                                                 binary=True)

# store pretrained vocab
pretrained_words = []
for word in embed_lookup.vocab:
    pretrained_words.append(word)

row_idx = 1

# get word/embedding in that row
word = pretrained_words[row_idx] # get words by index
embedding = embed_lookup[word] # embeddings by word

# vocab and embedding info
print("Size of Vocab: {}\n".format(len(pretrained_words)))
print('Word in vocab: {}\n'.format(word))
print('Length of embedding: {}\n'.format(len(embedding)))
#print('Associated embedding: \n', embedding)

# convert reviews to tokens
def tokenize_all_reviews(embed_lookup, reviews_split):
    # split each review into a list of words
    reviews_words = [review.split() for review in reviews_split]

    tokenized_reviews = []
    for review in reviews_words:
        ints = []
        for word in review:
            try:
                idx = embed_lookup.vocab[word].index
            except:
                idx = 0
            ints.append(idx)
        tokenized_reviews.append(ints)

    return tokenized_reviews


tokenized_reviews = tokenize_all_reviews(embed_lookup, reviews_split)
# testing code and printing a tokenized review
print(tokenized_reviews[0])


def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


# Test your implementation!
seq_length = 200
features = pad_features(tokenized_reviews, seq_length=seq_length)

## test statements - do not change - ##
assert len(features) == len(tokenized_reviews), "Your features should have as many rows as reviews."
assert len(features[0]) == seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches
print(features[:30, :10])


split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)
split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 50

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size,drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size,drop_last=True)

# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, embed_model, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional=True,
                 drop_prob=0.3):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim

        # 1. embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))  # all vectors
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True,
                            bidirectional=bidirectional)

        # 2. dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # 3. linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        if bidirectional:
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)

        else:
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

            # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        number = 1
        if self.bidirectional:
            number = 2

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_()
                      )
        return hidden


# Instantiate the model w/ hyperparams
vocab_size = len(pretrained_words)
output_size = 1
embedding_dim = len(embed_lookup[pretrained_words[0]])
hidden_dim = 256
n_layers = 2
bidirectional = False  # 这里为True，为双向LSTM

net = SentimentRNN(embed_lookup, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional)

print(net)

start = datetime.datetime.now()
print(start)

# loss and optimization functions
lr = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params
epochs = 3  # 3-4 is approx where I noticed the validation loss stop decreasing
print_every = 10
clip = 5  # gradient clipping

# move model to GPU, if available
if (train_on_gpu):
    net.cuda()

net.train()

# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)
    counter = 0

    # batch loop
    for inputs, labels in train_loader:
        counter += 1
        if (train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])
                if (train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())
            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
end = datetime.datetime.now()
print(end)


# Get test data loss and accuracy
test_losses = []  # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()

# iterate over test data
for inputs, labels in test_loader:
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])
    if (train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()

    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

# -- stats! -- ##

# avg test loss
print("Test loss: {:.6f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))


def tokenize_review(embed_lookup, test_review):
    test_review = test_review.lower()  # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    tokenized_review = []
    for word in test_words:
        try:
            idx = embed_lookup.vocab[word].index
        except:
            idx = 0
        tokenized_review.append(idx)

    return tokenized_review

def predict(embed_lookup, net, test_review, sequence_length=200):
    """
    Predict whether a given test_review has negative or positive sentiment.
    """
    net.eval()

    # tokenize review
    test_ints = tokenize_review(embed_lookup, test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features([test_ints], seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    if (train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if (pred.item() == 1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")

# set hyperparams
seq_length = 200  # good to use the length that was trained on
# negative test review
test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'

# test negative review
predict(embed_lookup, net, test_review_neg, seq_length)

# positive test review
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'

predict(embed_lookup, net, test_review_pos, seq_length)
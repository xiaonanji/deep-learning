# It is always good to store the model configurations to a separate file

# NN can only deal with fixed length of inputs
# Given that each review's length, we fix a length. Longer reviews will be truncated
# Shorter reviews will be padded with default words
# Data exploration should be used to determine the best max len parameter 
MAX_LEN = 128

# These two batch sizes are for NN
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8

# The number of iterations throughout the training sample is set to 10
EPOCHS = 10
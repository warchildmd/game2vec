import math
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pylab
from sklearn.manifold import TSNE

# 1. Read data
csv = pd.read_csv('data/steam-200k.csv', header=None, index_col=None,
                  names=['user_id', 'game', 'action', 'hours', 'other'])

# 1a. We might consider just play actions as relevant, or both play and purchase
# games = list(csv[csv.action == 'play'].game.unique())
idx_to_game = list(csv.game.unique())
game_to_idx = {}
for idx, game in enumerate(idx_to_game):
    game_to_idx[game] = idx

# There are 5155 unique games in the data set.
print('There are {0} unique games in the data set.'.format(len(idx_to_game)))

vocabulary_size = len(idx_to_game)

# 2. Create training data set. Games for a single user will be similar to sentences in text.
#    A single user game set is considered as a context.
pre_data = {}
for index, row in csv.iterrows():
    if row.user_id not in pre_data:
        pre_data[row.user_id] = set()
    # if game_to_idx[row.game] not in data[row.user_id]:
    pre_data[row.user_id].add(game_to_idx[row.game])

data = []
for x in pre_data.values():
    data.append(list(x))

'''
{1024, 825}
['XCOM Enemy Unknown', 'Aliens vs. Predator']
{3397, 966, 624, 528, 498, 1075, 1076, 1077, 187}
['9.03m', 'Happy Wars', 'Brick-Force', 'Unturned', 'Terraria', 'Overlord', 'Overlord Raising Hell', 'Overlord II', 'Trine']
{23}
['Robocraft']
{618}
['SMITE']
'''
random_user_sample = random.sample(data, 4)
for x in random_user_sample:
    print(x)
    print([idx_to_game[y] for y in x])


# 3. Batch generating function
# Generate data randomly
def generate_batch_data(game_sets, batch_size):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random set to start, skip sets smaller than 3
        rand_list = random.choice(game_sets)
        random.shuffle(rand_list)
        if len(rand_list) < 3:
            continue
        # Randomly select a game from the set as the target
        label = random.choice(rand_list)
        tuples = []
        for x in rand_list:
            for y in rand_list:
                if x != label and y != label and x != y:
                    tuples.append((label, x))
                    tuples.append((label, y))
            if len(tuples) > batch_size:
                break

        # extract batch and labels
        batch, labels = [list(x) for x in zip(*tuples)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return batch_data, label_data


sample_batch_data, sample_label_data = generate_batch_data(data, 8)
for x, y in zip(sample_batch_data, sample_label_data):
    print(x, '->', y)
    print(idx_to_game[x], '->', idx_to_game[y[0]])

# 4. Model
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 32  # Number of negative examples to sample.

# General defines
context_window = 2 * skip_window
num_labels = batch_size / context_window

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.float32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    final_embed = tf.nn.embedding_lookup(embeddings, train_dataset)

    # Compute the softmax loss, using a sample of the negative labels each time.
    print('softmax_weights: {}'.format(weights.get_shape().as_list()))
    print('softmax_biases: {}'.format(biases.get_shape().as_list()))
    print('final_embed: {}'.format(final_embed.get_shape().as_list()))
    #loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, train_labels, final_embed, num_sampled, vocabulary_size))
    # Get loss from prediction
    loss = tf.reduce_mean(tf.nn.nce_loss(weights, biases, train_labels, final_embed, num_sampled, vocabulary_size))

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between mini-batch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    # train_writer = tf.summary.FileWriter('logs/train', graph)

num_steps = 100001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialised')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch_data(data, batch_size)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step', step, ':', average_loss)
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = idx_to_game[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = idx_to_game[nearest[k]]
                    log = "%s %s," % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, metric='cosine')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(64, 64))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.savefig('tsne_skipgram.png', bbox_inches='tight')


words = [idx_to_game[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)

# Save data
pickle_data = {
    'embeddings': final_embeddings,
    'idx_to_game': idx_to_game,
    'game_to_idx': game_to_idx
}

np.save('embeddings_skipgram.npy', pickle_data)
print('Data saved to embeddings_skipgram.npy')

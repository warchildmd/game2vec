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
idx_to_game = list(csv.game[csv.action == 'play'].unique())
game_to_idx = {}
for idx, game in enumerate(idx_to_game):
    game_to_idx[game] = idx

# There are 3600 unique games in the data set.
print('There are {0} unique games in the data set.'.format(len(idx_to_game)))

vocabulary_size = len(idx_to_game)

# 2. Create training data set. Games for a single user will be similar to sentences in text.
#    A single user game set is considered as a context.
pre_data = {}
for index, row in csv.iterrows():
    if row.action == 'purchase':
        continue
    if row.user_id not in pre_data:
        pre_data[row.user_id] = set()
    pre_data[row.user_id].add((game_to_idx[row.game], row.hours))

data = []
for x in pre_data.values():
    data.append(list(x))


'''
[(477, 78.0), (248, 463.0)]
['Battlefield Bad Company 2', 'War Thunder']
[(173, 54.0), (605, 7.8), (677, 16.3), (2420, 63.0), (154, 35.0), (666, 28.0), (850, 17.9), (1325, 40.0), (717, 19.5), (141, 8.4), (554, 29.0), (523, 38.0), (208, 12.8), (669, 70.0), (996, 10.6), (1460, 3.9), (456, 12.9), (499, 2.7), (128, 4.4), (108, 29.0), (352, 30.0), (347, 34.0), (28, 56.0), (274, 247.0), (711, 105.0), (670, 129.0), (1524, 24.0), (1092, 18.0), (689, 64.0), (349, 9.6), (63, 58.0), (561, 2.6), (75, 23.0), (944, 91.0), (99, 18.8), (1364, 7.2), (375, 58.0), (113, 17.8), (62, 1.0), (816, 42.0), (359, 12.2), (230, 28.0), (1316, 5.7), (616, 0.5), (1320, 36.0)]
['Dark Souls Prepare to Die Edition', 'Stronghold HD', 'Dark Messiah of Might & Magic Single Player', 'Age of Wonders 2', 'Borderlands 2', 'BioShock 2', 'Max Payne 3', 'Fallen Enchantress Legendary Heroes', 'Fable - The Lost Chapters', 'Mark of the Ninja', 'Brtal Legend', 'XCOM Enemy Unknown', 'Valkyria Chronicles', 'Total War SHOGUN 2', 'The Darkness II', 'The Swapper', 'Magicka', 'Napoleon Total War', 'METAL GEAR RISING REVENGEANCE', 'Dishonored', 'Darksiders', 'The Elder Scrolls III Morrowind', "Sid Meier's Civilization V", 'Mount & Blade Warband', 'Space Engineers', 'Medieval II Total War', 'The Incredible Adventures of Van Helsing', 'Vampire The Masquerade - Bloodlines', 'Crusader Kings II', 'Star Wars Knights of the Old Republic', 'Deus Ex Human Revolution', 'Warhammer 40,000 Space Marine', 'Saints Row IV', 'NBA 2K15', 'Rogue Legacy', 'X3 Terran Conflict', 'Kingdoms of Amalur Reckoning', 'Metro Last Light', 'Portal 2', 'Company of Heroes (New Steam Version)', 'Alan Wake', 'Reus', 'Shadowrun Returns', 'Hitman Absolution', 'Age of Wonders III']
[(987, 97.0)]
['Supreme Commander 2']
[(21, 25.0), (9, 0.9)]
['Dota 2', 'Team Fortress 2']
'''
random_user_sample = random.sample(data, 4)
for x in random_user_sample:
    print(x)
    print([idx_to_game[y] for y, _ in x])


def random_multinomial_choice(tuples):
    x = random.random()
    total = sum([hours for _, hours in tuples])
    for game, hours in tuples:
        y = hours / total
        if y >= x:
            return game
        x -= y
    return tuples[-1][0]


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
        label = random_multinomial_choice(rand_list)
        tuples = []
        for x, _ in rand_list:
            for y, _ in rand_list:
                if x != label and y != label and x != y:
                    tuples.append((x, label))
                    tuples.append((y, label))
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
    train_labels = tf.placeholder(tf.float32, shape=[num_labels, 1])
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
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)

    # seq_ids only needs to be generated once so do this as a numpy array rather than a tensor.
    seq_ids = np.zeros(batch_size, dtype=np.int32)
    cur_id = -1
    for i in range(batch_size):
        if i % context_window == 0:
            cur_id += 1
        seq_ids[i] = cur_id
    print(seq_ids)

    # use segment_sum to add together the related words and reduce the output to be num_labels in size.
    final_embed = tf.segment_sum(embed, seq_ids)
    # final_embed = tf.reshape(final_embed, [int(batch_size / context_window), embedding_size])
    # print('Avg embedding size: {}'.format(final_embed.get_shape().as_list()))

    # Compute the softmax loss, using a sample of the negative labels each time.
    print('weights: {}'.format(weights.get_shape().as_list()))
    print('biases: {}'.format(biases.get_shape().as_list()))
    print('final_embed: {}'.format(final_embed.get_shape().as_list()))

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
        reworked_batch_labels = []
        for idx, el in enumerate(batch_labels):
            if idx % 2 == 0:
                reworked_batch_labels.append(el)
        feed_dict = {train_dataset: batch_data, train_labels: reworked_batch_labels}
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
    pylab.savefig('visuals/tsne_cbow_weighted.png', bbox_inches='tight')


words = [idx_to_game[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)

# Save data
pickle_data = {
    'embeddings': final_embeddings,
    'idx_to_game': idx_to_game,
    'game_to_idx': game_to_idx
}

np.save('saves/embeddings_cbow_weighted.npy', pickle_data)
print('Data saved to embeddings_cbow_weighted.npy')

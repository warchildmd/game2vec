#game2vec - games embeddings 

TensorFlow implementation of word2vec applied on https://www.kaggle.com/tamber/steam-video-games dataset, using both CBOW and Skip-gram.

Context for each game is extracted from the other games that the user owns. For example if a user has three games: Dota 2, CS: GO, and Rocket League, this (input -> label) pairs can be generated:

 * CBOW: ((Dota 2, CS: GO) -> Rocket League), ((Dota 2, Rocket League) -> CS: GO), ((CS: GO, Rocket League) -> Dota 2)
 * Skip-gram: (Rocket League -> (Dota 2, CS: GO)), (CS: GO -> (Dota 2, Rocket League)), (Dota 2 -> (CS: GO, Rocket League))


For more reference, please have a look at this papers:
 
 * [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
 * [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
 * [Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method](http://arxiv.org/pdf/1402.3722v1.pdf)
 
There are three training scripts:

 * **cbow.py** - training using CBOW, using both purchase and play actions into account as user context.
 * **cbow_weighted.py** - same as above, but, only play actions are taken into consideration, and the label is selected based on time played (more time played the game - higher the probability of being selected).
 * **skipgram.py** - training using Skip-gram, using both purchase and play actions into account as user context.
 
Each script outputs an image with the game embeddings visualised using t-SNE, and a .npy file containing embeddings, dictionary, and reverse dictionary:
```
{
    'embeddings': final_embeddings,
    'idx_to_game': idx_to_game,
    'game_to_idx': game_to_idx
}
```

There are three notebooks that load the embeddings + some examples:

 * **player_cbow.py** - loads embeddings_cbow.npy + example usage. 
 * **player_cbow_weighted.py** - loads embeddings_cbow_weighted.npy + example usage.
 * **player_skipgram.py** - loads embeddings_skipgram.npy + example usage.
 
Visualisations are found in .png files.

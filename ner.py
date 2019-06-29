from flair.data import Corpus
from flair.datasets import WIKINER_ENGLISH
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List

def train_ner(embedding, resultdir):
    # 1. get the corpus
    corpus: Corpus = WIKINER_ENGLISH().downsample(0.1)
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [

        WordEmbeddings(embedding),
        # comment in this line to use character embeddings
        # CharacterEmbeddings(),

        # comment in these lines to use flair embeddings
        # FlairEmbeddings('news-forward'),
        # FlairEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train(resultdir,
                learning_rate=0.1,
                mini_batch_size=32,
                max_epochs=150,
                monitor_test=True)

if __name__ == "__main__":
    embedding = '/mnt/mleszczy/results/embs/wiki/w2v_cbow_wiki.en.txt_2018_seed_1234_dim_400_lr_0.05.50.w.txt'
    resultdir = 'resources/taggers/example-ner-wiki'
    train_ner(embedding, resultdir)
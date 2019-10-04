from flair.data import Corpus, MultiCorpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
import numpy as np
import random
import torch
import argparse
from pathlib import Path
from flair.training_utils import EvaluationMetric
import time
import os

def save_random_train_set(corpus, path):
    stamp = time.time()

    if not os.path.exists(path):
        os.makedirs(path)

    dev_name = str(path) + str(stamp)+"_dev"
    test_name = str(path) + str(stamp)+"_test"
    train_name = str(path) + str(stamp)+"_train"
    
    with open(dev_name, 'w') as dev_f:
        for sent in corpus.dev:
            dev_f.write(str(sent.to_plain_string()) + "\n\n")
    dev_f.close()
    
    with open(test_name, 'w') as test_f:
        for sent in corpus.test:
            test_f.write(str(sent.to_plain_string()) + "\n\n")
    test_f.close()
    
    with open(train_name, 'w') as train_f:
        for sent in corpus.train:
            train_f.write(str(sent.to_plain_string()) + "\n\n")
    train_f.close()
    

def train_ner(embed_path, resultdir, datadir='resources/tasks', lr=0.1, use_crf=False, finetune=True, proportion=1.0, hidden_units=256):
    # 1. get the corpus
    multiple = int(1/proportion)
    one_copy: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, base_path=datadir)
    one_copy = one_copy.downsample(proportion)
    one_stats = one_copy.obtain_statistics()
    print(one_stats)

    # save for future analysis
    #save_random_train_set(one_copy, datadir + "/conll_03/proportion_" + str(proportion) + "/")
    
    corpus: MultiCorpus = MultiCorpus([one_copy]*multiple)
    stats = corpus.obtain_statistics()
    print(stats)
    print(corpus)
    
    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
       	
        WordEmbeddings(embed_path)
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=use_crf,
                                            relearn_embeddings=finetune)

    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    
    # 7. start training
    trainer.train(resultdir,
                learning_rate=lr,
                mini_batch_size=32,
                max_epochs=1,
                monitor_test=True)


def eval_ner(embed_path, resultdir, datadir='resources/tasks', use_crf=False):
    # 1. get the corpus
    #corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, base_path=datadir)
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        
        WordEmbeddings(embed_path)
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=use_crf,
                                            relearn_embeddings=False)

    # load checkpoitns
    checkpoint = tagger.load_checkpoint(f'{resultdir}/best-model.pt')

    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer = ModelTrainer.load_from_checkpoint(checkpoint, corpus)
    dev_micro_f1_score, test_micro_f1_score, dev_detailed_results_dict = trainer.final_test(Path(resultdir),
        embeddings_in_memory=True,
        evaluation_metric=EvaluationMetric.MICRO_F1_SCORE,
        eval_mini_batch_size=32)

    return dev_micro_f1_score, test_micro_f1_score, dev_detailed_results_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--resultdir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--use_crf", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument("--eval", action='store_true')
    args = parser.parse_args()
    seed = args.seed
    embedding = args.embedding
    resultdir = args.resultdir
    print('Setting seeds')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)
    if not args.eval:
        train_ner(embedding, resultdir, use_crf=args.use_crf, lr=args.lr, finetune=finetune)
    else:
        eval_ner(embedding, resultdir, use_crf=args.use_crf)

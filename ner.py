#import sys
#sys.path.insert(0, '/flair/')
from flair.data import Corpus
import flair.datasets
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List
import os
import numpy as np
import random
import torch
import argparse
from pathlib import Path
from flair.training_utils import EvaluationMetric
import gensim

def train_ner(cmdline_args, use_cuda = True):
    #0. setup

    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--resultdir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--use_crf", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--datadir", type=str, required=True)

    args = parser.parse_args(cmdline_args)
    embed_path = args.embedding
    resultdir = os.path.dirname(embed_path)
    datadir = args.datadir
    use_crf = args.use_crf
    lr = args.lr
    finetune = args.finetune
    seed = args.seed 

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        #WordEmbeddings(embedding)
	WordEmbeddings('glove')#,
	
	#PooledFlairEmbeddings('news_forward', pooling='min'),
	
	#PooledFlairEmbeddings('news_backward',pooling='min')
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)
                                            #use_crf=use_crf)
                                            #relearn_embeddings=finetune)

    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train(resultdir,
                #learning_rate=lr,
                #mini_batch_size=32,
                max_epochs=150) 
                #monitor_test=True)

    f1_scores, exact_match_scores = eval_ner(cmdline_args)
    print(f1_scores)
    return f1_scores, exact_match_scores

def eval_ner(cmdline_args):
    #def eval_ner(embedding, resultdir, datadir='resources/tasks', use_crf=False):

    # Parse cmdline args and setup environment

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--resultdir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--use_crf", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--datadir", type=str, required=True)

    args = parser.parse_args(cmdline_args)
    embed_path = args.embedding
    resultdir = os.path.dirname(embed_path)
    datadir = args.datadir
    use_crf = args.use_crf
    lr = args.lr
    finetune = args.finetune
    seed = args.seed 

    # convert our custom embeddings to gensim format
    #embedding = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=False)
    #embedding.save(embed_path)

    print('Setting seeds')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)

    # 1. get the corpus
    corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, base_path=args.datadir)
    #corpus = flair.datasets.UD_ENGLISH()
 
    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        #WordEmbeddings(embedding),
        WordEmbeddings('glove')#,
    	#PooledFlairEmbeddings('news_forward', pooling='min'),
	#PooledFlairEmbeddings('news_backward',pooling='min')
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=use_crf)
                                            #relearn_embeddings=False)

    # load checkpoitns
    checkpoint = tagger.load_checkpoint(f'{resultdir}/best-model.pt')

    # 6. initialize trainer
    from flair.trainers import ModelTrainer
    
    trainer: ModelTrainer = ModelTrainer.load_from_checkpoint(checkpoint, corpus)
    score = trainer.final_test(base_path=Path(resultdir),
                       #embeddings_in_memory=True, 
                       #evaluation_metric=EvaluationMetric.MICRO_F1_SCORE,
                       eval_mini_batch_size=32)
    
    f1_scores = []
    exact_match_scores = []
    f1_scores.append(score)
    return f1_scores, exact_match_scores

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="")
#    parser.add_argument("--embedding", type=str, required=True)
#    parser.add_argument("--resultdir", type=str, required=True)
#    parser.add_argument("--seed", type=int, required=True)
#    parser.add_argument("--lr", type=float, required=True)
#    parser.add_argument("--use_crf", type=bool, default=False)
#    parser.add_argument("--finetune", type=bool, default=False)
#    parser.add_argument("--eval", action='store_true')
#    args = parser.parse_args()
#    seed = args.seed
#    embedding = args.embedding
#    resultdir = args.resultdir
# if not args.eval:
#       train_ner(embedding, resultdir, use_crf=args.use_crf, lr=args.lr, finetune=finetune)
#   else:
#       eval_ner(embedding, resultdir, use_crf=args.use_crf)
    
#def main(args):
#    f1_scores, exact_match_scores = eval_ner(args)
#    return f1_scores, exact_match_scores

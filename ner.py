#import sys
#sys.path.insert(0, '/flair/')
from flair.data import Corpus
import flair.datasets
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
import os
import numpy as np
import random
import torch
import argparse
from pathlib import Path
from flair.training_utils import EvaluationMetric

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
    embedding = args.embedding
    resultdir = os.path.dirname(embedding)
    datadir = args.datadir
    use_crf = args.use_crf
    lr = args.lr
    finetune = args.finetune
    seed = args.seed 

    print('Setting seeds')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)

    # 1. get the corpus
    #    corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.UD_ENGLISH, base_path=args.datadir)
    corpus = flair.datasets.UD_ENGLISH()
    #print(len(corpus))

#    with open('tmp/eng.testb.bioes', 'w') as f: 
#        # go through each sentence
#        for sentence in corpus.test:
#
#            # go through each token of sentence
#            for token in sentence:
#                # print what you need (text and NER value)
#                f.write(f"{token.text}\t{token.get_tag('ner').value}\n")
#
#            # print newline at end of each sentence
#            f.write('\n') 

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings(embedding)
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

    f1_scores, exact_match_scores = eval(cmdline_args)
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
    embedding = args.embedding
    resultdir = os.path.dirname(embedding)
    datadir = args.datadir
    use_crf = args.use_crf
    lr = args.lr
    finetune = args.finetune
    seed = args.seed 

    print('Setting seeds')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)

    # 1. get the corpus
    corpus = flair.datasets.UD_ENGLISH()
 
    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings(embedding),
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
    trainer.final_test(Path(resultdir),
        embeddings_in_memory=True,
        evaluation_metric=EvaluationMetric.MICRO_F1_SCORE,
        eval_mini_batch_size=32)
    
    f1_scores = []
    exact_match_scores = []
    f1_scores.append(EvaluationMetric.MACRO_F1_SCORE)
    print("MACRO_F1_SCORE")
    print(EvaluationMetric.MACRO_F1_SCORE)
    print("MICRO_F1_SCORE")
    print(EvaluationMetric.MICRO_F1_SCORE)
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

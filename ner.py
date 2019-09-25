from flair.data import Corpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
import numpy as np
import random
import torch
import argparse
from pathlib import Path
from flair.training_utils import EvaluationMetric
import gensim
import tempfile

def train_ner(embed_path, resultdir, datadir, use_crf=False, lr=0.1, finetune=True):
    # 1. get the corpus
    corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, base_path=datadir)
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    #ATTEMPT 1
    #embedding, wordlist = load_embeddings(embed_path)
    #with open(embed_path, 'rb') as f:
    #	embed = pickle.load(f, encoding='bytes')

    
    #ATTEMPT 2	
    #embedding, wordlist = load_embeddings(embed_path)
    #save_path = resultdir + "/embedding.kv"
    #save_embeddings(save_path, embedding, wordlist)
    #f = open(embed_path, 'r', encoding='utf-8')
    #content=f.read()
    #f.close()
    #print("opened as utf-8")
    #f = open(save_path, 'w', encoding='latin1', errors='ignore')
    #f.write(content)
    #f.close()
    #print("saved as latin1")


    #ATTEMPT 3
    #f = open(embed_path, 'r')
    #gensim.utils.SaveLoad.save(embed_path)
    #gensim.models.KeyedVectors.load(save_path)
    
    print("made it past first gensim call")
    embedding_types: List[TokenEmbeddings] = [
       	WordEmbeddings(embed_path)
	#gensim.KeyedVectors.load_word2vec_format(datapath(save_path), binary=False)
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
                max_epochs=150,
                monitor_test=True)

def eval_ner(embedding, resultdir, datadir, use_crf=False):
    # 1. get the corpus
    corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, base_path=datadir)
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [

        gensim.models.KeyedVectors.load(str(embeddings))
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
    micro_f1_score = trainer.final_test(Path(resultdir),
        embeddings_in_memory=True,
        evaluation_metric=EvaluationMetric.MICRO_F1_SCORE,
        eval_mini_batch_size=32)

    return micro_f1_score

def load_embeddings(path):
    """
    Loads a GloVe or FastText format embedding at specified path. Returns a
    vector of strings that represents the vocabulary and a 2-D numpy matrix that
    is the embeddings.
    """
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        wordlist = []
        embeddings = []
        if is_fasttext_format(lines): lines = lines[1:]
        for line in lines:
            row = preprocess_embedding_file_line(line)
            # if '<Lua heritage>' in line: # this is the special case for translation iwslt14 task embedding
            #     row = ['<Lua heritage>'] + line.strip('\n').split(' ')[2:]
            # else:
            #     row = line.strip('\n').split(' ')
            wordlist.append(row.pop(0))
            embeddings.append([float(i) for i in row])
        embeddings = np.array(embeddings)
    assert len(wordlist) == embeddings.shape[0], 'Embedding dim must match wordlist length.'
    return embeddings, wordlist

def is_fasttext_format(lines):
    first_line = lines[0].strip('\n').split(' ')
    return len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit()

def preprocess_embedding_file_line(line, for_fairseq=False):
    if '<Lua heritage>' in line: # this is the special case for translation iwslt14 task embedding
        if for_fairseq:
            # we add this branch, because we want to strictly follow the preprocessing in the parse embedding function in fairseq utils.py
            row = ['<Lua heritage>'] + line.rstrip().split(" ")[2:]            
        else:
            row = ['<Lua heritage>'] + line.strip('\n').split(' ')[2:]
    else:
        if for_fairseq:
            row = line.rstrip().split(" ")
        else:
            row = line.strip('\n').split(' ')
    return row

def save_embeddings(path, embeds, wordlist):
    ''' save embeddings in text file format'''
    with open(path, 'w', encoding='latin1', errors="ignore") as f:
        for i in range(len(wordlist)):
            strrow = ' '.join([str(embed) for embed in embeds[i,:]])
            f.write('{} {}\n'.format(wordlist[i], strrow))

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

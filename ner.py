from flair.data import Corpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
import numpy as np
import csv
import random
import torch
import argparse
from pathlib import Path
from flair.training_utils import EvaluationMetric
import gensim
import tempfile

def train_ner(embed_path, resultdir, datadir, lr, use_crf=False, finetune=True):
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
       	WordEmbeddings(embed_path)
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SeqaauenceTagger

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

def eval_ner(embed_path, resultdir, datadir, use_crf=False):
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
    dev_micro_f1_score, test_micro_f1_score, dev_PER, dev_ORG, dev_LOC, dev_MISC = trainer.final_test(Path(resultdir),
        embeddings_in_memory=True,
        evaluation_metric=EvaluationMetric.MICRO_F1_SCORE,
        eval_mini_batch_size=32)

    return dev_micro_f1_score, test_micro_f1_score, dev_PER, dev_ORG, dev_LOC, dev_MISC


def get_sentences(path):
    num_errors = 0
    sentences = {}
    counts = {}
    err_counts = {}
    entity_counts = {}
    total_words = 0
    unique_words = 0
    unique_errors = 0
    total_entities = 0
    unique_entities = 0
    with open(path) as f:
        sentence = ""
        errors = []
        for line in f: 
            dirty_row = line.split()
            if len(dirty_row) == 4:
                sentence = sentence + ' ' + str(dirty_row[0])
                
                # counts entities                 
                if dirty_row[0] in counts:
                    counts[dirty_row[0]]['count'] = counts[dirty_row[0]]['count'] + 1
                    counts[dirty_row[0]]['true_tags'].append(dirty_row[1])
                    total_words += 1
                else:
                    counts[dirty_row[0]] = {'count':1, 'true_tags':[dirty_row[1]]}
                    unique_words = unique_words + 1
                    total_words += 1
                
                # counts entities                 
                if dirty_row[0] in entity_counts and dirty_row[1] != "O":
                    entity_counts[dirty_row[0]]['count'] = entity_counts[dirty_row[0]]['count'] + 1
                    entity_counts[dirty_row[0]]['true_tags'].append(dirty_row[1])
                    total_entities += 1
                elif dirty_row[0] not in entity_counts and dirty_row[1] != "O":
                    entity_counts[dirty_row[0]] = {'count':1, 'true_tags':[dirty_row[1]]}
                    unique_entities += 1
                    total_entities += 1
                
                # is this word an error
                if dirty_row[1] != dirty_row[2]:
                    errors.append(dirty_row)
                    if dirty_row[0] in err_counts:
                        err_counts[dirty_row[0]] = err_counts[dirty_row[0]] + 1
                    else:
                        err_counts[dirty_row[0]] = 1
                        unique_errors = unique_errors + 1
            else:
                sentences[sentence] = errors
                sentence=''
                errors=[]
      
    f.close()
    for item in sentences:
        for error in sentences[item]:
            total_count = counts[error[0]]['count']
            err_count = err_counts[error[0]]
            error.append(str(total_count))
            error.append(str(err_count))
            #print(error)
            num_errors += 1

    # Outputs
    print("PATH:")
    print(path)
    print()
    print("FORMAT ERRORS:")
    print("['Word', 'True tag', 'Predicted tag', 'Prediction confidence', 'Total occurrences of word in dataset', 'Erroneous predictions of word']")
    print()
    print("SUMMARY STATISTICS:")
    print("Total words (note not just entities): " + str(total_words))
    print("Unique words: " + str(unique_words))
    print("Words that appeared 1x: " + str(len([k for k,v in counts.items() if v['count']==1])))
    print("Words that appeared 2x: " + str(len([k for k,v in counts.items() if v['count']==2])))
    print("Words that appeared 3x: " + str(len([k for k,v in counts.items() if v['count']==3])))
    print("Words that appeared 4x: " + str(len([k for k,v in counts.items() if v['count']==4])))
    print()
    print("Total entities: " + str(total_entities))
    print("Unique entities: " + str(unique_entities))
    print("Entities that appeared 1x: " + str(len([k for k,v in entity_counts.items() if v['count']==1])))
    print("Entities that appeared 2x: " + str(len([k for k,v in entity_counts.items() if v['count']==2])))
    print("Entities that appeared 3x: " + str(len([k for k,v in entity_counts.items() if v['count']==3])))
    print("Entities that appeared 4x: " + str(len([k for k,v in entity_counts.items() if v['count']==4])))
    print()
    print("Total errors: " + str(num_errors))
    print("Unique errors: " + str(unique_errors))
    print("Errors on word that appeared 1x: " + str(len([k for k,v in err_counts.items() if v==1])))
    print("Errors on word that appeared 2x: " + str(len([k for k,v in err_counts.items() if v==2])))
    print("Errors on word that appeared 3x: " + str(len([k for k,v in err_counts.items() if v==3])))
    print("Errors on word that appeared 4x: " + str(len([k for k,v in err_counts.items() if v==4])))
    print("\n")
    return sentences

def compare_paths(path_1, path_2, true_tag=None, guessed_tag=None, count=1, eval_type="all"):
 
    sentences_1 = get_sentences(path_1)
    sentences_2 = get_sentences(path_2)

    path_1_count = 0
    path_2_count = 0 
    if eval_type == "all":
        for sentence in sentences_1:
            if len(sentences_1[sentence]) > 0 or len(sentences_2[sentence]) > 0:
                print(sentence)
                print("PATH 1")
                for error in sentences_1[sentence]:
                    print(error)
                print("\nPATH 2")
                for error in sentences_2[sentence]:
                    print(error)
                print()
                print('*********************************************')
                print()
    
    if eval_type == "some_tags":
        for sentence in sentences_1:
            to_print=False
            if len(sentences_1[sentence]) > 0 or len(sentences_2[sentence]) > 0:
                output = sentence + "\n" 
                output += "PATH 1:\n"
                for error in sentences_1[sentence]:
                    if error[1] in true_tag and error[2] in guessed_tag:
                        output+=', '.join(error)
                        output+="\n"
                        path_1_count += 1
                        to_print = True
                output+="\nPATH 2:\n"
                for error in sentences_2[sentence]:
                    if error[1] in true_tag and error[2] in guessed_tag:
                        output+=', '.join(error)
                        output+="\n"
                        path_2_count += 1
                        to_print = True
                if to_print:
                    print(output)
                    print()
                    print('*********************************************')
                    print()


    if eval_type == "count_based":
        for sentence in sentences_1:
            to_print=False
            if len(sentences_1[sentence]) > 0 or len(sentences_2[sentence]) > 0:
                output = sentence + "\n" 
                output += "PATH 1:\n"
                for error in sentences_1[sentence]:
                    if str(error[4]) == str(count):
                        output+=', '.join(error)
                        output+="\n"
                        path_1_count += 1
                        to_print = True
                output+="\nPATH 2:\n"
                for error in sentences_2[sentence]:
                    if str(error[4]) == str(count):
                        output+=', '.join(error)
                        output+="\n"
                        path_2_count += 1
                        to_print = True
                if to_print:
                    print(output)
                    print()
                    print('*********************************************')
                    print()

    print("path_1_count = " + str(path_1_count))
    print('path_2_count = ' + str(path_2_count))
        # interesting concepts: confidently worse, precision errors, recall errors, both wrong, one wront

if __name__ == "__main__":
    #get_sentences('/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/seed,1_embeddim,300_compresstype,nocompress_bitrate,32/ner_ler_0.1/dev.tsv')
    #parser = argparse.ArgumentParser(description="")
    #parser.add_argument("--eval_type", type=str, required=True)
    #args = parser.parse_args()
    other=['O'] 
    all_classes=['S-PER', 'E-PER', 'I-PER', 'B-PER', 'O-PER', 
                 'S-LOC', 'E-LOC', 'I-LOC', 'B-LOC', 'O-LOC', 
                 'S-ORG', 'E-ORG', 'I-ORG', 'B-ORG', 'O-ORG', 
                 'S-MISC', 'E-MISC', 'I-MISC', 'B-MISC', 'O-MISC']
    compare_paths('/proj/smallfry/embeddings/glove400k/2019-08-28-randV1/embeddim,300_compresstype,random_seed,1_randembeddim,800/ner_ler_0.1/dev.tsv', 
                  '/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/seed,1_embeddim,300_compresstype,nocompress_bitrate,32/ner_ler_0.1/dev.tsv', 
                  true_tag=all_classes, 
                  guessed_tag=other,
                  eval_type="some_tags")

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
#    print('Setting seeds')
#    torch.manual_seed(seed)
#    if torch.cuda.is_available():
#        torch.cuda.manual_seed(seed)
#        torch.backends.cudnn.deterministic=True
#    np.random.seed(seed)
#    random.seed(seed)
#    if not args.eval:
#        train_ner(embedding, resultdir, use_crf=args.use_crf, lr=args.lr, finetune=finetune)
#    else:
#        eval_ner(embedding, resultdir, use_crf=args.use_crf)

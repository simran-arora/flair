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
    unique_words = 0
    unique_errors = 0
    with open(path) as f:
        sentence = ""
        errors = []
        for line in f: 
            dirty_row = line.split()
            if len(dirty_row) == 4:
                sentence = sentence + ' ' + str(dirty_row[0])
                
                # counts                
                if dirty_row[0] in counts:
                    counts[dirty_row[0]]['count'] = counts[dirty_row[0]]['count'] + 1
                    counts[dirty_row[0]]['true_tags'].append(dirty_row[1])
                else:
                    counts[dirty_row[0]] = {'count':1, 'true_tags':[dirty_row[1]]}
                    unique_words = unique_words + 1
                
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
        if len(sentences[item]) > 0:
            #print()
            #print(item)
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
    print("Total errors: " + str(num_errors))
    print("Unique words: " + str(unique_words))
    print("Words that appeared 1x: " + str(len([k for k,v in counts.items() if v['count']==1])))
    print("Words that appeared 2x: " + str(len([k for k,v in counts.items() if v['count']==2])))
    print("Words that appeared 3x: " + str(len([k for k,v in counts.items() if v['count']==3])))
    print("Words that appeared 4x: " + str(len([k for k,v in counts.items() if v['count']==4])))
    print()
    print("Unique errors: " + str(unique_errors))
    print("Errors on word that appeared 1x: " + str(len([k for k,v in err_counts.items() if v==1])))
    print("Errors on word that appeared 2x: " + str(len([k for k,v in err_counts.items() if v==2])))
    print("Errors on word that appeared 3x: " + str(len([k for k,v in err_counts.items() if v==3])))
    print("Errors on word that appeared 4x: " + str(len([k for k,v in err_counts.items() if v==4])))
    return sentences


#a watch out for words that are incorrect multiple times
def compare_paths(path_1, path_2, eval_type):
    path_1_tokens = {}
    path_2_tokens = {}
    counts_1 = {}
    counts_2 = {}
    i = 0
    unique_words_1 = 0
    unique_words_2 = 0
    size_1 = 0
    size_2 = 0

    with open(path_1) as f_1:
        reader = csv.DictReader(f_1, delimiter=" ", quotechar='"', fieldnames=['word', 'true_tag', 'expected_tag', 'confidence'])
        for row in reader:
            if row['word'] in counts_1:
                counts_1[row['word']] = counts_1[row['word']] + 1
                if row['true_tag'] != row['expected_tag']:
                    path_1_tokens[row['word']].append({'true_tag':row['true_tag'], 'expected_tag':row['expected_tag'], 'confidence':row['confidence'], 'counts':counts_1[row['word']]})
            else:
                counts_1[row['word']] = 1 
                unique_words_1 = unique_words_1 + 1
                if row['true_tag'] != row['expected_tag']:
                    path_1_tokens[row['word']] = [{'true_tag':row['true_tag'], 'expected_tag':row['expected_tag'], 'confidence':row['confidence'], 'counts':counts_1[row['word']]}]
    
    with open(path_2) as f_2:
        reader = csv.DictReader(f_2, delimiter=" ", quotechar='"', fieldnames=['word', 'true_tag', 'expected_tag', 'confidence'])
        for row in reader:
            if row['word'] in counts_2:
                counts_2[row['word']] = counts_2[row['word']] + 1
                if row['true_tag'] != row['expected_tag'] and row['word'] != "" and row['true_tag'] != "":
                    path_2_tokens[row['word']].append({'true_tag':row['true_tag'], 'expected_tag':row['expected_tag'], 'confidence':row['confidence'], 'counts':counts_2[row['word']]})
            else:
                counts_2[row['word']] = 1 
                unique_words_2 = unique_words_2 + 1
                if row['true_tag'] != row['expected_tag'] and row['word'] != "" and row['true_tag'] != "":
                    path_2_tokens[row['word']] = [{'true_tag':row['true_tag'], 'expected_tag':row['expected_tag'], 'confidence':row['confidence'], 'counts':counts_2[row['word']]}]
 
    if eval_type == "path_1_worse":
        for item in path_1_tokens:
            if item not in path_2_tokens:
                print(item + " " + str(path_1_tokens[item]))
                i = i + 1
    elif eval_type == "path_1_confidently_worse":
        for item in path_1_tokens:
            for tok in item:
                if float(path_1_tokens[tok]['confidence']) > 0.9:
                    if tok not in path_2_tokens:
                        print(tok + " " + str(path_1_tokens[tok]))
                        i = i + 1
    elif eval_type == "path_2_worse":
        for item in path_2_tokens:
            for tok in item:
                if tok not in path_1_tokens:
                    print(tok + " " + str(path_2_tokens[tok]))
                    i = i + 1
    elif eval_type == "path_2_confidently_worse":
        for item in path_2_tokens:
            for tok_2 in item:
                if float(path_2_tokens[tok_2]['confidence']) > 0.9:
                    if tok_2 not in path_1_tokens:
                        print(tok_2 + " " + str(path_2_tokens[tok_2]))
                        i = i + 1
    elif eval_type == "path_1_errors":
        for item in path_1_tokens:
            for tok in item:
                i = i + 1
                print(str(tok) + " " + str(path_1_tokens[tok]))
    elif eval_type == "path_2_errors":
        for item in path_2_tokens:
            i = i + 1
            #print("\n")
            print(str(item) + " " + str(path_2_tokens[item]))
    elif eval_type == "path_1_count_1":
        for item in path_1_tokens:
            if path_1_tokens[item]['counts'] == 2:
                i = i + 1
    elif eval_type == "path_2_count_1":
        for item in path_2_tokens:
            if path_2_tokens[item]['counts'] == 2:
                i = i + 1
    elif eval_type == "both_wrong":
        for tok_1 in path_1_tokens:
            if tok_1 in path_2_tokens:
                print(tok_1 + " " + str(path_1_tokens[tok_1]))
                print(tok_1 + " " + str(path_2_tokens[tok_1]))
                print("")
                i = i + 1
    elif eval_type == "path_1_precision":
        for tok_1 in path_1_tokens:
            if path_1_tokens[tok_1]['true_tag'] != "O" and path_1_tokens[tok_1]['expected_tag'] == "O":
                i = i + 1
    elif eval_type == "path_2_precision":
        for tok_2 in path_2_tokens:
            if path_2_tokens[tok_2]['true_tag'] != "O" and path_2_tokens[tok_2]['expected_tag'] == "O":
                i = i + 1
    elif eval_type == "path_1_recall":
        for tok_1 in path_1_tokens:
            if path_1_tokens[tok_1]['true_tag'] == "O" and path_1_tokens[tok_1]['expected_tag'] != "O":
                i = i + 1
    elif eval_type == "path_2_recall":
        for tok_2 in path_2_tokens:
            if path_2_tokens[tok_2]['true_tag'] == "O" and path_2_tokens[tok_2]['expected_tag'] != "O":
                i = i + 1
 
    print("Total count: " + str(i))
    print("Eval type: " + str(eval_type))
    print("Path 1: " + str(path_1))
    print("Path 2: " + str(path_2))
    f_1.close()
    f_2.close()

def count_occurrences(path, word):
    path_1_tokens = {}
    counts = {}
    unique_words = 0
    size = 0
    with open(path) as f_1:
        reader = csv.DictReader(f_1, delimiter=" ", quotechar='"', fieldnames=['word', 'true_tag', 'expected_tag', 'confidence'])
        for row in reader:
            size = size + 1
            if row['word'] in counts:
                counts[row['word']] = counts[row['word']] + 1
            else:
                counts[row['word']] = 1 
                unique_words = unique_words + 1
            if row['true_tag'] != row['expected_tag']:
                path_1_tokens[row['word']] = {'true_tag':row['true_tag'], 'expected_tag':row['expected_tag'], 'confidence':row['confidence'], 'counts':counts[row['word']]}
    
    for item in path_1_tokens:
        print(str(item) + " " + str(path_1_tokens[item]))
    
    average_count = size/unique_words
    print("Average Count: " + str(average_count))
    print("The word " + str(word) + " occurs " + str(counts[word]) + " times.")
    print("Path: " + path)
    f_1.close()


if __name__ == "__main__":
    get_sentences('/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/seed,1_embeddim,300_compresstype,nocompress_bitrate,32/ner_ler_0.1/dev.tsv')
    #count_occurrences('/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/seed,1_embeddim,300_compresstype,nocompress_bitrate,32/ner_ler_0.1/dev.tsv', 'Seattle')
    #parser = argparse.ArgumentParser(description="")
    #parser.add_argument("--eval_type", type=str, required=True)
    #args = parser.parse_args()
    #compare_paths('/proj/smallfry/embeddings/glove400k/2019-08-28-randV1/embeddim,300_compresstype,random_seed,1_randembeddim,800/ner_ler_0.1/dev.tsv', 
    #              '/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/seed,1_embeddim,300_compresstype,nocompress_bitrate,32/ner_ler_0.1/dev.tsv', 
    #              args.eval_type)

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

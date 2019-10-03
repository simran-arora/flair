from typing import List
import numpy as np
import csv
import random
import torch
import argparse
from pathlib import Path
import gensim
import tempfile


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

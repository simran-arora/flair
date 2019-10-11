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
    print_bool = False
    if print_bool:
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

# return sentences for which they both missed
def set_intersection(sentences_1, sentences_2):
    sent_set_1 = list(sentences_1.keys())
    sent_set_2 = list(sentences_2.keys())

    for sent in sent_set_1:
        if sentences_1[sent] == []:
            sent_set_1.remove(sent)

    for sent in sent_set_2:
        if sentences_2[sent] == []:
            sent_set_2.remove(sent)

    sent_intersection = list(set(sent_set_1) & set(sent_set_2))
    return sent_intersection

def set_difference(sentences_1, sentences_2):
    sent_set_1 = list(sentences_1.keys())
    sent_set_2 = list(sentences_2.keys())

    for sent in sent_set_1:
        if sentences_1[sent] == []:
            sent_set_1.remove(sent)

    for sent in sent_set_2:
        if sentences_2[sent] == []:
            sent_set_2.remove(sent)

    sent_difference = list(set(sent_set_1) - set(sent_set_2))
    return sent_difference

def compare_paths(path_1, path_2, true_tag=None, guessed_tag=None, count=1, eval_type="all"):
 
    sentences_1 = get_sentences(path_1)
    sentences_2 = get_sentences(path_2)
    
    sent_intersection = set_intersection(sentences_1, sentences_2)

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

    if eval_type == "bucket_confidence":
        # maps from a confidence bucket to errors of this confidence level
        bins = {}
        bins_errors = {}
        bin_thresholds = [0.0, 0.2, 0.4, 0.6, 0.8]
        for bin in bin_thresholds:
            bins[bin] = 0
            bins_errors[bin] = []

        for sentence in sentences_1:
            for error in sentences_1[sentence]:
                confidence = float('%.2f'%float(error[3]))
                i = len(bin_thresholds) - 1
                while(confidence < bin_thresholds[i]):
                    i= i-1
                bins[bin_thresholds[i]]= bins[bin_thresholds[i]] + 1
                bins_errors[bin_thresholds[i]].append(sentence)
              
        for bin in bins:
            print("Bin: " + str(bin) + ", Count: " + str(bins[bin]))

        bins_2 = {}
        bins_errors_2 = {}
        for bin in bin_thresholds:
            bins_2[bin] = 0
            bins_errors_2[bin] = []

        for sentence in sentences_2:
            for error in sentences_2[sentence]:
                confidence = float('%.2f'%float(error[3]))
                i = len(bin_thresholds) - 1
                while(confidence < bin_thresholds[i]):
                    i= i-1
                bins_2[bin_thresholds[i]]= bins_2[bin_thresholds[i]] + 1
                bins_errors_2[bin_thresholds[i]].append(sentence)
              
        for bin in bins_2:
            print("Bin: " + str(bin) + ", Count: " + str(bins_2[bin]))

        #I know which sentences are in each bin for each type of embedding
        bin_intersection = list(set(bins_errors[0.4]) & set(bins_errors_2[0.4]))
        for sent in bin_intersection:   
            print(sent)
            print("PATH 1")
            for err in sentences_1[sent]:
                print(err)
            print("PATH 2")
            for err in sentences_2[sent]:
                print(err)
            print()

    print("path_1_count = " + str(path_1_count))
    print('path_2_count = ' + str(path_2_count))

# want a dict of {doc_number: {sentence_numbers: sentences in this doc}}
# want the counts of words that appear and the tags with which they appear
def test_set_properties(path):
    word_counts = {}
    training_sentences = {}
    with open(path) as f:
        sentence = []
        doc = {}
        doc_num = 0
        sent_num = 0
        for row in f:
            split = row.split()
            word = ''
            if split:
                word = split[0]
                
            if word and word != "-DOCSTART-":
                if word in word_counts:
                    word_counts[word]['counts'] = word_counts[word]['counts'] + 1
                    word_counts[word]['true_tags'].add(split[3])
                else:
                    word_counts[word] = {'counts':1, 'true_tags':{split[3]}}
                sentence.append(split)            

            if not word and sentence: 
                doc[sent_num] = sentence
                sentence = []
                sent_num = sent_num + 1
                
            if word == "-DOCSTART-":
                training_sentences[doc_num] = doc
                doc_num = doc_num + 1
                doc = {}
                sent_num = 1
    
    count_frequencies = {}
    for word in word_counts:
        count = int(word_counts[word]['counts'])
        if count in count_frequencies:
            count_frequencies[count] = count_frequencies[count] + 1
        else:
            count_frequencies[count] = 1

    better_count = {}
    for ct in count_frequencies:
        if count_frequencies[ct] < 100:
            better_count[ct] = count_frequencies[ct]

    import matplotlib.pylab as plt
    lists = sorted(better_count.items())
    x, y = zip(*lists)
    plt.plot(x, y)
    dr = '/proj/smallfry/git/smallfry_internal/src/smallfry/third_party/flair/data_scrape/'
    file_name = '{}training_set_wc.pdf'.format(dr)
    plt.savefig(file_name)   

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
    #compare_paths('/proj/smallfry/embeddings/glove400k/2019-08-30-randV1/embeddim,300_compresstype,randcirc_seed,1_randembeddim,800/proportion_1.0/ner_lr_0.1_5/dev.tsv', 
    #              '/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/seed,1_embeddim,300_compresstype,nocompress_bitrate,32/proportion_1.0/ner_lr_0.1_6/dev.tsv', 
    #              true_tag=all_classes, 
    #              guessed_tag=other,
    #              eval_type="bucket_confidence")
    
    test_set_properties('/proj/smallfry/git/smallfry_internal/src/smallfry/third_party/flair/tests/resources/tasks/conll_03/eng.train')

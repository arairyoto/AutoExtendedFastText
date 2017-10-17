# Python Version of WordNetExtractor.java
import os
import sys
#WordNet
import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn
#fastText multilingual
from fasttext import FastVector

import codecs

import util

class MultiWordNetExtractor:
    def __init__(self, folder, langs):
        self.folder = folder
        for lang in langs:
            if lang not in wn.langs():
                print("language: '%s' is not supported, try another language" % lang)
        self.langs = langs
        #initialize
        self.WordIndex = {}
        self.SynsetIndex = {}
        self.pos_list = ['a', 's', 'r', 'n', 'v']
        self.pointer_map = {"@":"hypernym", "&":"similar", "$":"verbGroup", "!":"antonym"}
        self.Shared = util.Shared()

    def main(self):
        self.model = {}
        # Loading fastText
        self.model['eng'] = FastVector(vector_file='/Users/arai9814/model/wiki.en.vec')
        self.model['jpn'] = FastVector(vector_file='/Users/arai9814/model/wiki.ja.vec')
        self.model['fra'] = FastVector(vector_file='/Users/arai9814/model/wiki.fr.vec')
        # Transform multi-lingual vector to same vector space
        self.model['eng'].apply_transform('alignment_matrices/en.txt')
        self.model['jpn'].apply_transform('alignment_matrices/ja.txt')
        self.model['fra'].apply_transform('alignment_matrices/fr.txt')

        ver = wn.get_version()
        print("RESOURCE: WN " + str(ver) + "\n")
        print("LANGUAGE: "+str(self.langs)+"\n")
        print("VECTORS: " + self.folder + "\n")
        print("TARGET: " + self.folder + "\n")

        self.extractWordsAndSynsets(self.folder + "words.txt",self.folder + "synsets.txt",self.folder + "lemmas.txt")
        self.extractSynsetRelations(self.folder + "hypernym.txt", '@')
        self.extractSynsetRelations(self.folder + "similar.txt",  '&')
        self.extractSynsetRelations(self.folder + "verbGroup.txt",  '$')
        self.extractSynsetRelations(self.folder + "antonym.txt",  '!')

        print("DONE")

    def extractWordsAndSynsets(self, filenameWords, filenameSynsets,  filenameLexemes):
        #file
        fWords = codecs.open(filenameWords, 'w', 'utf-8')
        fSynsets = codecs.open(filenameSynsets, 'w', 'utf-8')
        fLexemes = codecs.open(filenameLexemes, 'w', 'utf-8')

        wordCounter = 0
        wordCounterAll = 0
        synsetCounter = 0
        synsetCounterAll = 0
        lexemCounter = 0
        lexemCounterAll = 0

        ovv = []

        for pos in self.pos_list:
            for synset in wn.all_synsets(pos=pos):
                synsetCounterAll += 1
                synsetId = synset.name()
                self.SynsetIndex[synsetId] = synsetCounterAll

                fSynsets.write(synsetId+" ")

                wordInSynset = 0

                for lang in self.langs:
                    for lemma in synset.lemmas(lang=lang):
                        lexemCounterAll += 1
                        word = lemma.name()
                        wordId = lemma.name()+':'+lang

                        if word in self.model[lang]:
                            wordInSynset += 1
                            if wordId not in self.WordIndex:
                                fWords.write(wordId + " " + self.Shared.getVectorAsString(self.model[lang][word]) + "\n")
                                wordCounter += 1
                                self.WordIndex[wordId] = wordCounter

                            lexemCounter += 1
                            #lemma name
                            sensekey = wordId+':'+synsetId

                            fSynsets.write(sensekey + ",")
                            fLexemes.write(str(self.WordIndex[wordId]) + " " + str(synsetCounterAll) + "\n")
                        else:
                            ovv.append(wordId)

                fSynsets.write("\n")
                if wordInSynset is not 0:
                    synsetCounter += 1
                else:
                    self.SynsetIndex[synsetId] = -1
        fWords.close()
        fSynsets.close()
        fLexemes.close()

        print("   Words: %d / %d\n" % (wordCounter, wordCounter+len(ovv)))
        print("  Synset: %d / %d\n" % (synsetCounter, synsetCounterAll))
        print("  Lexems: %d / %d\n" % (lexemCounter, lexemCounterAll))

    def extractSynsetRelations(self, filename, relation_symbol):
        affectedPOS = {}

        f = codecs.open(filename, 'w', 'utf-8')

        for pos in self.pos_list:
            for synset in wn.all_synsets(pos=pos):
                synsetId = synset.name()
                targetSynsets = synset._related(relation_symbol)
                for targetSynset in targetSynsets:
                    targetSynsetId = targetSynset.name()
                    key = targetSynset.pos()

                    if key in affectedPOS:
                        affectedPOS[key] += 1
                    else:
                        affectedPOS[key] = 1

                    if self.SynsetIndex[synsetId] >= 0 and self.SynsetIndex[targetSynsetId] >= 0:
                        f.write(str(self.SynsetIndex[synsetId]))
                        f.write(" ")
                        f.write(str(self.SynsetIndex[targetSynsetId]))
                        f.write("\n")
        f.close()
        print("Extracted %s: done!\n" % self.pointer_map[relation_symbol])

        for k,v in affectedPOS.items():
            print("  %s: %d\n" % (k, v))

if __name__ == '__main__':
    #path to input word embeddings
    # file_name = '/Users/arai9814/Downloads/GoogleNews-vectors-negative300.bin'
    #path to output folder
    folder = 'fastText/'
    #language
    langs = ['eng','jpn','fra']
    mwne = MultiWordNetExtractor(folder, langs)
    mwne.main()

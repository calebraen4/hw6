# qasystem.py
# This program is a Question-Answering system
# Questions and relevant documents are required
# inputs
# Simply input a directory name containing
# the data

###################################################################
########################### IMPORTS ###############################
###################################################################

# For string manipulation utitlities
import string

# nltk toolkit
import nltk

# Get stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Get ngram computation method
from nltk import ngrams

# Get pos_tag method
from nltk.tag import pos_tag

# Get word_tokenize method
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# Needed for tagger
nltk.download('averaged_perceptron_tagger')

from nltk import RegexpParser
# Get regex
import re

# Numpy
import numpy  

# And math
import math

from sklearn.feature_extraction.text import CountVectorizer

# For word lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') 

# Spacy for NER
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')

###################################################################
################## VARIABLE DECLARACTIONS #########################
###################################################################

# stopwords
stopWords = list(stopwords.words('english'))
# Question markers
qM = ['who', 'when', 'what', 'where', 'why', 'how', 'name']

# QA system class.
class Qasystem:
    # Init method
    # Takes in the data directory
    def __init__(self, dataDir):
        # Initialize directory
        self.dir = dataDir
        # And lemmatizer for future use
        self.lemmatizer = WordNetLemmatizer()
        grammar = "NP: {<DT>?<JJ>*<NN>}"
        self.parser = RegexpParser(grammar)

        # Initialize data files 
        self.questions = dataDir + "/qadata/questions.txt"
        self.docs = dataDir +"/topdocs/top_docs."

    def answer(self):
        # Process questions by stripping stop words and
        # punctuation
        questions = self.processQuestions(self.questions)

        # Initialize dictionary to extract passages
        passages = {}
        answers = {}

        # For each question
        for q in questions:
            # Get top 10 relevant passages
            passages[q] = self.retrievePassages(q, questions[q])
            # And then extract answers from that
            answers[q]  = self.extractAnswers(passages[q], questions[q])

        return answers

    # Process all questions in questionFile by producing a list of keywords
    def processQuestions(self, questionFile):
        # Create dictionary with key qid and value question without
        # stop words and punctuation
        questions = {}

        # Open question file
        with open(questionFile, 'r', encoding='utf-8-sig') as f:
            # Get all lines while stripping trailing characters
            lines = (line.rstrip() for line in f)
            lines = list(line for line in lines if line)

            for i in range(len(lines)):
                if i % 2 == 0:
                    qid = lines[i][lines[i].find(':') + 2:]
                    questions[qid] = self.formulateQuery(lines[i+1])

        return questions
        
    # Given a question, formulate a query
    def formulateQuery(self, question):
        # Lemmatizes question and removes stop words to return a list of questions
        # of words with no duplicate word
        return list(set([word for word in self.lemmatize(question) if word in qM or word not in stopWords]))

    # Retrieves top-10 passages based on cosine similarity
    def retrievePassages(self, qid, question): 
        # list of tuples, tuples contain passage, and cosine sim of question and that passage
        passages = [([], 0)] * 10

        # Create vectorizer with question as vocabulary
        vectorizer = CountVectorizer(vocabulary=question)
        questionV = vectorizer.fit_transform([" ".join(question)])

        # String to hold raw text
        text = ""

        # Open file of topdocs
        with open(self.docs + qid, 'r', encoding='ISO-8859-1') as f:
            text = " ".join([line for line in f if line[0] != "<" and line[0:3] != "Qid"])
        
        # For each sentence check cosine similarity
        for sentence in sent_tokenize(text):
            # Get the sentence without stopWords
            passage = [word for word in word_tokenize(sentence) if word not in stopWords]

            # Create feature vector for passage
            passageV = vectorizer.fit_transform([" ".join(self.lemmatize(" ".join(passage)))])

            # Compute cosine similarity
            cosSim =\
                    numpy.dot(questionV.toarray()[0], passageV.toarray()[0]) /\
                    (numpy.linalg.norm(questionV.toarray()[0]) *\
                    numpy.linalg.norm(passageV.toarray()[0]))

            # And check that it's a number
            if math.isnan(cosSim):
                cosSim = 0
        
            # Compare with minimum result of cosine similarities in passages
            if(cosSim > min(passages, key=lambda x:x[1])[1]):
                passages[passages.index(min(passages, key=lambda x:x[1]))] = \
                    (passage, cosSim)

        # Return
        return passages
    
    # Method to lemmatize a given text
    def lemmatize(self, text):

        # Strip punctuation
        t = text.translate(str.maketrans('', '', string.punctuation))

        # Tag sentence for verbs
        tagged = pos_tag(t.split())
        # Create list to return
        lemmed = []
        for tag in tagged:
            # If a verb, lemmatize as verb
            if tag[1][0] == 'V':
                lemmed.append(self.lemmatizer.lemmatize(tag[0].lower(), 'v'))
            else:
                lemmed.append(self.lemmatizer.lemmatize(tag[0].lower()))

            # Case for what's since it's not working too well
            if lemmed[-1] == "whats":
                lemmed[-1] = "what"

        return lemmed

    # Method to generate ngrams
    # Uses nltk 
    # With n-gram tiling
    def genNgrams(self, passages, n):
        # Create dictionary with key as ngram and frequency as value
        grams = {}
        # For all i from 0 to n

        for i in range(0, n + 1):
            for passage in passages:
                # For each passage
                # Create ngram
                sequences = ngrams(passage[0], i)

                # Treat to get appropriate output
                for seq in sequences:
                    sequence = ""
                    for w in seq:
                        sequence += w + " "
                    sequence = sequence[:-1]

                    if sequence.lower() not in stopWords:
                        if sequence in grams:
                            grams[sequence] += 1

                        else:
                            grams[sequence] = 1

        # Return in decreasing order
        return {k: v for k, v in sorted(grams.items(),reverse=True, key=lambda item: item[1])}

    # Method to extract answer given top 10 passages
    def extractAnswers(self, passages, question):

        # List to store answers
        answers = []

        # Use n as 10 as max length answer is 10
        # Get top 20 ngrams
        c = 0
        grams = []
        for g in self.genNgrams(passages, 10):
            if c == 20:
                break
            grams.append(g)
            c += 1

        # Get the question type
        qType = [word for word in question if word in qM][0]

        # For each sentence in given passages
        for sentence in passages:
            # Treat to get NER
            doc = nlp(" ".join(sentence[0]))
            # Get answer given each sentence
            for a in self.getAnswer(qType, doc, grams):
                if a not in answers:
                    answers.append(a)


        # Check that we have enough answers to return
        for g in grams:
            if len(answers) ==  10:
                break
            if g not in answers:
                answers.append(g)

        return answers
    
    # Method to extract answers given named entities and ngrams
    def getAnswer(self, qT, names, grams):
        # Initialize triggers and pass in to method add
        if qT == "where":
            triggers = ["GPE", "FAC"]
            return self.add(triggers, names, grams)

        elif qT == "when":
            triggers = ["DATE", "EVENT", "TIME"]
            return self.add(triggers, names, grams)

        elif qT == "who":
            triggers = ["PERSON", "NORP", "ORG", "GPE"]
            return self.add(triggers, names, grams)

        elif qT == "what":
            triggers = ["FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "ORG", "GPE"]
            return self.add(triggers, names, grams)

        elif qT == "name":
            triggers = ["PERSON", "WORK_OF_ART", "ORG", "GPE", "LOC", "NORP",
                    "FAC", "PRODUCT", "EVENT", "LAW", "LANGUAGE", "DATE", "TIME"]
            return self.add(triggers, names, grams)

        elif qT == "how":
            triggers = ["CARDINAL", "ORDINAL", "MONEY", "GPE"]
            return self.add(triggers, names, grams)

    # Method to avoid copying code avobe
    def add(self, triggers, names, grams):
        # Create list to return
        a = []
        for ent in names.ents:
            if ent.label_ in triggers and ent.text in grams:
                a.append(ent.text)
        return a
                

# Method to print answers
def printAnswer(answers): 
    for q in answers:
        print(q)
        for a in answers[q]:
            print(a)

# Main
if __name__ == "__main__":
    data = "./hw6_data/training"
    #data= "./hw6_data/test"

    system = Qasystem(data)
    printAnswer(system.answer())

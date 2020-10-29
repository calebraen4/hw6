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


###################################################################
################## VARIABLE DECLARACTIONS #########################
###################################################################

# stopwords
stopWords = list(stopwords.words('english'))
# Question markers
qM = ['who', 'when', 'what', 'where', 'why', 'how']

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
            answers[q]  = self.extractAnswer(passages[q], 3, questions[q])

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
        return list(set([word for word in self.lemmatize(question) if word not in stopWords and word not in qM]))

    # Retrieves top-10 passages based on cosine similarity
    def retrievePassages(self, qid, question): 
        # list of tuples, tuples contain passage, and cosine sim of question and that passage
        passages = [([], 0)] * 10

        # Open file of topdocs
        with open(self.docs + qid, 'r', encoding='ISO-8859-1') as f:

            # Initialize variables to keep track and store information
            count = 0  #Number of tokens in block  
            passage = [] # Passage is a list of tokens in that passage

            # For each line of topdocs
            for line in f:

                # Create vectorizer with question as vocabulary
                vectorizer = CountVectorizer(vocabulary=question)
                questionVector = vectorizer.fit_transform([" ".join(question)])

                # skip if line starts with tag or Qid
                if (line[0] != "<" and line[0:3] != "Qid"):
                    # Split the line 
                    # and add words to the passage until 20 words are in a passage
                    for word in line.split():
                        if count < 20:
                            passage.append(word)
                            count += 1 
                        else:
                            # List for the passage without stopwords
                            # for better cosine similarity metrics
                            # We want to keep the passage with words to extract answer
                            passageNoStopWords = [word for word in passage if word not in stopWords]

                            # Create a feature vector for passage with lemmatized words and no stop words
                            passageVector = vectorizer.fit_transform([" ".join(self.lemmatize(" ".join(passageNoStopWords)))])

                            # Compute cosine similarity
                            cosSim =\
                                    numpy.dot(questionVector.toarray()[0], passageVector.toarray()[0]) /\
                                    (numpy.linalg.norm(questionVector.toarray()[0]) *\
                                    numpy.linalg.norm(passageVector.toarray()[0]))
                            
                            # Check that it's a number
                            if(math.isnan(cosSim)):
                                cosSim = 0
                            
                            # find smallest cosine in passages and update 
                            if(cosSim > min(passages, key=lambda x:x[1])[1]):
                                passages[passages.index(min(passages, key=lambda x:x[1]))] = \
                                    (passage, cosSim)
                            # Reset for next passage
                            count = 0
                            passage = []
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

        return lemmed

    # Method to generate ngrams
    # Uses nltk 
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

        return {k: v for k, v in sorted(grams.items(),reverse=True, key=lambda item: item[1])}

    # Method to extract answer given top 10 passages
    def extractAnswer(self, passages, n, question):
        # List to store answers
        answers = []
        # Generate ngrams
        grams = self.genNgrams(passages[0:10], 10)

        count = 0
        for gram in grams:
            answers.append(gram)
            if count == 10:
                break
            count += 1

        # For top 10 passages, print passage
        """
        for passage in passages[0:10]:
            print(passage)
            tagged = pos_tag(passage[0])

            if len(passage[0]) == 0:
                print(question)
                print("What")
            result = self.parser.parse(tagged)
            for elem in result:
                if len(elem) == 1:
                    print("")
        """

        # Need to extract noun phrases

        
        # For who questions
        # Determine the type of question
        """
        if "who" in question:
            for passage in passages[0:10]:
                tagged_sent = pos_tag(passage[0])
                for word in tagged_sent:
                    if word[1] == "NNP":
                        for seq in grams:
                            if word[0] in seq.split():
                                grams[seq] +=10
            #print(grams)
            ordered = {k: v for k, v in sorted(grams.items(), reverse=True, key=lambda item: item[1])}
            count = 0
            for seq in ordered:
                print(seq)
                count += 1
                if count == 10:
                    break


        """
        """
        elif "what" in question:
            pass
            #print("What")
        elif "where" in question:
            pass
            #print("Where")
        elif "when" in question:        
            pass
            #print("When")
        """
        return answers


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

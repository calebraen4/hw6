import string
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
# scikit
# countVectorizer

# stopwords
stopWords = list(stopwords.words('english'))

# QA system class.
class Qasystem:
    # Init method
    # Takes in the data directory
    def __init__(self, trainDir, testDir):
        self.dir = ""
        self.lemmatizer = WordNetLemmatizer()

        # Initialize data files for training
        self.trainQs = trainDir + "/qadata/questions.txt"
        self.answerPs = trainDir + "/qadata/answer_patterns.txt"
        self.relevantDs = trainDir + "/qadata/relevant_docs.txt"

        # Initialize data files for testing
        self.testQs = testDir + "/qadata/questions.txt"

    def train(self):
        print("Training")
        questions = self.processQuestions(self.trainQs)
        passages = {}

        for q in questions:
            print(questions[q])
            passages[q] = self.retrievePassages(q, questions[q])
            self.extractAnswer(passages[q], 3, questions[q])

    # Process all questions in questionFile by producing a list of keywords
    def processQuestions(self, questionFile):
        questions = {}

        with open(questionFile, 'r', encoding='utf-8-sig') as f:
            lines = (line.rstrip() for line in f)
            lines = list(line for line in lines if line)

            for i in range(len(lines)):
                if i % 2 == 0:
                    qid = lines[i][lines[i].find(':') + 2:]
                    questions[qid] = self.formulateQuery(lines[i+1])
        return questions
        
    # Given a question, formulate a query
    def formulateQuery(self, question):
        return list(set([word for word in self.lemmatize(question) if word not in stopWords]))


    # Retrieves top-10 passages based on cosine similarity
    def retrievePassages(self, qid, question): 

        # list of tuples, tuples contain passage, and cosine sim of question and that passage
        passages = [([], 0)] * 10

        # Open 
        with open("hw6_data/training/topdocs/top_docs."+ qid, 'r', encoding='ISO-8859-1') as f:
            # Initialize variables to keep track and store information
            count = 0  #Number of tokens in block  
            passage = [] # Passage is a list of tokens in that passage
            for line in f:
                # skip if line starts with tag or Qid
                if (line[0] != "<" and line[0:3] != "Qid"):
                    for word in line.split():
                        if count < 20:
                            passage.append(word)
                            count += 1 
                        else:
                            vectorizer = CountVectorizer(vocabulary=question)
                            passageVector = vectorizer.fit_transform([" ".join(self.lemmatize(" ".join(passage)))])
                            questionVector = vectorizer.fit_transform([" ".join(question)])

                            cosSim =\
                                    numpy.dot(questionVector.toarray()[0], passageVector.toarray()[0]) /\
                                    (numpy.linalg.norm(questionVector.toarray()[0]) *\
                                    numpy.linalg.norm(passageVector.toarray()[0]))
                            
                            if(math.isnan(cosSim)):
                                cosSim = 0
                            
                            # find smallest cosine in passages and update 
                            if(cosSim > min(passages, key=lambda x:x[1])[1]):
                                passages[passages.index(min(passages, key=lambda x:x[1]))] = \
                                    (passage, cosSim)
                            count = 0
                            passage = []
                            questionSim = [0] * 20
        return passages
    
    def lemmatize(self, passage):
        q = passage.translate(str.maketrans('', '', string.punctuation))

        tagged = pos_tag(q.split())
        lemmed = []
        for tag in tagged:
            if tag[1][0] == 'V':
                lemmed.append(self.lemmatizer.lemmatize(tag[0].lower(), 'v'))
            else:
                lemmed.append(self.lemmatizer.lemmatize(tag[0].lower()))

        return lemmed

    def genNgrams(self, passages, n):
        grams = {}
        for i in range(0, n + 1):
            for passage in passages:
                sequences = ngrams(passage[0], i)
                for seq in sequences:
                    sequence = ""
                    for w in seq:
                        sequence += w + " "
                    sequence = sequence[:-1]

                    if sequence in grams:
                        grams[sequence] += 1

                    else:
                        grams[sequence] = 1

        return {k: v for k, v in sorted(grams.items(),reverse=True, key=lambda item: item[1])}

    def extractAnswer(self, passages, n, question):
        grams = self.genNgrams(passages[0:10], n)

        for passage in passages[0:10]:
            print(" ".join(passage[0]) + "\n") 

        
        # For who questions
        # Determine the type of question
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

    def test(self):
        pass
# Main
if __name__ == "__main__":
    train = "./hw6_data/training"
    test = "./hw6_data/test"

    system = Qasystem(train, test)
    system.train()

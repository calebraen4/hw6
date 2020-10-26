import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
import re
import numpy  
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# scikit
# countVectorizer


#TODO
# Gather training questions,
# Process questions
# Query formulation

# stopwords
stopWords = list(stopwords.words('english'))

# QA system class.
class Qasystem:
    # Init method
    # Takes in the data directory
    def __init__(self, trainDir, testDir):
        print("Initializing")

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

        # Remove punctuation
        q = re.sub(r'[^\w\s]', '', question)

        # Return list of words without stopwords
        return [word.lower() for word in word_tokenize(q) if not word in stopWords]

    # Retrieves top-10 passages based on cosine similarity
    def retrievePassages(self, qid, question): 

        # list of tuples, tuples contain passage, and cosine sim of question and that passage
        passages = [([], 0)] * 10

        # Open 
        with open("hw6_data/training/topdocs/top_docs."+ qid, 'r', encoding='ISO-8859-1') as f:
            # Initialize variables to keep track and store information
            count = 0  #Number of tokens in block  
            passage = [] # Passage is a list of tokens in that passage
            questionSim = [0] * 20 # Question similarity
            passageSim = [1] * 20 # Passage similarity
            for line in f:
                # skip if line starts with tag or Qid
                if (line[0] != "<" and line[0:3] != "Qid"):
                    for word in line.split():
                        if count < 20:
                            passage.append(word)
                            count += 1 
                        else:
                            passageLower = [word.lower() for word in passage]
                            for word in question:
                                if word in passageLower:
                                    questionSim[passageLower.index(word)] = 1

                            cosSim =\
                                    numpy.dot(questionSim, passageSim) /\
                                    (numpy.linalg.norm(questionSim) *\
                                    numpy.linalg.norm(passageSim))
                            
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
        grams = self.genNgrams(passages, n)
        
        # For who questions
        # Determine the type of question
        if "who" in question:
            print(grams)

            print("Who question")
            words = ""
            for gram in grams:
                words += " " + gram
            tagged_sent = pos_tag(words.split())

            print(tagged_sent)

        elif "what" in question:
            pass
            #print("What")
        elif "where" in question:
            pass
            #print("Where")
        elif "when" in question:        
            pass
            #print("When")

    def test(self):
        pass
# Main
if __name__ == "__main__":
    train = "./hw6_data/training"
    test = "./hw6_data/test"

    system = Qasystem(train, test)
    system.train()

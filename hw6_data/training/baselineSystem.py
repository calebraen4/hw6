#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import numpy  
import math


# stop words to remove
stop_words = list(stopwords.words('english'))

# proccess questions, and calls passageRetr on each question to get most simiilar passages
#
def proccessing():
  with open("./qadata/questions.txt", 'r', encoding='utf-8-sig') as f:

    # removes blank lines from questions
    lines = (line.rstrip() for line in f)
    lines = list(line for line in lines if line)

    # gets question number and question
    for i in range(len(lines)):
      if i % 2 == 0:
        qid = lines[i][lines[i].find(':') + 2:]
        question = lines[i+1]
        print(qid)
        print(question)

        # returns top 10 most similar passages
        print(passageRetr(qid, question))
        


# returns top 10 most similar passages, in lsit of tuples format
#
def passageRetr(qid, question):

  #remove all punctation 
  question1 = re.sub(r'[^\w\s]', '', question)

  filteredQuestion = []

  # remove stopwords
  for w in question1.split():
    if w not in stop_words:
      filteredQuestion.append(w.lower())

  print(filteredQuestion)

  count = 0
  passage = []
  questionSim = [0] * 20
  passageSim = [1] * 20

  # list of tuples, tupels contain passage, and cosine sim of question and that passage
  tenPassages = [("blank", 0)] * 10

  with open("./topdocs/top_docs."+ qid, 'r', encoding='ISO-8859-1') as fr:
    for line in fr:
      
      # skip if line starts with tag or Qid
      if (line[0] != "<" and line[0:3] != "Qid"):
        for word in line.split():

          # add 20 tokens
          if (count < 20):
            passage.append(re.sub(r'[^\w\s]','',word.lower()) )
            count += 1
          else:
            count = 0

            # fill in question feature vector on passage vector
            for word in filteredQuestion:
              if word in passage:
                questionSim[passage.index(word)] = 1

            cosine_similarity = numpy.dot(questionSim, passageSim) / (numpy.linalg.norm(questionSim) * numpy.linalg.norm(passageSim))
            
            if(math.isnan(cosine_similarity)):
              cosine_similarity = 0
            
            # find smallest cosine in passages and update 
            if(cosine_similarity > min(tenPassages, key=lambda x:x[1])[1]):
              tenPassages[tenPassages.index(min(tenPassages, key=lambda x:x[1]))] = (passage, cosine_similarity)

            passage = []
            questionSim = [0] * 20
    return tenPassages



# ----------------------------------------------------------------------------------------------------------------------- #
# main

proccessing()

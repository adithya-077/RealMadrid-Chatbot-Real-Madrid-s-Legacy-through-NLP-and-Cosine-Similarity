import nltk
import random
import string
import warnings
from fastapi import FastAPI


txtfile = open("./rm.txt",'r',errors='ignore').read()
file = txtfile.lower()
sentance_token = file.split('\n\n')
word_token = nltk.word_tokenize(file)

lem = nltk.stem.WordNetLemmatizer()



warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
GREETINGS_INPUT = ("HELLO","HOLA","VANAKAM","NAMASTE","HEY THERE!!","HEYY","HEY")
GREETINGS_RES = ("HELLO","HOLA","VANAKAM","NAMASTE","HEY THERE!!","HEYY","HEY")
rempun = dict((ord(i),None) for i in string.punctuation)
nltk.download('punkt')
nltk.download('wordnet')


def lemnorm(text):
    res = nltk.word_tokenize(text.lower().translate(rempun))
    return [lem.lemmatize(i) for i in res]
def coreFun(userResp):
    cres = ''
    TfVec = TfidfVectorizer(tokenizer=lemnorm,stop_words="english",max_df=0.85, min_df=0.008)
    prodata = TfVec.fit_transform(sentance_token)
    userResp = nltk.sent_tokenize(userResp)
    userResp = TfVec.transform(userResp)
    cosinesim = cosine_similarity(userResp,prodata)
    maxsim = cosinesim.argsort()
    ind = maxsim[0]
    res = maxsim.flatten()
    res.sort()
    resf = res[-1]
    if(resf!=0):
        for i in ind[::-1][:3]:
            cres=cres+sentance_token[i] +"\n\n"

        return cres
    else:
        cres=cres+"I cant find the answer or no related information available."
        return cres

def GREETINGS(txt):
    if txt.split()[0] in [i.lower() for i in GREETINGS_INPUT]:
        return random.choice(GREETINGS_RES)
    else :
        return coreFun(txt)


app = FastAPI()
@app.get("/UserQuery/{query}")
async def root(query):
    return GREETINGS(query)

    
    

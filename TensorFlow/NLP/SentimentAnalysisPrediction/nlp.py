import os
import nltk
import nltk.corpus
from nltk.corpus import brown #to divide 
#print(os.listdir(nltk.data.find("corpora")))
"""nltk.download('corpora/gutenberg')
print(nltk.corpus.gutenberg.fileids())"""

IA = """John McCarthy, known as the Father of Artificial Intelligence,
was a pioneering computer scientist who coined the term Artificial Intelligence (AI) 
in 1956. He played a crucial role in the development of AI as a field, contributing to 
the creation of Lisp, one of the earliest programming languages for AI research. 
McCarthy's work laid the foundation for concepts like machine learning, reasoning,
and problem-solving in computers. He also proposed the idea of time-sharing in computing, 
which significantly influenced modern operating systems. His contributions continue to shape 
the evolution of AI, making him one of the most influential figures in computer science."""

print(type(IA))

#To tokinize all words
from nltk.tokenize import word_tokenize
AI_tokens = word_tokenize(IA)
print(AI_tokens)


from nltk.probability import FreqDists
fdist = FreqDists()

for word in AI_tokens:
    fdist[word.lower()]+=1
#This count the probability of looking all wors and how it much repeated(how many appears)
print(fdist)

#This counts how many paragraphs they are 
from nltk.tokenize import blankline_tokenize
AI_blank = blankline_tokenize(IA)


from nltk.util import bigrams, trigrams, ngrams
#It gives you a set of two words, and you can do trigrams and ngrams(all words you want)
quote_biagrams = list(nltk.bigrams(AI_tokens)) #Pass the tokens you create
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:39:22 2019

@author: fengzhenXu
"""

import nltk
import sys
from nltk.corpus import brown

brown_tags_words=[]
for sent in brown.tagged_sents():
    brown_tags_words.append(("START","START"))
    brown_tags_words.extend([(tag[:2],word) for (word,tag) in sent])
    brown_tags_words.append(("END","END"))
    
#print(brown_tags_words[:3])
    
# conditional frequency distribution  
cfd_tagwords=nltk.ConditionalFreqDist(brown_tags_words)
#print(cfd_tagwords)
# conditional probability distribution
cpd_tagwords=nltk.ConditionalProbDist(cfd_tagwords,nltk.MLEProbDist)
#print(cpd_tagwords)

print("The probability of an adjective (JJ) being 'new' is ",cpd_tagwords["JJ"].prob("new"))
print("The probability of a verb(VB) being 'duck' is",cpd_tagwords["VB"].prob("duck"))

brown_tags=[tag for (tag,word) in brown_tags_words]

cfd_tags=nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
cpd_tags=nltk.ConditionalProbDist(cfd_tags,nltk.MLEProbDist)

print("If we have just seen 'DT',the probablity of 'NN' is",cpd_tags["DT"].prob("NN"))
print( "If we have just seen 'VB', the probability of 'JJ' is", cpd_tags["VB"].prob("DT"))
print( "If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))

prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("want") * \
    cpd_tags["VB"].prob("TO") * cpd_tagwords["TO"].prob("to") * \
    cpd_tags["TO"].prob("VB") * cpd_tagwords["VB"].prob("race") * \
    cpd_tags["VB"].prob("END")

print( "The probability of the tag sequence 'START PP VB TO VB END' for 'I want to race' is:", prob_tagsequence)

#Viterbi
distinct_tags=set(brown_tags)
sentence=["I","want","to","race"]
sentlen=len(sentence)

viterbi=[]
backpointer=[]

first_viterbi={}
first_backpointer={}
for tag in distinct_tags:
    if tag=="START":continue
    first_viterbi[tag]=cpd_tags["START"].prob(tag)*cpd_tagwords[tag].prob(sentence[0])
    first_backpointer[tag]="START"
print(first_viterbi)
print(first_backpointer)
viterbi.append(first_viterbi)
backpointer.append(first_backpointer)

currentbest=max(first_viterbi.keys(),key=lambda tag:first_viterbi[tag])
print("Word","'"+sentence[0]+"'","current best two-tag sequence:",first_backpointer[currentbest],currentbest)

for wordindex in range(1,len(sentence)):
    this_viterbi={}
    this_backpointer={}
    prev_viterbi=viterbi[-1]
    for tag in distinct_tags:
        if tag=="START":continue
        best_previous=max(prev_viterbi.keys(),key=lambda prevtag:prev_viterbi[prevtag]*cpd_tags[prevtag].prob(tag)*cpd_tagwords[tag].prob(sentence[wordindex]))
        this_viterbi[tag]=prev_viterbi[best_previous]*cpd_tags[best_previous].prob(tag)*cpd_tagwords[tag].prob(sentence[wordindex])
        this_backpointer[tag]=best_previous
        
    #currentbest=max(this_viterbi.keys(),key=lambda tag：this_viterbi[tag])
    currbest = max(this_viterbi.keys(), key = lambda tag: this_viterbi[ tag ])
    print("Word", "'" + sentence[ wordindex] + "'", "current best two-tag sequence:", this_backpointer[currbest], currbest)
    viterbi.append(this_viterbi)
    backpointer.append(this_backpointer)
    
prev_viterbi=viterbi[-1]
best_previous=max(prev_viterbi.keys(),key=lambda prevtag:prev_viterbi[prevtag]*cpd_tags[prevtag].prob("END"))
prob_tagsequence=prev_viterbi[best_previous]*cpd_tags[best_previous].prob("END")
best_tagsequence=["END",best_previous]
backpointer.reverse()

current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]

best_tagsequence.reverse()
print( "The sentence was:", end = " ")
for w in sentence: print( w, end = " ")
print("\n")
print( "The best tag sequence is:", end = " ")
for t in best_tagsequence: print (t, end = " ")
print("\n")
print( "The probability of the best tag sequence is:", prob_tagsequence)













    
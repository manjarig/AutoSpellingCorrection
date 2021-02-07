import re
import random
import string
import numpy as np
random.seed(17)
import math
def main():
    predicted_wordlist=[]
    cleanfile=readfile()
    print("clean data length")
    clean_train_data,clean_test_data=split_data(cleanfile)
    noisy_train_data,noisy_test_data,DICT_ERROR=build_noisy_file(cleanfile)
    print(len(noisy_test_data))
    transition,emission,pi=build_state_and_emissionsymbols(clean_train_data,noisy_train_data,DICT_ERROR)
    predicted_wordlist=givenoisyword_to_Viterbi(noisy_test_data,clean_test_data,transition,emission,pi)
    precision(predicted_wordlist,clean_test_data,noisy_test_data)
#####################TEXT PREPROCESSING##############################################################
def readfile():
    list1=[]
    count_words=0
    with open('Pride&Prejudice.txt',encoding="utf8") as fileinput:
        fileinput=fileinput.read().split()
        for line in fileinput: 
            line=line.lower()
            PUNCT=re.compile('[.!?,#@^&\-: ”;“ —\'1234567890()]')
            string_ = re.sub(PUNCT, " ", line)
            string_ = string_.replace(' ','')
            count_words+=1
            if string_:
                list1.append(string_)   
        #print("Total Words ",count_words)
        return list1  

def split_data(cleanfile):
    clean_train_data=[]
    clean_test_data=[]
    index = math.ceil(0.8 * len(cleanfile))
    for i in range(0,index):
        clean_train_data.append(cleanfile[i])
    for i in range(index,len(cleanfile)):
        clean_test_data.append(cleanfile[i])
    return clean_train_data,clean_test_data

def build_noisy_file(cleanfile):
    DICT_ERROR={  
                    'q':['w','a','s'], 'w':['q','a','e'], 'e':['w','r','s'], 'r':['e','t','f'],
                    't':['r','y','g'], 'y':['t','h','u'], 'u':['y','i','j'], 'i':['u','o','k'],
                    'o':['i','p','l'], 'p':['o','i','l'], 'a':['q','s','z'], 's':['a','x','d'],
                    'd':['e','s','f'], 'f':['e','d','g'], 'g':['v','f','h'], 'h':['g','n','j'],
                    'j':['h','n','k'], 'k':['l','m','j'], 'l':['m','p','k'], 'z':['a','x','s'],
                    'x':['s','z','c'], 'c':['d','f','v'], 'v':['f','c','b'], 'b':['v','g','n'],
                    'n':['j','b','h'], 'm':['k','n','j'] 
                    }
    noisy_file=[]
    for word in cleanfile:
        temp_word=""
        for i in range(0,len(word)):        
            r=random.uniform(0,1)
            if(r<=0.1):
                temp_word+=random.choice(DICT_ERROR[word[i]])
            else:
                temp_word+=word[i]
        noisy_file.append(temp_word.strip())
    noisy_train_data=[]
    noisy_test_data=[]
    index = math.ceil(0.8 * len(cleanfile))
    for i in range(0,index):
        noisy_train_data.append(noisy_file[i])
    for i in range(index,len(cleanfile)):
        noisy_test_data.append(noisy_file[i])      
    return noisy_train_data,noisy_test_data,DICT_ERROR
    
def  build_state_and_emissionsymbols(cleanfile,noisy_train_data,DICT_ERROR):    
    pi = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0, 'e': 0.0, 'f': 0.0, 'g': 0.0, 'h': 0.0, 'i': 0.0, 'j': 0.0, 'k': 0.0,
      'l': 0.0, 'm': 0.0,
      'n': 0.0, 'o': 0.0, 'p': 0.0, 'q': 0.0, 'r': 0.0, 's': 0.0, 't': 0.0, 'u': 0.0, 'v': 0.0, 'w': 0.0, 'x': 0.0,
      'y': 0.0, 'z': 0.0}

    for i in range(0,len(cleanfile)):
        pi[cleanfile[i][0]]+=(1/float(len(cleanfile)))
    print("PRIOR PROBABILITY")
    print(pi)
    print("___________________________________________________________________________________________")
    transition = np.zeros([26,26])
    emission=np.zeros([26,26])
    ####### use 97 for ASCII conversion###################
    for word in cleanfile:
        for i in range(len(word)):   
            if i != len(word) - 1:
                transition[ord(word[i])-97][ord(word[i + 1])-97] += 1
    print("TRANSITION MATRIX")
    print(transition)
    total = np.sum(transition,axis=1)
    transition = (transition+1)/(total+26.0)
    ######calculate emission probability#####
    for word in range(len(cleanfile)):
        for i in range(len(cleanfile[word])):
            emission[ord(cleanfile[word][i])-97][ord(noisy_train_data[word][i])-97] += 1
    total1 = np.sum(emission,axis=1)
    emission = (emission+1)/(total1+26.0)
    print("____________________________________________________________________________________________________")
    print("EMISSION MATRIX")
    print(emission)
    return transition,emission,pi
    
def viterbi(word,clean_test_data,transition,emission,pi): 
    pi_list = []
    # print("real word:" + str(word))
    for key,values in pi.items():
        pi_list.append(values)
    pi = np.array(pi_list)
    pi = pi.reshape(1,26)
    delta1 = pi*emission[:,ord(word[0])-97]
    delta1 = delta1.reshape(26,1)   
    test = []
    psi = delta1
    test.append(psi)
    for alpha in range(1,len(word)):
        delta = test[-1] * transition
        ele_max = np.amax(delta,axis=0).reshape(26,1)
        psi = ele_max*(emission[:,ord(word[alpha])-97].reshape(26,1))
        test.append(psi)
    max_value_position = np.argmax(test[-1],axis=0)
    pred_word=''
    for i in range(len(test)-1,-1,-1):        
        delta_ = test[i]*transition[:,max_value_position]
        max_value_position = np.argmax(delta_,axis=0)
        pred_word=(chr(97+int(max_value_position)))+pred_word   
    # print("Predicted word is " + str(pred_word))
    return pred_word

def givenoisyword_to_Viterbi(noisy_test_data,clean_test_data,transition,emission,pi):
    predicted_wordlist=[]
    for noisy_test_word in noisy_test_data:
        pred_word=viterbi(noisy_test_word,clean_test_data,transition,emission,pi)
        predicted_wordlist.append(pred_word)
    return predicted_wordlist

def precision(predicted_wordlist,clean_test_data,noisy_test_data):
    truepositive=0
    falsepositive=0
    falsenegative=0
    totalcorrupt=0
    truenegative=0
    
    for i in range(len(clean_test_data)):
        for j in range(len(clean_test_data[i])):

            if clean_test_data[i][j]!=noisy_test_data[i][j] and clean_test_data[i][j]==predicted_wordlist[i][j]:
                truepositive+=1 
            if clean_test_data[i][j]==noisy_test_data[i][j] and clean_test_data[i][j]!=predicted_wordlist[i][j]:
                falsepositive+=1
            if clean_test_data[i][j]!=predicted_wordlist[i][j] and clean_test_data[i][j]!=noisy_test_data[i][j]:
                falsenegative+=1       
            if clean_test_data[i][j]==noisy_test_data[i][j]and clean_test_data[i][j]==predicted_wordlist[i][j]:
                truenegative+=1

    print("**************EVALUATION METRICS******************************************************************************")
    print("True positive   :"  +str(truepositive))
    print("False positive  :"  +str(falsepositive))
    print("False Negative  :"  +str(falsenegative))  
    print("True Negative   :"  +str(truenegative))
    print("Precision is    :"  +str( truepositive/(truepositive+falsepositive ) *100))
    print("Recall is       :"  +str(truepositive/(truepositive+falsenegative) * 100 ))

if __name__ == "__main__":
    main()
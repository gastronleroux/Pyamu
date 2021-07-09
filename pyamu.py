import math
from random import random
from difflib import SequenceMatcher
import numpy as np
#import spacy
#from spellchecker import SpellChecker



#  Natural Language Processing
# -----------------------------------------------------------------------------------------

# French language lemmatizer
# To lemmatize your sentence:
# lemmaObjectList = frLemmatize(yourSentence)
#
# To iterate trough your lemmatized sentence:
# for x in lemmaObjectList:
#   ...
#
# LemmaObject attributes:
# 
# x.word            --- the word originally appearing in the text.
#
# x.lemma           --- the base form of the original word. If the original
#                       word is not found, it is made by analogy. If there is 
#                       a problem finding the analogy, the original word is
#                       repeated here.
#
# x.analogy         --- a word found by analogy with the original word. None 
#                       if the original word is found or no analogy is found.
#
# x.analogyLemma    --- the base form of the analogous word. None if the 
#                       original word is found or no analogy is found.
#
# x.corrected       --- word corrected. If the original word is found or there 
#                       is a problem finding a similar word, the original word 
#                       is repeated here.
#
# x.correctedLemma  --- base form of the word after proofreading. If the
#                       original word is found, the underlying form repeats
#                       itself. If there is a problem finding a similar word,
#                       the original word is repeated here.
#
# x.similarity      --- the similarity between the original word and the corrected
#                       word expressed in a floating point number from 0.0 to 1.0.
#
# x.probability     --- the probability of the correctness of the correction
#                       expressed as a floating point number from 0.0 to 1.0.
#

class LemmaObject:
    def __init__(self, word, lemma, analogy, analogyLemma,
                 corrected, correctedLemma, similarity, probability):
        self.word = word
        self.lemma = lemma
        self.analogy = analogy
        self.analogyLemma = analogyLemma
        self.corrected = corrected
        self.correctedLemma = correctedLemma
        self.similarity = similarity
        self.probability = probability
        
def frLemmatize(sen):
    lemmaObjectList = []
    # python -m spacy download fr_core_news_lg
    sp = spacy.load('fr_core_news_lg')
    vocab = list(sp.vocab.strings)[214:]
    spell = SpellChecker(language='fr') 
    sentence = sp(sen)
    suffixs = {}
    suffixWords = {}
    
    for word in sentence:
        if word.is_oov:
            suffixs[word.text] = word.text
    while len([1 for suffix in suffixs.values() if suffix == '']) < len(suffixs):
        for word in vocab:
            for originalWord in suffixs.keys():
                if suffixs[originalWord] != '' and len(word) >= len(suffixs[originalWord]) \
                        and word[-len(suffixs[originalWord]):] == suffixs[originalWord] \
                        and not sp(word)[0].is_oov:
                    suffixWords[originalWord] = {'analogy': word, 'suffix': suffixs[originalWord]}
                    suffixs[originalWord] = ''
        for originalWord in suffixs.keys():
            if suffixs[originalWord] != '':
                suffixs[originalWord] = suffixs[originalWord][1:]
            
    
    for word in sentence:
        lemma = word.lemma_
        analogy = None
        analogyLemma = None
        corrected = word.text
        correctedLemma = word.lemma_
        similarity = 1.0
        probability = 1.0
        
        if word.is_oov:
            if word.text in suffixWords:
                analogy = suffixWords[word.text]['analogy']
                analogyLemma = sp(analogy)[0].lemma_
                stem = analogy[:-len(suffixWords[word.text]['suffix'])]
                suffix = analogyLemma[len(stem):]
                lemma = word.text[:-len(suffixWords[word.text]['suffix'])] + suffix
            
            corrected = spell.correction(word.text)
            correctedLemma = sp(corrected)[0].lemma_
            similarity = SequenceMatcher(None, word.text, corrected).ratio()
            if word.text == corrected:
                probability = 0.0
            elif len(spell.candidates(word.text)) > 1:
                candidates = spell.candidates(word.text)
                wordsProbabilityMean = sum([spell.word_probability(prob) for prob in candidates]) / len(candidates)
                correctedProbability = spell.word_probability(corrected)
                probability = (1 - (wordsProbabilityMean / correctedProbability))
            if probability != 0.0:
                probability = probability  * 0.9 + similarity * 0.1
                    
        lemmaObjectList.append(LemmaObject(word.text, lemma, analogy, analogyLemma, corrected, correctedLemma, similarity, probability))
        
    return lemmaObjectList



#  Neural Networks
# -----------------------------------------------------------------------------------------

# Displays McCulloch-Pitts neuron model's results for 
# NOT, AND, NAND and OR Gates using appropriate weight factor
# and theta values.
# e.g. displayGates()
# Gate Not
# u1 = 0, result = 1
# u1 = 1, result = 0

# Gate And
# u1 = 0, u2 = 0, result = 0
# ...
def displayGates():
    def f(x):
        if x < 0: return 0
        elif x >= 0: return 1

    def scalar_prod(un, wn, theta):
        un.append(-theta)
        wn.append(1)
        return f(sum([un[i]*wn[i] for i in range(len(un))]))

    def display(uns, wn, theta, title):
        print("Gate", title,)
        for i in range(len(uns)):
            for j in range(len(uns[i])):
                print("u", j+1, " = ", uns[i][j], sep='', end=', ')
            print("result =", scalar_prod(uns[i], wn, theta))
        print()

    display([[0], [1]], [-2], -1, "Not")
    display([[0, 0], [1, 0], [0, 1], [1, 1]], [2, 2], 3, "And")
    display([[0, 0], [1, 0], [0, 1], [1, 1]], [-2, -2], -3, "Nand")
    display([[0, 0], [1, 0], [0, 1], [1, 1]], [2, 2], 1, "Or")


# Implementation of the Gradient Method Algorithm.
# Finds the local and global minimums of the following functions and
# the points that reach its minimum:
# (1) F1(x1, x2, x3) = 2x1^2 + 2x2^2 + x3^2 − 2x1x2 − 2x2x3 − 2x1 + 3
# (2) F2(x1, x2) = 3x1^4 + 4x1^3 − 12x1^2 + 12x2^2 − 24x2
# In the end displays the minimums and the result.
def displayGradientMethod():
    def F1(vec):
        return (2 * vec[0]**2 + 2 * vec[1]**2 + vec[2]**2 - 2 * vec[0] * vec[1] - 2 * vec[1] * vec[2] - 2 * vec[0] + 3)
    def F2(vec):
        return (3 * vec[0]**4 + 4 * vec[0]**3 - 12 * vec[0]**2 + 12 * vec[1]**2 - 24 * vec[1])
    Fs = [F1, F2]

    #Partial derivative functions for F1(x1, x2, x3) and F2(x1,x2)
    def F1x1(vec): return (4 * vec[0] - 2 * vec[1] - 2)
    def F1x2(vec): return (4 * vec[1] - 2 * vec[0] - 2 * vec[2])
    def F1x3(vec): return (2 * vec[2] - 2 * vec[1])
    def F2x1(vec): return (12 * vec[0]**3 + 12 * vec[0]**2 - 24 * vec[0])
    def F2x2(vec): return (24 * vec[1] - 24)
    DFs = [[F1x1, F1x2, F1x3], [F2x1, F2x2]]

    # f - 0 means F1, 1 means F2
    def gradient(f, N, c, eps):
        vecOld = [random()*N*2-N for i in range(len(DFs[f]))]
        vecNew = [vecOld[i] - c * DFs[f][i](vecOld) for i in range(len(DFs[f]))]
        while max([vecNew[i] - vecOld[i] for i in range(len(DFs[f]))]) > eps:
            vecOld = vecNew
            vecNew = [vecOld[i] - c * DFs[f][i](vecOld) for i in range(len(DFs[f]))]
        print("Results for F", (f+1), ", N = ", N, ", c = ", c, ", eps = ", eps, ":", sep='')
        print("Local min:", vecNew)
        print("Global min:", Fs[f](vecNew), "\n")
    gradient(0, 1, 0.01, 0.00001)
    gradient(1, 1, 0.01, 0.00001)


# Implementation of the gradient method algorithm 
# for backpropagation that performs XOR training.
def displayBackPropagation():
    u = [[0,0,1], [0,1,1], [1,0,1], [1,1,1]]
    z = [0,1,1,0]
    c = 0.5
    eps = 0.00001
    beta = 2.5
    N = 1

    def Df(x):
        return((beta * math.exp(-beta*x))/(1+math.exp(-beta*x))**2)
    def f(x):
        return(1/(1+math.exp(-beta*x)))
        
    def update(s_old, w_old):
        x = [[f(sum([w_old[i][j]*u[p][j] for j in range(3)])) for i in range(2)]+[1] for p in range(4)]
        y = [f(sum([s_old[i] * x[p][i] for i in range(3)])) for p in range(4)]
        DE_s = [sum([(y[p] - z[p]) * Df(sum([s_old[k]*x[p][k] for k in range(3)]))*x[p][i] for p in range(4)]) for i in range(3)]
        DE_w = [[sum([(y[p]-z[p])*Df(sum([s_old[k]*x[p][k] for k in range(3)]))*s_old[i]*Df(sum([w_old[i][l]*u[p][l] for l in range(3)]))*u[p][j] for p in range(4)]) for j in range(3)] for i in range(2)]
        s_new = [s_old[i] - c * DE_s[i] for i in range(3)]
        w_new = [[w_old[i][j] - c * DE_w[i][j] for j in range(3)] for i in range(2)]
        return s_new, w_new, y
        
    def propagation():
        s_old = [0, 1, 2]
        w_old = [[0, 1, 2], [0, 1, 2]]
        s_new, w_new, y = update(s_old, w_old)
        
        while max( max([abs(s_new[i] - s_old[i]) for i in range(3)]), max([max([abs(w_new[i][j] - w_old[i][j]) for j in range(3)]) for i in range(2)]) ) > eps:
            s_old = s_new
            w_old = w_new
            s_new, w_new, y = update(s_old, w_old)
        print("y =", y, "\ns_old =", s_old, "\nw_old =", w_old, '\n')
        
    propagation()


# Implementation of Boltzmann Machine (BM) on the example of image "1" for z.
# Gets different number of reversed "1" image occurence depending on temperature chosen.
# max_t = number of steps to take
# Displays newly computed images troughout max_t iterations.
# In the end displays a comparision between "1" image and it's reversed version in occurences.
def displayBoltzmannMachine(max_t = 15):
    n = 25
    in_one_row = 5
    z = np.array(
        [0, 0, 0, 0, 0,
        0, 1, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0])
    nz = np.array([(i+1)%2 for i in z])
    c = np.array([[0 if i==j else (z[i]-1/2) * (z[j] - 1/2) for j in range(n)] for i in range(n)])
    w = 2*c
    theta = np.array([sum(c[i]) for i in range(n)])

    def display(x):
        count = 0
        while count < len(x):
            print(x[count: count + in_one_row])
            count += in_one_row
        print()

    def f(x, T):
        return 1/(1 + math.exp(-x/T))

    def next(x, T):
        B = [random() for i in range(n)]
        u = [sum(w[i]*x) - theta[i] for i in range(n)]
        new_x = np.array([1 if 0 <= B[i] and B[i] <= f(u[i], T)
            else 0 for i in range(n)])
        return new_x

    def iterate(x, t, T):
        count_z = 0
        count_nz = 0
        while(t < max_t):
            if (x == z).all(): count_z += 1
            elif (x == nz).all(): count_nz += 1
            print("t =", t)
            display(x)
            t += 1
            x = next(x, T)
        print("\n\nNumber of occurences: ", count_z, " (", count_z/max_t, "%)", sep='')
        display(z)
        print("Number of occurences: ", count_nz, " (", count_nz/max_t, "%)", sep='')
        display(nz)

    T = 1
    t = 0
    x = np.array([1 if random() > 1/2 else 0 for i in range(n)])

    iterate(x, t, T)


#  Combinatorial Algorithms prep functions
# -----------------------------------------------------------------------------------------


# Check if x is None. If so, returns y, if not, returns x.
def ifVarNone(x, y):
    if x is None: return y
    else: return x

# Binary list to subset array
def BtoL(T):
    return [i+1 for i in range(len(T)) if T[i]]
    
# Subset array to binary list
def LtoB(n, T):
    return [int(i in T) for i in range(1,n+1)]

# Binary list to decimal number
def BtoD(T):
    n = 0
    for i in T: n = 2 * n + i
    return n

# Decimal number to binary list
def DtoB(r, n):
    x = [int(i) for i in list("{0:b}".format(int(r)))]
    while(len(x) < n): x[0:0] = [0]
    return x

# Product of binomial coefficient for n and k
def binomial(n, k):
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n-k)):
        b *= n
        b /= t+1
        n -= 1
    return b

#Graph class
class G:
    def __init__(self, V, E, w):
        self.V = V
        self.E = E
        self.w = w

    def edgeWeight(self, i, j):
        if(i == j):
            return 0
        elif (i,j) in self.w.keys():
            return self.w[(i,j)]
        elif (j,i) in self.w.keys():
            return self.w[(j,i)]
        else:
            return float('inf')


#  Combinatorial Algorithms
# -----------------------------------------------------------------------------------------

# Returns an array of arrays containing numbers in a lexicographical.
# If k is not specified, it's equal to n. Default initial value is 1.
# n = number of elements, k = maximum value, initial = minimum value
# e.g. lexicalOrder(1, 3) == [
# [1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2],
# [2, 1, 1], [2, 1, 2], [2, 2, 1], [2, 2, 2]
# ]
def lexicalOrder(n, k = None, initial = 1):
    k = ifVarNone(k, n)

    if initial > k: 
        raise ValueError("Initial value can't be bigger than the maximum value")
    if n < 1: 
        raise ValueError("Number of values must be a positive number")

    output = []
    x = [initial for i in range(n)]
    while(True):
        output.append(x.copy())
        for i in range(n-1, -1, -1):
            
            if x[i] < k:
                x[i] += 1
                if i < n-1:
                    x[i+1:] = [initial for j in range(len(x[i+1:]))]
                break
        else:
            break
    return output


# Returns a bijection dictionary between binary strings of length n and {1, 2, ..., n} subsets.
# The key is a binary string (tuples), the value is a {1, 2, ..., n} subset (array),
# e.g. bijectionSubsets(2)[[1, 0]] == [1]
def bijectionSubsets(n):
    if n < 0: raise ValueError("Number of values mustn't be a negative number")

    output = {}
    x = [0 for i in range(n)]
    while(True):
        output[tuple(x)] = [i+1 for i in range(n) if x[i] == 1]
        for i in range(n-1, -1, -1):
            if x[i] < 1:
                x[i] += 1
                if i < n-1:
                    x[i+1:] = [0 for j in range(len(x[i+1:]))]
                break
        else:
            break
    return output


# Returns position (int) of the subset T ⊂ {1, ..., m} (where m <= n) in the lexicographic ordering
# (according to characteristic vectors) of the subsets of the set {1, ..., n}.
# e.g. lexicalPosition(5, [2, 3, 5]) == 13
def lexicalPosition(n, T):
    return BtoD((int(i in T)) for i in range(1,n+1))


# Returns subset (array) of T with a given position r in the lexicographic ordering
# (according to characteristic vectors) of the subsets of the set {1, ..., n}.
# e.g. lexicalSubsetFromPosition(5, 13) == [2, 3, 5]
def lexicalSubsetFromPosition(n, r):
    w_ch = [int(i) for i in list("{0:b}".format(int(r)))]
    while(len(w_ch) < n): w_ch[0:0] = [0]
    T = [i for i in range(1, n+1) if w_ch[i-1]]
    return T


# Returns an array of arrays containing numbers in a lexicographical.
# If k is not specified, it's equal to n.
# n = number of elements, k = maximum value
# Works recursively.
# e.g. lexicalOrderRecursive(2, 3) == [
# [1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]
# ]
def lexicalOrderRecursive(n, k = None,):
    def recur(n, k, output, x = [], i = 0):
        if(x == []): x = [0]*(n+1)
        if i == n:
            output.append(x[1:])
        else:
            for j in range(1, k+1):
                x[i+1] = j
                recur(n, k, output, x, i+1)

    output = []
    k = ifVarNone(k, n)
    recur(n, k, output)
    return output


# Returns an array of all subsets (arrays) of the set {1, ..., n}
# in minimum (Gray) order using Hamming weights.
# e.g. grayOrder(2) == [[], [2], [1, 2], [1]]
def grayOrder(n):
    output = []
    T = [0]*n
    output.append(BtoL(T))
    while not (T[0] == 1 and len(T)>1 and T[1:] == [0]*(n-1)):
        if sum(T)%2 == 0: 
            T[-1] = (T[-1]+1)%2
        else: 
            for i in range(n-1, -1, -1):
                if T[i] == 1 and i != 0:
                    T[i-1] = (T[i-1]+1)%2
                    break
        output.append(BtoL(T))
    return output


# Returns the rank (int) of the subset T ⊂ {1, ..., n} in the order
# of minimal changes (Gray) of the subsets of the set {1, ..., n}
# e.g. grayRank(4, [1, 3]) == 12
def grayRank(n, T):
    pBitRangi = 0
    R = []
    binT = LtoB(n, T)
    for i in range(n):
        R.append(pBitRangi ^ binT[i])
        pBitRangi = R[-1]
    return BtoD(R)


# Returns a subset of T (array) with a given position r in the order 
# of minimal changes (Gray) of the subsets of the set {1, ..., n}.
# e.g graySubsetFromRank(4, 12) == [1, 3]
def graySubsetFromRank(n, r):
    return BtoL([DtoB(r, n)[i] ^ DtoB(r//2, n)[i] for i in range(n)])


# Returns successor (array) of the k-element subset T of the set {1, ..., n}
# in the lexicographic ordering of k-element subsets.
# Returns -1 if there is no successor.
# e.g. kSubsetSuccessor(8, 4, [1, 2, 7, 8]) == [1, 3, 4, 5]
def kSubsetSuccessor(n, k, T):
    maxT = [i for i in range(n-k+1, n+1)]
    if T == maxT: return -1
    for i in range(k-1, -1, -1):
        if T[i] != maxT[i]:
            T[i] += 1
            T[i:] = [j+T[i] for j in range(len(T[i:]))]
            break
    return T


# Returns the rank (int) of the k-element subset T of the set {1, ..., n}
# in the lexicographic ordering of k-element subsets.
# e.g. pyamu.kSubsetRank(5, 3, [2, 3, 5]) == 7
def kSubsetRank(n, k, T):
    T = [0, *T]
    return int(sum([sum([binomial(n - j, k - i) for j in range(T[i-1]+1, T[i])]) for i in range(1, k+1)]))


# Returns a subset of T (array) with rank r in the lexicographic
# ordering of k-element subsets of the set {1, ..., n}.
# e.g. kSubsetFromRank(5, 3, 7) == [2, 3, 5]
def kSubsetFromRank(n, k, r):
    T = [0]*k
    x = 1
    for i in range(1, k+1):
        while binomial(n-x, k-i) <= r:
            r = r - binomial(n-x, k-i)
            x += 1
        T[i-1] = x
        x += 1
    return T


# Returns all divisions (3d array) of the set {1, ..., n} by moving between active item blocks.
# e.g. subsetDivision(3) == [
# [ [1, 2, 3] ], 
# [ [1, 2], [3] ], 
# [ [1], [2], [3] ],
# [ [1], [2, 3] ],
# [ [1, 3], [2] ]
# ]
def subsetDivision(n):

    def divide(B):
        divisions = []
        T = [[] for i in range(max(B))]
        for i in range(len(B)):
            T[B[i]-1].append(i+1)
        for t in T:
            if len(t)>0:
                divisions.append(t)
        return divisions

    N = [0]*(n+1)
    P = [0]*(n+1)
    B = [0]+[1]*(n)
    PR = [0]+[1]*(n)
    j = n
    output = []

    output.append(divide(B[1:]))
    while not(PR[j] and B[j] == j or not PR[j] and B[j] == 1):
        k = B[j]
        if PR[j]:
            if N[k] == 0:
                N[k] = j
                N[j] = 0
                P[j] = k
            if N[k] > j:
                P[j] = k
                N[j] = N[k]
                P[N[j]] = j
                N[k] = j
            B[j] = N[k]
        else:
            B[j] = P[k]
            if j == k:
                if N[k] == 0:
                    N[P[k]] = 0
                else:
                    N[P[k]] = N[k]
                    P[N[k]] = P[k] 
        j = n
        while j > 1 and ((PR[j] and B[j] == j) or (not PR[j] and B[j] == 1)):
            PR[j] = (PR[j]+1)%2
            j -= 1
        output.append(divide(B[1:]))
    return output


# Returns an RGF function (array) f: {1, ..., n} → Z+ corresponding to the
# given division of the set {1, ..., n}.
# e.g. rgfFunction(10, [[3, 6, 7], [1, 2], [5, 8, 9], [4, 10]]) == 
#   [1, 1, 2, 3, 4, 2, 2, 4, 4, 3]
def rgfFunction(n, B):
    B.sort()
    B = [[]]+B
    k = len(B)
    f = [0]*(n+1)
    j = 1
    for i in range(1, k):
        while f[j] != 0:
            j += 1
        h = 1
        while j not in B[h]:
            h += 1
        for g in B[h]:
            f[g] = h
    return(f[1:])
    

# Returns a division of the set {1, ..., n} corresponding to the
# given RGF function f: {1, ..., n} → Z+.
# e.g. rgfDivision([1, 1, 2, 3, 4, 2, 2, 4, 4, 3]) == 
#   [[1, 2], [3, 6, 7], [4, 10], [5, 8, 9]]
def rgfDivision(f):
    f = [0]+f
    n = len(f[1:])
    k = 1
    for j in range(1, n+1):
        if f[j] > k:
            k = f[j]
    B = [[] for j in range(k+1)]
    for j in range(1, n+1):
        B[f[j]].append(j)
    return(B[1:])
    

# Returns all RGF functions (array of arrays) f: {1, ..., n} → Z+ in lexicographic order.
# e.g. rgfGenerate(3) == [[1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [1, 2, 3]]
def rgfGenerate(n):
    output = []
    f = [1]*(n+1)
    F = [2]*(n+1)
    end = False
    while not end:
        output.append(f[1:])
        j = n
        while f[j] == F[j]:
            j -= 1
        if j > 1:
            f[j] += 1
            for i in range(j+1, n+1):
                f[i] = 1
                if f[j] == F[j]:
                    F[i] = F[j] + 1
                else:
                    F[i] = F[j]
        else:
            end = True
    return output


# Returns successor (array) of the p permutation of the set {1, ..., n} in lexicographic order.
# e.g. permutationSuccessor([3, 6, 2, 7, 5, 4, 1]) == [3, 6, 4, 1, 2, 5, 7]
def permutationSuccessor(p):
    brak = False
    for i in range(len(p)-2, -1, -1):
        if p[i] < p[i+1]:
            break
    else: brak = True
    for j in range(len(p)-1, -1, -1):
        if p[j] > p[i]:
            break
    else: brak = True
    if brak:
        print("Brak następnika")
        return
    p[i], p[j] = p[j], p[i]
    p[i+1:] = p[i+1:][::-1]
    return p


# Returns the rank (int) of the permutation p of the set {1, ..., n} in lexicographic order.
# e.g. permutationRank([2, 4, 1, 3]) == 10
def permutationRank(p):
    r = 0
    n = len(p)
    p = [0]+p
    for j in range(1,n+1):
        r += (p[j]-1)*math.factorial(n-j)
        p[j+1:] = [i-1 if i > p[j] else i for i in p[j+1:]]
    return r

# Returns a permutation (array) of the set {1, ..., n} from rank r in lexicographic order.
# e.g. permutationFromRank(4, 10) == [2, 4, 1, 3]
def permutationFromRank(n, r):
    p = [0]+[1 for i in range(n)]
    for j in range(1, n):
        d = r%math.factorial(j+1)/math.factorial(j)
        r = r - d * math.factorial(j)
        p[n-j] = d + 1
        for i in range (n-j+1, n+1):
            if p[i] > d:
                p[i] += 1
    return [int(i) for i in p[1:]]


# Returns the number of divisions of n into k components (dynamic programming method)
# e.g. numDivision(7, 3) == 4
def numDivision(n, k):
    P = [[0 for j in range(k+1)] for i in range(n+1)]
    P[0][0] = 1
    for i in range(1, n+1):
        for j in range(1, min(i, k)+1):
            if i < 2*j:
                P[i][j] = P[i-1][j-1]
            else:
                P[i][j] = P[i-1][j-1] + P[i-j][j]
    return P[n][k]


# Returns conjugate division (array) to a given division (a1, ..., am) of n.
# e.g. conjugateDivision([4, 3, 2, 2, 1]) == [5, 4, 2, 1]
def conjugateDivision(a):
    m = len(a)
    a = [0]+a
    b = [0] + [1 for i in range(1,a[1]+1)]
    for j in range(2, m+1):
        for i in range(1, a[j]+1):
            b[i] += 1
    return(b[1:])


# Returns all divisions (array) of the number n in normal form using recursion.
# e.g. allDivisions(5) == [
# [1, 1, 1, 1, 1], [2, 1, 1, 1], [2, 2, 1], [3, 1, 1], [3, 2], [4, 1], [5]
# ]
def allDivisions(n):
    def divide(n, b, m, aL, output):
        if n == 0:
            output.append(aL[1:m+1])
        else:
            for i in range(1, min(b,n)+1):
                aL[m+1] = i
                divide(n-i, i, m+1, aL, output)

    aL = [0 for i in range(n+1)]
    output = []
    divide(n, n, 0, aL, output)
    return output


# Returns Prufer's code (array) for a given tree of E edges with n vertices.
# e.g. prufer(8, [{1, 7}, {1, 6}, {1, 4}, {6, 8}, {4, 5}, {8, 3}, {8, 2}]) == 
#   [1, 4, 1, 8, 8, 6]
def prufer(n, E):
    L = []
    d = [0 for i in range(n+1)]
    for xy in E:
        xy = list(xy)
        d[xy[0]] += 1
        d[xy[1]] += 1
    for i in range(n-2):
        x = n
        while(d[x] != 1):
            x -= 1
        y = n
        while {x, y} not in E:
            y -= 1
        L.append(y)
        E.remove({x, y})
        d[x] -= 1
        d[y] -= 1
    return L


# Returns E edges (array of sets) for a tree corresponding to the given code of Prufer L.
# e.g. fromPrufer([1, 4, 1, 8, 8, 6]) == [
# {1, 7}, {4, 5}, {1, 4}, {8, 3}, {8, 2}, {8, 6}, {1, 6}
# ]
def fromPrufer(L):
    n = len(L)+2
    L = [0]+L
    E = []
    d = [1 for i in range(n+1)]
    for i in range(1, n-1):
        d[L[i]] += 1
    for i in [i for i in range(1, n-1)]+[1]:
        x = n
        while d[x] != 1:
            x -= 1
        y = L[i]
        d[x] -= 1
        d[y] -= 1
        E.append({x,y})
    return E


# Returns the rank (int) of the Prufer L code
# e.g. pruferRank([1, 4, 1, 8, 8, 6]) == 12797
def pruferRank(L):
    n = len(L)+2
    L = [i-1 for i in L]
    r = int(''.join([str(i) for i in L]), n)
    return r


# Returns Prufer code (array) of length n - 2 with the rank r
# e.g. pruferFromRank(8, 12797) == [1, 4, 1, 8, 8, 6]
def pruferFromRank(n, r):
    L = [0 for i in range(n-1)]
    for i in range(n-2, 0, -1):
        L[i] = (r%n)+1
        r = math.floor((r-L[i]+1)/n)
    return L[1:]


# Returns a minimal spanning tree (in form of E edges) in a given graph with weights on the edges
# and the total weight of the spanning tree using Prim's algorithm.
# e.g. 
# V = [1, 2, 3, 4, 5, 6, 7]
# E = [{1,2}, {1,3}, {2,3}, {2,5}, {3,6}, {2,4}, {3,4}, {4,5}, {4,6}, {5,6}, {5,7}, {6,7}]
# w = {(1,2):1, (1,3):4, (2,3):2, (2,5):5, (3,6):2, (2,4):3, (3,4):3, (4,5):1, (4,6):4, (5,6):3, (5,7):5, (6,7):4}
# g = G(V, E, w)
# prim(g) == {
# 'weight': 13,
# 'E': [{1, 2}, {2, 3}, {3, 6}, {2, 4}, {4, 5}, {6, 7}]
# }
def prim(g):
    n = len(g.V)
    W = [[0 for j in range(n+1)] for i in range(n+1)]
    for i in range(1, len(g.V)+1):
        for j in range(1, len(g.V)+1):
            W[i][j] = g.edgeWeight(i, j)
    F = []
    N = [0 for i in range(n+1)]
    D = [0 for i in range(n+1)]
    for i in range (2, n+1):
        N[i] = 1
        D[i] = W[1][i]
    for repeat in range(n-1):
        min = float('inf')
        for i in range(2,n+1):
            if 0 <= D[i] and D[i] < min:
                min = D[i]
                k = i
        e = {k, N[k]}
        F.append(e)
        D[k] = -1
        for i in range(2, n+1):
            if(W[i][k] < D[i]):
                D[i] = W[i][k]
                N[i] = k
    sumWeight = sum([g.edgeWeight(i, j) for i, j in F])
    return {"weight": sumWeight, "E": F}
        
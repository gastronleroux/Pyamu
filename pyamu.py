import math

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
        
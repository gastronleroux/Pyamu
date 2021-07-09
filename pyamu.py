
# Combinatorial Algorithms
# ------------------------------------------------------------------

# Check if x is None. If so, returns x, if not, returns y.
def ifVarDef(x, y):
    if x is None: return y
    else: return x

# Returns an array of arrays containing numbers in a lexicographical
# n = number of elements, k = maximum value, initial = minimum value
def lexicalOrder(n, k = None, initial = 1):
    k = ifVarDef(k, n)

    if initial > k: raise ValueError("Initial value can't be bigger than the maximum value")
    if n < 1: raise ValueError("Number of values must be a positive number")

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
# The key is a binary string (tuples), the value is a {1, 2, ..., n} subset (array), e.g. bij(2)[[1, 0]] == [1]
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


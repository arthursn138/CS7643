from os import replace
import numpy as np

# Q1: Create a zero vector of size 10
def q1():
    x = np.zeros((10))
    return x

# Q2: Create an int64 zero matrix of size (10, 10) with the diagonal values set to -1
def q2():
    x = np.zeros((10, 10), dtype=np.int64)
    # x = x0.copy()
    for i in range(10):
        x[i,i] = -1
    return x

# Q3: Create an 8x8 matrix and fill it with a checkerboard pattern with 0s and 1s (starting with 0 at [0, 0])
def q3():
    x = np.zeros((8,8))
    for i in range(8):
        if i % 2 == 0:
            for j in range(8):
                if j % 2 != 0:
                    x[i, j] = 1
        else:
            for j in range(8):
                if j % 2 == 0:
                    x[i, j] = 1

    return x

# Q4: Randomly place five 1s in a given zero matrix
def q4(x):
    r, c = x.shape
    f = x.flatten()
    idx = np.random.choice(f.shape[0], 5, replace=False)
    # print(idx)
    for i in idx:
        f[i] = 1
    x = f.reshape(r, c)
    return x

# Q5: Given a tensor (image) of dimensions of (H, W, C), return the same tensor as (C, H, W)
def q5(im):
    h, w, c = im.shape
    # im = im.reshape(c, h, w)
    im = np.moveaxis(im, 2, 0)
    return im

# Q6: Given a tensor (image) of dimension (C, H, W) with channel ordering of RGB, swap the channel ordering to BGR
def q6(im):
    c, h, w = im.shape
    bgr = np.array((im[2,:,:], im[1,:,:], im[0,:,:]))
    
    return bgr

# Q7: Given a 1D array, negate (i.e., multiply by -1) all elements in indices [3, 8], in-place
def q7(x):
    x = np.concatenate((x[:3], -x[3:9], x[9:]))
    return x

# Q8: Convert a float64 array to a uint8 array
def q8(x):
    x = x.astype(np.uint8)
    return x

# Q9: Subtract the mean of each row in the matrix (i.e., subtract the mean of row1 from each element in row1 and continue)
def q9(x):
    x = x - np.mean(x, axis=1).reshape(x.shape[0], -1)
    return x

# Q10: The same as Q9, but without a loop (if you used a loop)
def q10(x):
    x = x - np.mean(x, axis=1).reshape(x.shape[0], -1)
    return x

# Q11: Sort the rows of a matrix by the matrix's second column
def q11(x):
    x = x[x[:, 1].argsort()]
    return x

# Q12: Convert an array of size 5 with N=10 (all are values within [0, 9]) to a one-hot encoding matrix as described in the notebook
def q12(x):
    return x

# Q13: In a single expression, multiply the n-th row of a given matrix with the n-th element in a given vector.
def q13(x, y):

    return x

# Q14: Without using `np.pad`, pad an array with a border of zeros
def q14(x):
    r, c = x.shape
    v = np.zeros((c + 2, 0))
    h = np.zeros((r, 0))
    x = np.hstack((h.T, x, h.T))
    x = np.vstack((v, x, v))
    return x


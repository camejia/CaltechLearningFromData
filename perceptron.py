# Online Homework #1, Problems 7-10

import numpy as np

d = 2 # dimensionality
N = 10 # number of training data points
ntrial = 1000 # number of trials to run
Ntest = 2 * N # number of testing data points

# Initialize number of iterations, and number of disagreements on classification of a point
# Note that this total is not reset after each trial (i.e. include ntrial factor in average at the end)
niter = 0
ndisagree = 0
for trial in range(ntrial):
    # random points, includes training (N) and test (Ntest)
    xall = np.random.uniform(-1, 1, (N + Ntest, d))
    # start and stop endpoints of dividing segment
    origin = np.random.uniform(-1, 1, d)
    endpt = np.random.uniform(-1, 1, d)

    # shift so start is the origin (makes later calculations simpler)
    xall -= origin
    endpt -= origin
    
    # Cross product to determine whether x is left or right of the line
    yall = np.sign(xall[:, 0] * endpt[1] - xall[:, 1] * endpt[0])
    
    # Augment the x matrix with ones for bias
    xall = np.concatenate((np.ones((N + Ntest, 1)), xall), axis=1)
    
    # Separate the training data from the testing data
    x = xall[: N, :]
    xtest = xall[N :, :]
    y = yall[: N]
    ytest = yall[N :]
    
    # Start the PLA with the weight vector w being all zeros
    w = np.zeros((d + 1, 1))
    
    while True:
        # Calculate h per the perceptron algorithm
        h = np.sign(np.dot(x, w))
        # Check if h matches y
        nz = np.flatnonzero(y - h[:, 0])
        if np.size(nz) == 0:
            break
        # Choose a point randomly from the set of misclassified points
        ind = np.random.choice(nz)
        # Update w per the perceptron algorithm
        w[:, 0] += y[ind] * x[ind, :]
        # TBR: Not sure if niter should be incremented here, or at the top of the while loop
        niter += 1
    # Check our predictions on the test data.  Equations are same as above
    h = np.sign(np.dot(xtest, w))
    nz = np.flatnonzero(ytest - h[:, 0])
    ndisagree += np.size(nz)

print 'avg. iter: ', float(niter) / float(ntrial)
print 'avg. disagree: ', float(ndisagree) / (float(ntrial) * float(Ntest))
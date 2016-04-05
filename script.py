import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys, time

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD

    classes = int(np.max(y))    
    r,c = X.shape
    means = np.empty((c, classes))

    #print means

    covmat = np.zeros(c)
    for i in xrange (1, classes + 1):
        subMat = X[y.flatten() == i,:]
        covmat = covmat + (subMat.shape[0]-1) * np.cov(np.transpose(subMat))
        means[:, i-1] = np.transpose(np.mean(subMat, axis=0))        

    covmat = (1.0/(r - classes)) * covmat

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    classes = int(np.max(y))
    r,c = X.shape
    means = np.empty((c, classes))

    #print means

    covmats = []

    for i in xrange (1, classes+1):
        subMat = X[y.flatten() == i,:]
        covmats.append(np.cov(np.transpose(subMat)))
        means[:, i-1] = np.transpose(np.mean(subMat, axis=0))
        
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    count = 0.0
    inv = np.linalg.inv(covmat)
    ytest = ytest.astype(int)
    r,c = Xtest.shape
    classes = means.shape[1]
    
    for i in xrange (1, c + 1):
        pd,sno = 0,0
        row = np.transpose(Xtest[i-1,:])

        for j in xrange (1, classes + 1):
            expow = row - means[:, j-1]
            res = np.exp((-1/2)*np.dot(np.dot(np.transpose(row - means[:, j-1]),inv),expow))
            if (res > pd):
                sno,pd = j,res
        
        if (sno == ytest[i-1]):
            count = count + 1

    acc = count/c

    return acc,ytest

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    count = 0.0    
    r,c = Xtest.shape    
    classes = means.shape[1];
    normalizers = np.zeros(classes)
    covmats_tmp = np.copy(covmats)

    ytest = ytest.astype(int)
        
    #calculating the accuracy of QDA training
    for i in range (1, classes+1):
        d = np.shape(covmats_tmp[i-1])[0]
        normalizers[i-1] = 1.0/(np.power(2*np.pi, d/2)*np.power(np.linalg.det(covmats_tmp[i-1]),1/2))

        covmats_tmp[i-1] = np.linalg.inv(covmats_tmp[i-1])

    
    for i in range (1, r + 1):
        pd,sno = 0,0

        row = np.transpose(Xtest[i-1,:])
        for k in range (1, classes+1):
            invCov = covmats_tmp[k-1]
            ex_pow = row - means[:, k-1]
            result = normalizers[k-1]*np.exp((-1/2)*np.dot(np.dot(np.transpose(row - means[:, k-1]),invCov),ex_pow))
            if (result > pd):
                sno,pd = k,result

        if (sno == ytest[i-1]):
            count = count + 1

    acc = count/r
    return acc,ytest


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    
    
    # formula to calculate w using least square approach: w = ((Xt.X)^-1).Xt.y
    Xt = np.transpose(X)
    term1 = np.dot(Xt, X)
    term1_inv = inv(term1)
    term2 = np.dot(term1_inv, Xt)
    w = np.dot(term2, y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # formula to calculate w using ridge regression (norm 2): w = ((Xt.X + lambda.I)^-1).Xt.y
    d = len(X[0])
    I = np.identity(d)
    Xt = np.transpose(X)
    prod1 = np.dot(Xt, X)
    prod2 = np.dot(lambd, I)
    inver = inv(np.add(prod1, prod2))
    prod3 = np.dot(inver, Xt)
    w = np.dot(prod3, y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # rmse
    
    
    #wt = np.transpose(w)
    sqsum = 0
    n = len(Xtest)
    for i in xrange(n):
        xi = Xtest[i]
        yi = ytest[i]
        #dot = np.dot(wt, xi)
        dot = np.dot(xi,w)
        diff = yi - dot
        sqsum += diff**2
        
    rmse = sqrt(sqsum)/n
    return rmse
    
def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             

    N = X.shape[0]
    w1=np.array([w]).T
#    print(w.shape)
#    print(X.shape)
#    print(y.shape)
#    print(lambd)
    Xw = np.dot(X,w)
#    print(Xw.shape)
    yx = y - Xw
#    print(yx.shape)
    error = np.sum(np.square(yx))/N + lambd*np.dot(w.T,w)
    error_grad = np.dot(y.T,X)/(-2*N) + np.dot(w,np.dot(X.T,X))/N + lambd*w.T
#    print(error.shape)
#    print(error_grad.shape)

    return error.flatten(), error_grad.flatten()


def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD

    N=x.shape[0]
    Xd = np.ones((p+1,N))
    for i in xrange(p+1):
        Xd[i,:] = np.power(x,i)

    Xd = np.transpose(Xd)
    return Xd

# Main script

# Problem 1
print "=======PROBLEM 1 ========"

# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

plt.title("Ridge regression (Lambda vs RMSE)")
zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

# Problem 2

print "=======PROBLEM 2========"
start = time.time()
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))
end = time.time()
print "Time: %.3fs" %(end-start)


# Problem 3
print "=======PROBLEM 3========"
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
min_rmse = sys.float_info.max
opt_lambda = 0
start = time.time()
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    
    if rmses3[i] < min_rmse:
        min_rmse = rmses3[i]
        opt_lambda = lambd

    i = i + 1
    
print "Optimal lambda is: ", opt_lambda
print "Min RMSE is: ", min_rmse
end = time.time()
print "Time: %.3fs" %(end-start)

plt.figure()
plt.title("Ridge regression (Lambda vs RMSE)")
plt.plot(lambdas,rmses3)
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.legend(('Test','Train'))
plt.show()


# Problem 4
print "=======PROBLEM 4========"
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.title("Gradient Descent for Ridge regression (Lambda vs RMSE)")
plt.plot(lambdas,rmses4)
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.legend(('Test','Train'))
plt.show()

# Problem 5
print "=======PROBLEM 5========"
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.title("Non-linear regression (Attributes vs RMSE)")
plt.plot(range(pmax),rmses5)
plt.xlabel('Attributes')
plt.ylabel('RMSE')

plt.legend(('No Regularization','Regularization'))
plt.show()

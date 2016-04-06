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

    classes = np.unique(y)
    means_l=[]
    cov_l=[]
    
    for i in classes:
        subMat = X[y.flatten() == i,:]
        mean_t = np.mean(subMat,axis=0)

        sub_diff =subMat - mean_t
        
        cov_l.append(np.cov(sub_diff,rowvar =0))
        
        means_l.append(mean_t)
        
    means=np.asarray(means_l)

    covmat=cov_l[0]

    covmat = np.cov(X-np.mean(X,axis=0),rowvar =0)
        
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
        
    classes = np.unique(y)
    means_l=[]
    cov_l=[]
    for i in classes:
        subMat = X[y.flatten() == i,:]
        mean_t = np.mean(subMat,axis=0)

        sub_diff = subMat - mean_t
        
        cov_l.append(np.cov(sub_diff,rowvar =0))
        
        means_l.append(mean_t)

    means=np.asarray(means_l)        
  
    return means,cov_l

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    out=[]

    for i in range(means.shape[0]):
        inv_covar = np.linalg.inv(covmat)
        det_covar = np.linalg.det(covmat)
        outlist=[]
        D=inv_covar.shape[0]
        
        for x in Xtest:            
            b=(np.sqrt(det_covar)*(np.power(np.pi*2,D/1)))
            p=np.dot((x - means[i]).reshape(1,-1),inv_covar)
            q=np.dot(p,(x - means[i]).reshape(-1,1))
            

            temp_pred_per_t=np.exp(-0.5*q[0][0])/b;        
            outlist.append(temp_pred_per_t)
            outlist_arr=np.asarray(outlist,dtype='float32')

        out.append(outlist_arr.flatten())
    ops=np.asarray(out,dtype='float32')
        
    count=0
    pred_t=[]

    for y in range(len(ytest)):        
        acc1=np.argmax(ops[:,y])+1
        pred_t.append(acc1)
    
    pred=np.asarray(pred_t,dtype='float32').reshape(-1,1)
    
    acc=100*np.mean((pred == ytest).astype(float))

    return acc,pred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    out=[]

    for i in range(means.shape[0]):
        inv_covar = np.linalg.inv(covmats[i])
        det_covar = np.linalg.det(covmats[i])

        outlist=[]
        D=inv_covar.shape[0]
        for x in Xtest:
            b=(np.sqrt(det_covar)*(np.power(np.pi*2,D/2)))
            p=np.dot((x - means[i]).reshape(1,-1),inv_covar)
            q=np.dot(p,(x - means[i]).reshape(-1,1))
            

            temp_pred_per_t=np.exp(-0.5*q[0][0])/b;        
            outlist.append(temp_pred_per_t)
            outlist_asarray=np.asarray(outlist,dtype='float32')            
        out.append(outlist_asarray.flatten())

    our_outputs=np.asarray(out,dtype='float32')    
    count=0

    pred_t=[]

    for y in range(len(ytest)):        
        pred_t.append(np.argmax(our_outputs[:,y])+1)
    
    pred=np.asarray(pred_t,dtype='float32').reshape(-1,1)
    acc=100*np.mean((pred == ytest).astype(float))
    
    return acc,pred

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
        
    rmse = sqrt(sqsum/n)
    return rmse
    
def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             

    N = X.shape[0]
  #  w = np.reshape(w, (w.size, 1))
    w1=np.array([w]).T
#    print(w.shape)
#    print(X.shape)
#    print(y.shape)
#    print(lambd)
    Xw = np.dot(X,w)
#    print(Xw.shape)
    yx = y - Xw
#    print(yx.shape)
    error = np.sum(np.square(yx))/(2*N) + lambd*np.dot(w.T,w)
#    error=np.dot(np.transpose(yx),yx)/(2*N) + np.dot(lambd,np.dot(w.T,w))
    error_grad = np.dot(y.T,X)/(-2*N) + np.dot(w,np.dot(X.T,X))/N + lambd*w
#    print(error)
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
ldaacc,_ = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,_ = qdaTest(means,covmats,Xtest,ytest)
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
plt.show()

plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.show()

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
rmse_test = testOLERegression(w,Xtest,ytest)
rmse_train = testOLERegression(w, X, y)

w_i = learnOLERegression(X_i,y)
rmse_i_test = testOLERegression(w_i,Xtest_i,ytest)
rmse_i_train = testOLERegression(w_i, X_i, y)

print('RMSE without intercept (Test) '+str(rmse_test))
print('RMSE with intercept (Test) '+str(rmse_i_test))
print('RMSE without intercept (Train) '+str(rmse_train))
print('RMSE with intercept (Train) '+str(rmse_i_train))

end = time.time()
print "Time: %.3fs" %(end-start)


# Problem 3
print "=======PROBLEM 3========"
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3_test = np.zeros((k, 1))
rmses3_train = np.zeros((k, 1))
min_rmse = sys.float_info.max
opt_lambda = 0
start = time.time()
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3_test[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3_train[i] = testOLERegression(w_l, X_i, y)
    
    if rmses3_test[i] < min_rmse:
        min_rmse = rmses3_test[i]
        opt_lambda = lambd
        min_rmse_train = rmses3_train[i]

    i = i + 1
    
print "Optimal lambda is: ", opt_lambda
print "Min RMSE is: ", min_rmse
print "Min RMSE (train) is ", min_rmse_train
end = time.time()
print "Time: %.3fs" %(end-start)

plt.figure()
plt.title("Ridge regression (Lambda vs RMSE)")
plt.plot(lambdas,rmses3_test)
plt.plot(lambdas,rmses3_train)
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.legend(('Test','Train'))
plt.show()


print "===========OLE vs Ridge============="
w_rr = learnRidgeRegression(X_i, y, opt_lambda)
print "Sum of weight elements (OLE): ", np.sum(w_i)
print "Sum of weight elements (Ridge): ", np.sum(w_rr)
print "Variance (OLE): ", np.var(w_i)
print "Variance (Ridge)", np.var(w_rr)

plt.figure()
plt.title("OLE vs Ridge weights")
plt.plot(range(0, w_i.shape[0]), w_i)
plt.plot(range(0, w_rr.shape[0]), w_rr)
plt.xlabel('Weights')
plt.legend(('OLE','Ridge'))
plt.show()



# Problem 4
print "=======PROBLEM 4========"
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
rmses4_train =np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
min_rmse = sys.float_info.max
opt_lambda = 0
start = time.time()
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    rmses4_train[i] = testOLERegression(w_l_1,X_i,y)

    if rmses4[i] < min_rmse:
        min_rmse = rmses4[i]
        opt_lambda = lambd

    i = i + 1

print "Optimal lambda is: ", opt_lambda
print "Min RMSE is: ", min_rmse
end = time.time()
print "Time: %.3fs" %(end-start)

plt.title("Gradient Descent for Ridge regression (Lambda vs RMSE)")
plt.plot(lambdas,rmses4)
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.legend(('Test','Train'))
plt.show()

plt.title('Plot Train Data')
plt.plot(lambdas,rmses4_train)
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.legend(('Train','Test'))
plt.show()

# Problem 5
print "=======PROBLEM 5========"
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
min_rmse = sys.float_info.max
min_rmse_reg = sys.float_info.max
start = time.time()
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    rmse5_train = testOLERegression(w_d1, Xd, y)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    rmse5_train_reg = testOLERegression(w_d2, Xd, y)
    
    if rmses5[p,0] < min_rmse:
        min_rmse = rmses5[p,0]
        min_rmse5_train = rmse5_train
        #min_p = p
      
    if rmses5[p,1] < min_rmse_reg:
        min_rmse_reg = rmses5[p,1]
        min_rmse5_train_reg = rmse5_train_reg
        #min_p_reg = p 

print "Min RMSE is: ", min_rmse
print "Min RMSE (reg) is: ", min_rmse_reg
print "Min RMSE train (no reg) is: ", min_rmse5_train
print "Min RMSE train (reg) is: ", min_rmse5_train_reg
end = time.time()
print "Time: %.3fs" %(end-start)
plt.title("Non-linear regression (Attributes vs RMSE)")
plt.plot(range(pmax),rmses5)
plt.xlabel('Attributes')
plt.ylabel('RMSE')

plt.legend(('No Regularization','Regularization'))
plt.show()

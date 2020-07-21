"""
Module with artificial neural networks.
"""

import numpy as np

from .coretools import ensure_np_array

def _softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)

def _sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def _relu(x):
    return np.maximum(x,0)

def _drelu(x):
    return np.heaviside(x,0.5)

class ANN:
    # constructor, allocate space for parameters
    def __init__(self,layers,random_weights=False,init_b=0.0,activation="softplus"):
        if activation == 'softplus':
            self._activation=_softplus
            self._dactivation=_sigmoid
        elif activation == 'relu':
            self._activation=_relu
            self._dactivation=_drelu
        else:
            raise ValueError("Unknown activation type: "+activation)
        self.layers=layers
        self.W=[]
        self.b=[]
        for i in range(len(self.layers)-1):
            self.W.append(np.zeros(shape=(self.layers[i],self.layers[i+1])))
            self.b.append(np.zeros(shape=(self.layers[i+1]))+init_b)
        self.W.append(np.zeros(shape=(self.layers[len(layers)-1],1)))
        self.b.append(np.zeros(shape=(1))+init_b)

        n=0
        for i in range(len(self.W)):
            n+=np.prod(self.W[i].shape)
        self.nparW=n

        n=0
        for i in range(len(self.b)):
            n+=np.prod(self.b[i].shape)
        self.nparB=n

        self.npar=self.nparW + self.nparB

        self.narg=self.layers[0]

        if random_weights:
            # see Deep Learning, Eq. 8.23
            for i in range(len(self.W)):
                self.W[i]+=np.random.uniform(-1.0,1.0,size=self.W[i].shape)*np.sqrt(6/(np.sum(self.W[i].shape)))

    # set array of parameters
    def setpar(self,par):
        assert len(par)==self.npar
        n=0
        for i in range(len(self.W)):
            self.W[i]=np.reshape(par[n:n+np.prod(self.W[i].shape)],self.W[i].shape)
            n+=np.prod(self.W[i].shape)
        for i in range(len(self.b)):
            self.b[i]=np.reshape(par[n:n+np.prod(self.b[i].shape)],self.b[i].shape)
            n+=np.prod(self.b[i].shape)

    def getpar(self):
        par=np.zeros(self.npar)
        n=0
        for i in range(len(self.W)):
            m=np.prod(self.W[i].shape)
            par[n:n+m]=self.W[i].flatten()
            n+=m
        for i in range(len(self.b)):
            m=np.prod(self.b[i].shape)
            par[n:n+m]=self.b[i]
            n+=m
        return par

    # compute function and derivative wrt flatten parameters for a vector of samples
    def derpar(self,x):
        x = ensure_np_array(x)

        if len(x.shape)==1:
          f, der = self.derpar(x.reshape((-1,len(x))))
          return f[0], der[0]
        elif len(x.shape)>2:
          raise TypeError("Incorrectly shaped x")

        assert x.shape[1]==self.narg
        f,df_dW,df_db=self.deriv(x)
        der=np.zeros((x.shape[0],self.npar))
        n=0
        for i in range(len(self.W)):
            m=np.prod(self.W[i].shape)
            der[:,n:n+m]=df_dW[i].reshape(x.shape[0],-1)
            n+=m
        for i in range(len(self.b)):
            m=np.prod(self.b[i].shape)
            der[:,n:n+m]=df_db[i].reshape(x.shape[0],-1)
            n+=m
        return f,der

    # evaluate the NN on a single point or on an array of points
    def apply(self,x):
        x = ensure_np_array(x)
        if len(x.shape)==1:
            return self.apply(x.reshape((1,len(x))))[0]
        elif len(x.shape)>2:
            raise TypeError("Incorrectly shaped x")
        assert x.shape[1]==self.narg
        for i in range(len(self.W)):
            x=np.matmul(x,self.W[i])+self.b[i]
            if i+1<len(self.W):
                x = self._activation(x)
        return x[:,0]

    # compute derivatives with respect to parameters
    def deriv(self,x):

        x = ensure_np_array(x)

        if len(x.shape)==1:
            f,df_dW,df_db = self.deriv(x.reshape((1,len(x))))
            for i in range(len(df_dW)):
                df_dW[i]=df_dW[i][0]
                df_db[i]=df_db[i][0]
            return f[0], df_dW, df_db
        elif len(x.shape)>2:
            raise TypeError("Incorrectly shaped x")

        assert x.shape[1]==self.narg

        vec=x.shape[0]

        # allocate hidden nodes
        h=[None]*len(self.layers) # hidden nodes
        ht=[None]*len(self.layers) # non-linear functions of hidden nodes

        # allocate derivatives
        df_dW=[None]*len(self.layers)
        df_db=[None]*len(self.layers)

        # forward propagation
        ht[0]=+x

        for i in range(len(self.layers)-1):
            h[i+1]=np.matmul(ht[i],self.W[i])+self.b[i]
            ht[i+1] = self._activation(h[i+1])

        f=(np.matmul(ht[-1],self.W[-1])+self.b[-1])[:,0]

        # backward propagation

        df_db[-1]=np.ones((vec,1))
        df_dW[-1]=ht[-1][:,:,np.newaxis]

        for i in reversed(range(len(self.layers)-1)):
            df_db[i]=np.matmul(df_db[i+1],self.W[i+1].T) * self._dactivation(h[i+1])
            df_dW[i]=ht[i][:,:,np.newaxis]*df_db[i][:,np.newaxis,:]

        return f,df_dW,df_db

    # these are aliases for backward compatibility
    def applyVec(self,x):
        return self.apply(x)
    def derivVec(self,x):
        return self.deriv(x)
    def derparVec(self,x):
        return self.derpar(x)

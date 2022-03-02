"""
Module with artificial neural networks.

ANN can be constructed with `cuda=True`, in which case it will use cudamat.
"""

import numpy as np

try:
    import cudamat as cm  # pylint: disable=import-error
    _HAS_CUDAMAT=True
except ModuleNotFoundError:
    _HAS_CUDAMAT=False

def _ensure_cm_init():
    if not hasattr(cm.CUDAMatrix, 'ones'):
        cm.cublas_init()
    elif not isinstance(cm.CUDAMatrix.ones,cm.CUDAMatrix):
        cm.cublas_init()

from . import coretools
from .coretools import ensure_np_array

def _softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)

def _sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def _relu(x):
    return np.maximum(x,0)

def _drelu(x):
    return np.heaviside(x,0.5)

def _sigmoid_cudamat(x):
    cm.sigmoid(x)

def _relu_cudamat(x):
    x.maximum(0.0)

def _drelu_cudamat(x):
    x.greater_than(0.0)

class ANN:
    # constructor, allocate space for parameters
    def __init__(self,layers,random_weights=False,init_b=0.0,activation="softplus",cuda=False):

        if cuda is None:
            cuda = _HAS_CUDAMAT

        if cuda:
            _ensure_cm_init()

        self.cuda=cuda

        if not self.cuda:
            if activation == 'softplus':
                self._activation=_softplus
                self._dactivation=_sigmoid
            elif activation == 'relu':
                self._activation=_relu
                self._dactivation=_drelu
            else:
                raise ValueError("Unknown activation type: "+activation)
        else:
            if not _HAS_CUDAMAT:
                raise ValueError("Cudamat not available, can only run ANN with numpy")
            if activation == 'softplus':
                self._cu_activation=cm.log_1_plus_exp
                self._cu_dactivation=_sigmoid_cudamat
            elif activation == 'relu':
                self._cu_activation=_relu_cudamat
                self._cu_dactivation=_drelu_cudamat
            else:
                raise ValueError("Unknown activation type: "+activation)
        self.activation=activation
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
        if self.cuda:
            self.cuda_setup()

    def cuda_setup(self):
        self.cu_W=[]
        for i in range(len(self.W)):
            self.cu_W.append(cm.CUDAMatrix(self.W[i]))
        self.cu_b=[]
        for i in range(len(self.b)):
            self.cu_b.append(cm.CUDAMatrix(np.reshape(self.b[i],(1,-1))))

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
        if self.cuda:
            self.cuda_setup()
        return self

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
        der=np.zeros((x.shape[0],self.npar),dtype=f.dtype)
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

        if not self.cuda:
            for i in range(len(self.W)):
                x=np.matmul(x,self.W[i])+self.b[i]
                if i+1<len(self.W):
                    x = self._activation(x)
        else:
            cu_x=cm.CUDAMatrix(x)
            for i in range(len(self.cu_W)):
                cu_x=cm.dot(cu_x,self.cu_W[i])
                cu_x.add_row_vec(self.cu_b[i])
                if i+1<len(self.cu_W):
                    self._cu_activation(cu_x)
            x=np.array(cu_x.asarray())

        return x[:,0]


    def forward(self,x):
        x = ensure_np_array(x)

        if len(x.shape)==1:
            f,df_dW,df_db = self.deriv(x.reshape((1,len(x))))
            for i in range(len(df_dW)):
                df_dW[i]=df_dW[i][0]
                df_db[i]=df_db[i][0]
            return f[0], df_dW, df_db
        elif len(x.shape)>2:
            raise TypeError("Incorrectly shaped x")

        # allocate hidden nodes
        h=[None]*len(self.layers)  # hidden nodes
        ht=[None]*len(self.layers)  # non-linear functions of hidden nodes

        if not self.cuda:
            # forward propagation
            ht[0]=x.copy()
            for i in range(len(self.layers)-1):
                h[i+1]=np.matmul(ht[i],self.W[i])+self.b[i]
                ht[i+1] = self._activation(h[i+1])
            f=(np.matmul(ht[-1],self.W[-1])+self.b[-1])[:,0]
        else:
            ht[0]=cm.CUDAMatrix(x)
            for i in range(len(self.cu_W)-1):
                h[i+1]=cm.dot(ht[i],self.cu_W[i])
                h[i+1].add_row_vec(self.cu_b[i])
                ht[i+1]=h[i+1].copy()
                self._cu_activation(ht[i+1])
            f=cm.dot(ht[-1],self.cu_W[-1])
            f.add_row_vec(self.cu_b[-1])

        class State(coretools.Result):
            pass
        return State(f=f,h=h,ht=ht)

    def backward(self,deriv,hidden):
        # allocate derivatives
        df_dW=[None]*len(self.layers)
        df_db=[None]*len(self.layers)

        if not self.cuda:
            df_db[-1]=np.ones((len(hidden.f),1))
            df_dW[-1]=np.matmul(deriv,hidden.ht[-1])[:,np.newaxis]
            for i in reversed(range(len(self.layers)-1)):
                df_db[i]=np.matmul(df_db[i+1],self.W[i+1].T) * self._dactivation(hidden.h[i+1])
                df_dW[i]=np.einsum("i,ij,ik->jk",deriv,hidden.ht[i],df_db[i])
            for i in range(len(self.layers)):
                df_db[i]=np.matmul(deriv,df_db[i])

        else:
            if not isinstance(deriv,cm.CUDAMatrix):
                deriv=deriv.reshape((1,-1))
                deriv=cm.CUDAMatrix(deriv)

            if deriv.shape[1]==1:
                deriv=deriv.transpose()

            vec=deriv.shape[1]

            df_db[-1]=cm.CUDAMatrix(np.ones((vec,1)))
            df_dW[-1]=cm.dot(deriv,hidden.ht[-1]).transpose()

            for i in reversed(range(len(self.layers)-1)):
                self._cu_dactivation(hidden.h[i+1])
                df_db[i]=cm.dot(df_db[i+1],self.cu_W[i+1].transpose())
                df_db[i].mult(hidden.h[i+1])

                hidden.ht[i].mult_by_col(deriv.transpose())
                df_dW[i]=cm.dot(hidden.ht[i].transpose(),df_db[i])

            for i in range(len(self.layers)):
                df_db[i]=cm.dot(deriv,df_db[i])
            for i in range(len(df_db)):
                df_db[i]=df_db[i].asarray()[0,:]
            for i in range(len(df_db)):
                df_dW[i]=df_dW[i].asarray()

        return df_dW,df_db

    def backward_par(self,deriv,hidden):
        df_dW,df_db=self.backward(deriv,hidden)
        if not self.cuda:
            der=np.zeros(self.npar,dtype=hidden.f.dtype)
        else:
            der=np.zeros(self.npar,dtype=df_dW[0].dtype)
        n=0
        for i in range(len(self.W)):
            m=np.prod(self.W[i].shape)
            der[n:n+m]=df_dW[i].flatten()
            n+=m
        for i in range(len(self.b)):
            m=np.prod(self.b[i].shape)
            der[n:n+m]=df_db[i]
            n+=m
        assert(n==self.npar)
        return der


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
        h=[None]*len(self.layers)  # hidden nodes
        ht=[None]*len(self.layers)  # non-linear functions of hidden nodes

        # allocate derivatives
        df_dW=[None]*len(self.layers)
        df_db=[None]*len(self.layers)

        if not self.cuda:

            # forward propagation
            ht[0]=x.copy()

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

        else:

            # forward propagation
            ht[0]=cm.CUDAMatrix(x)

            for i in range(len(self.cu_W)-1):
                h[i+1]=cm.dot(ht[i],self.cu_W[i])
                h[i+1].add_row_vec(self.cu_b[i])
                ht[i+1]=h[i+1].copy()
                self._cu_activation(ht[i+1])

            f=cm.dot(ht[-1],self.cu_W[-1]).asarray()[:,0]+self.b[-1][0]

            # backward propagation

            df_db_host=[None]*len(self.layers)

            df_db[-1]=cm.CUDAMatrix(np.ones((vec,1)))
            df_dW[-1]=ht[-1].asarray()[:,:,np.newaxis]

            df_db_host[-1]=df_db[-1].asarray()

            for i in reversed(range(len(self.layers)-1)):
                self._cu_dactivation(h[i+1])
                df_db[i]=cm.dot(df_db[i+1],self.cu_W[i+1].transpose())
                df_db[i].mult(h[i+1])
                df_db_host[i]=df_db[i].asarray()  # should be moved to CPU anyway
 
                # this is still on CPU, but could be done on GPU avoiding the movement of ht[i]
                df_dW[i]=ht[i].asarray()[:,:,np.newaxis]*df_db_host[i][:,np.newaxis,:]

            df_db=df_db_host

            f=np.array(f)

        return f,df_dW,df_db

    # these are aliases for backward compatibility
    def applyVec(self,x):
        return self.apply(x)
    def derivVec(self,x):
        return self.deriv(x)
    def derparVec(self,x):
        return self.derpar(x)

    def dumpPlumed(self,path,style="ann",prefix=None,arguments=None):
        if style == "ann":
            with open(path,"w") as f:
                for i in range(len(self.W)):
                    ni=self.W[i].shape[0]
                    no=self.W[i].shape[1]
                    print("#! FIELDS "+" ".join(["w"+str(j) for j in range(ni)]),file=f)
                    for j in range(no):
                        print(' '.join(map(str, self.W[i][:,j])),file=f)
                    print("#! FIELDS "+" ".join(["b"+str(j) for j in range(no)]),file=f)
                    print(' '.join(map(str, self.b[i])),file=f)
                    if i+1 < len(self.W):
                        print("#! FIELDS "+" ".join(["activation"+str(j) for j in range(no)]),file=f)
                        print(' '.join([self.activation]*no),file=f)
        elif style == "combine":
            if not prefix:
                raise ValueError("with style combine, prefix is needed")
            if not arguments:
                raise ValueError("with style combine, arguments is needed")
            with open(path,"w") as f:
                if self.activation=="softplus":
                    func="log(1+exp(-abs(x)))+max(x,0)"
                else:
                    raise ValueError("only activation softplus supported")
                print(prefix+"_one: CONSTANT VALUE=1.0",file=f)
                for i in range(len(self.W)):
                    no=self.W[i].shape[1]
                    for j in range(no):
                        coeff=','.join(map(str, self.W[i][:,j]))
                        coeff+=","+str(self.b[i][j])
                        if i+1<len(self.W):
                            result="{}_h_{}_{}".format(prefix,i,j)
                        else:
                            result="{}_result".format(prefix)
                        print("{}: COMBINE ...".format(result),file=f)
                        print("  PERIODIC=NO",file=f)
                        print("  COEFFICIENTS={}".format(coeff),file=f)
                        print("  ARG={},{}_one".format(arguments,prefix),file=f)
                        print("...",file=f)
                    if i+1<len(self.W):
                        for j in range(no):
                            print("{}_ht_{}_{}: CUSTOM PERIODIC=NO ARG={}_h_{}_{} FUNC={}".format(prefix,i,j,prefix,i,j,func),file=f)
                        arguments=",".join(["{}_ht_{}_{}".format(prefix,i,j) for j in range(no)])
        else:
            raise ValueError("unknown style")


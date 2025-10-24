import numpy as np 

class layers :
    def __init__ (self) :
        pass 

    def get_weight(self) :
        raise NotImplementedError
    
    def get_gradient (self) :
        raise NotImplementedError
    
    def __call__(self,x) :
        raise NotImplementedError
    
    def backward(self,grad_out) :
        raise NotImplementedError
    
    def update_weight(self,w ) :
        raise NotImplementedError

class Linear (layers) :
    def __init__ (self,units) :
        self.units = units 
        self.weight = None 
        self.bias = None 
        self.xhist = None 
        self.gradient_Weight = None 
        self.gradient_bias = None 
    
    def build_weight(self,features) :
        glorot_uniform = np.sqrt(6 / (features + self.units))
        self.weight = np.random.uniform(
            low=-glorot_uniform,high=glorot_uniform,
            size=(features,self.units)
        )
        self.bias = np.zeros((1,self.units))
    
    def __call__(self, x):
        if self.weight is None or self.bias is None :
            self.build_weight(x.shape[-1])
        self.xhist = x 
        return np.matmul(x,self.weight) + self.bias 
    
    def backward(self, grad_out):
        grad = np.matmul(np.swapaxes(self.xhist,-2,-1),grad_out)
        self.gradient_Weight = grad 
        self.gradient_bias = np.sum(grad_out,axis=0,keepdims=True)
        return np.matmul(grad_out,self.weight.swapaxes(-2,-1))
    
    def get_gradient(self):
        return [self.gradient_Weight,self.gradient_bias]
    
    def get_weight(self):
        return [self.weight,self.bias]
    
    def update_weight(self, w):
        self.weight = w 
    
    def update_bias (self,b) :
        self.bias = b 
    
class ReLU(layers) :
    def __init__(self):
        super().__init__()
        self.hist = None 
    
    def __call__(self, x):
        self.hist = x 
        return np.maximum(0,x)
    
    def backward(self, grad_out):
        return np.where(self.hist > 0,1,0) * grad_out
    
    def get_gradient(self):
        return None 
    
    def get_weight(self):
        return None 
    
    def update_weight(self, w):
        pass 

class Sigmoid (layers) :
    def __init__(self):
        super().__init__()
        self.hist = None 
    
    def __call__(self, x):
        self.hist = 1 / (1 + np.exp(-x))
        return self.hist
    
    def backward(self, grad_out):
        grad =  self.hist * (1 - self.hist)
        return grad * grad_out 
    
    def get_gradient(self):
        return None 
    
    def get_weight(self):
        return None 
    
    def update_weight(self, w):
        pass 

class BinaryCrossEntropy () :
    def __init__(self):
        self.epsilon = 1e-6
        self.cache_y_true = None 
        self.cache_y_pred = None 
    
    def __call__(self,y_true,y_pred) :
        self.cache_y_true = y_true
        self.cache_y_pred = y_pred 
        n = len(y_true)
        loss = (-1/n) * np.sum(y_true * np.log(y_pred + self.epsilon) + 
                               (1 - y_true) * np.log(1 - y_pred  + self.epsilon))
        return loss 
    
    def backward(self) :
        return (self.cache_y_pred - self.cache_y_true) / len(self.cache_y_true)


class Sequential (layers) :
    def __init__ (self,component : list) :
        self.component = component 
    
    def __call__(self,x) :
        for component in self.component :
            x = component(x)
        return x 
    
    def backward(self, grad_out):
        for component in reversed(self.component) :
            if isinstance(component,layers) :
                grad_out = component.backward(grad_out)
    
    def get_weight(self):
        weight = list()
        for component in self.component :
            if isinstance(component,layers) :
                wg = component.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
        return weight
    
    def get_gradient(self):
        gradient = list()
        for component in self.component :
            if isinstance(component,layers) :
                gl = component.get_gradient()
                if gl is not None :
                    for g in gl :
                        gradient.append(g)
        return gradient
    
    def update_weight(self, w : list):
        counter = 0 
        for component in self.component :
            if isinstance(component,Linear) :
                component.update_weight(w[counter])
                counter += 1 
                component.update_bias(w[counter])
                counter +=1 
            
class GlobalAveragePooling (layers) :
    def __init__(self,axis):
        super().__init__()
        self.x_hist = None 
        self.axis = axis 
    
    def __call__(self,x) :
        self.x_hist = x 
        return np.mean(x,axis=self.axis,keepdims=False )
    
    def backward(self, grad_out):
        if isinstance(self.x_hist,np.ndarray):
            if self.axis is None :
                N = self.x_hist.size
            elif isinstance(self.axis,int) :
                N = self.x_hist[self.axis]
            else :
                N = 1 
                for ax in self.axis :
                    N *= self.x_hist.shape[ax]
        grad = np.expand_dims(grad_out,axis = self.axis)
        grad = np.broadcast_to(grad/N,self.x_hist.shape)
        return grad 
    
    def get_gradient(self):
        return None 
    
    def get_weight(self):
        return None 
    
    def update_weight(self, w):
        pass 
            
class Softmax (layers) :
    def __init__ (self,axis=None,keepdim=False) :
        self.axis = axis 
        self.keepdim = keepdim
    
    def get_weight(self):
        return None 
    
    def get_gradient(self):
        return None 
    
    def __call__(self, x):
        xmax = np.max(x,axis=self.axis,keepdims=self.keepdim)
        x_exp = np.exp(x-xmax) 
        x_sum = np.sum(x_exp,axis=self.axis,keepdims=self.keepdim)
        x_sum[x_sum==0] = 1e-6
        return x_exp / x_sum
    
    def backward(self, grad_out):
        return grad_out
    
    def update_weight(self, w):
        pass 


class SparsecategoricalCrossentropy (layers) :
    def __init__ (self) :
        self.y_trainhist = None 
        self.y_predhist = None 
        self.idx = None 

    def get_gradient(self):
        return None 
    
    def get_weight(self):
        return None 
    
    def __call__ (self,y_train,y_pred) :
        self.y_trainhist = y_train 
        self.y_predhist = y_pred 
        self.idx = np.arange(len(y_train))

        loss = -np.log(y_pred[self.idx,y_train])
        return loss.mean()
    
    def backward(self):
        grad = self.y_predhist
        y_true = self.y_trainhist
        grad[self.idx,y_true] -= 1
        return grad / len(y_true)

    def update_weight(self, w):
        pass 

class MeanAbsoluteError (layers) :
    def __init__ (self) :
        self.ytrue_hits = None 
        self.y_pred_hits = None 
    
    def __call__ (self,y_true,y_pred) : 
        self.ytrue_hits = y_true
        self.y_pred_hits = y_pred
        return np.mean(np.power((y_pred - y_true),2))
    
    def backward(self):
        return (2 / len(self.ytrue_hits)) * (self.y_pred_hits - self.ytrue_hits)
    
    def get_weight(self):
        return None 
    
    def get_gradient(self):
        return None 
    
    def update_weight(self, w):
        pass 
    
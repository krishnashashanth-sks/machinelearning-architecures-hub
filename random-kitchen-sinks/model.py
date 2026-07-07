import numpy as np

class RandomKitchenSinks:
  def __init__(self,n_components=100,gamma=1.0):
    self.n_components=n_components
    self.gamma=gamma
    self.random_weights=None
    self.random_bias=None
  def fit(self,X,y=None):
    self.random_weights=np.random.normal(loc=0,scale=np.sqrt(2*self.gamma),size=(X.shape[1],self.n_components))
    self.random_bias=np.random.uniform(0,2*np.pi,size=(self.n_components,))
  def transform(self,X):
    projection=X @self.random_weights+self.random_bias
    return np.sqrt(2/self.n_components)*np.cos(projection)
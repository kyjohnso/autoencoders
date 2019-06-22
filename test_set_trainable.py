#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Dense, Input, Flatten, Reshape
from keras.losses import mse, mae, binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam

def settrainable(model, toset):
    for layer in model.layers:
        layer.trainable = toset
    model.trainable = toset

input_shape=(1024,1)
layers = 4
latent = 1024

inputs = Input(shape=input_shape)
x = Flatten()(inputs)

for ilayer in range(layers):
    x = Dense(latent,activation='relu')(x)
    
outputs = Reshape(input_shape)(x)
model1 = Model(inputs,outputs)
model1.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
model1.summary()

inputs2 = Input(shape=input_shape)
x = Flatten()(inputs2)

for ilayer in range(layers):
    x = Dense(latent,activation='relu')(x)
    
outputs2 = Reshape(input_shape)(x)
model2 = Model(inputs2,outputs2)
model2.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
model2.summary()

settrainable(model1,True)
settrainable(model2,False)
outputs3 = model2(model1(inputs))
model3 = Model(inputs,outputs3)
model3.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
model3.summary()


# In[2]:


assert (model3.layers[1].layers[2].get_weights()[0] == model1.layers[2].get_weights()[0]).all()
assert (model3.layers[2].layers[2].get_weights()[0] == model2.layers[2].get_weights()[0]).all()
wm1 = model3.layers[1].layers[2].get_weights()[0]
wm2 = model3.layers[2].layers[2].get_weights()[0]


# In[3]:


input_data = np.random.uniform(0,1,(10000,1024,1))
output_data = np.random.uniform(0,1,(10000,1024,1))


# In[4]:


model3.fit(input_data,output_data,epochs=2)


# In[17]:


assert (model3.layers[1].layers[2].get_weights()[0] == model1.layers[2].get_weights()[0]).all()
assert (model3.layers[2].layers[2].get_weights()[0] == model2.layers[2].get_weights()[0]).all()
assert not (model3.layers[1].layers[2].get_weights()[0] == wm1).all()
assert (model3.layers[2].layers[2].get_weights()[0] == wm2).all()


# In[19]:


model2.fit(input_data,output_data,epochs=2)


# In[20]:


assert (model3.layers[2].layers[2].get_weights()[0] == model2.layers[2].get_weights()[0]).all()
assert not (model3.layers[2].layers[2].get_weights()[0] == wm2).all()


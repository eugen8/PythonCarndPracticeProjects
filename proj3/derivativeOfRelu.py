
# coding: utf-8

# In[14]:

# Import needed libraries and other python stuff here.
# Show figures directly in the notebook.
# %matplotlib inline
import matplotlib.pyplot as plt # For plotting.
import numpy as np # To create matrices.


# In[15]:

# Here we define the ReLU function.
def f(x):
    """ReLU returns 1 if x>0, else 0."""
    return np.maximum(0,x)

# If we give ReLU a positive number, it returns the same positive number.
print( f(1))
print (f(3))


# In[16]:

X = np.arange(-4,5,1)
Y = f(X)


# In[18]:

plt.plot(X,Y,'o-')
plt.ylim(-1,5); plt.grid(); plt.xlabel('$x$', fontsize=22); plt.ylabel('$f(x)  $', fontsize=22)
# plt.show()


# In[ ]:



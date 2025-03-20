import adtoy
import numpy as np
# In[56]:


print("scaler multiply test")


# In[57]:


a = adtoy.tensor(2.0, True)


# In[58]:


b = adtoy.tensor(3.0, True)


# In[59]:


c = adtoy.tensor(4.0, True)


# In[60]:


d = a * b * c
e = a * b
f = e + d


# In[61]:


f.backward(1)


# In[62]:


print(a.grad)


# In[63]:


print(b.grad)


# In[64]:


print(c.grad)


# In[65]:


import torch
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = torch.tensor(4.0, requires_grad=True)
d = a * b * c
e = a * b
f = e + d
f.backward()
print(a.grad)   


# In[66]:


print(b.grad)


# In[67]:


print("arbitrary-dim matrix multiplication\add test")


# In[68]:


a = adtoy.tensor([[[[2.0, 3.0], [3.0, 4.0], [3.0, 4.0]],
           [[4.0, 6.0], [2.0, 7.0], [3.0, 4.0]],
           [[1.0, 2.0], [3.0, 5.0], [3.0, 4.0]]],
           [[[2.0, 3.0], [3.0, 4.0], [3.0, 4.0]],
           [[4.0, 6.0], [2.0, 7.0], [3.0, 4.0]],
           [[1.0, 2.0], [3.0, 5.0], [3.0, 4.0]]]], True)


# In[69]:


print(a.data.shape)


# In[70]:


b = adtoy.tensor(np.array([[1.0],[2.0]]), True)


# In[71]:


print(b.data.shape)


# In[72]:


c = a @ b


# In[73]:


print(c.data.shape)


# In[74]:


d = adtoy.tensor(np.array([[1.0, 2.0]]), True)
print(d.data.shape)


# In[75]:


e = d @ b
print(e.data.shape)


# In[76]:


f = c @ e + c
print(f.data.shape)


# In[77]:


f.backward(f.get_ones())


# In[78]:


print(a.grad)


# In[79]:


print(b.grad)


# In[80]:


print(c.grad)


# In[81]:


print(d.grad)


# In[82]:


print(e.grad)


# In[83]:


import torch
a = torch.tensor(
    [[[[2.0, 3.0], [3.0, 4.0], [3.0, 4.0]],
       [[4.0, 6.0], [2.0, 7.0], [3.0, 4.0]],
       [[1.0, 2.0], [3.0, 5.0], [3.0, 4.0]]],
       [[[2.0, 3.0], [3.0, 4.0], [3.0, 4.0]],
       [[4.0, 6.0], [2.0, 7.0], [3.0, 4.0]],
       [[1.0, 2.0], [3.0, 5.0], [3.0, 4.0]]]], requires_grad=True)
b = torch.tensor([[1.0],[2.0]], requires_grad=True)
c = a @ b
d = torch.tensor([[1.0, 2.0]], requires_grad=True)
e = d @ b
f = c @ e + c
a.retain_grad()
b.retain_grad()
c.retain_grad()
d.retain_grad()
e.retain_grad()
f.retain_grad()


# In[84]:


f.backward(gradient=torch.ones_like(f))


# In[85]:


print(a.grad)


# In[86]:


print(b.grad)


# In[87]:


print(c.grad)


# In[88]:


print(d.grad)


# In[89]:


print(e.grad)


# In[ ]:


print("matrix sclar multiplication and add test")


# In[135]:


a = adtoy.tensor(2.0, True)
b = adtoy.tensor(np.array([[1, 2],[3, 4]]), True)


# In[137]:


print(b.data)


# In[138]:


c = a * b


# In[139]:


print(c.data)


# In[140]:


c.backward(c.get_ones())


# In[141]:


print(a.grad)


# In[142]:


print(b.grad)


# In[146]:


a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor([[1.0, 2.0],[3.0, 4.0]], requires_grad=True)
c = a * b
a.retain_grad()
b.retain_grad()
c.retain_grad()
c.backward(gradient=torch.ones_like(c))


# In[147]:


print(a.grad)


# In[148]:


print(b.grad)


# In[ ]:


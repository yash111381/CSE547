import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[251]:


faces_data = pd.read_csv("data/faces.csv", header = None)


# In[252]:


print(faces_data.head())


# function pca_custom computes the eigen values and the eigen vectors of the provided matrix

# In[253]:


def pca_custom(matrix):
    sigma = 1/len(matrix)*(np.array(matrix.T).dot(np.array(matrix)))
    temp = np.linalg.eigh(sigma)
    eigVectorsMatrix = temp[1].T
    temp2 = sorted(zip(temp[0],eigVectorsMatrix), key = lambda x:x[0], reverse=True)
    eigVectors = []
    eigValues = []
    for e,v in temp2:
        eigValues.append(e)
        eigVectors.append(v)
    return (eigValues, eigVectors)


# Now, we compute the eigen vectors and eigenvalues of the faces_data that we read in from the input file

# In[254]:


eigValues, eigVectors = pca_custom(faces_data)


# **(1) The required eigenvalues are as follows:**

# In[255]:


for i in [0,1,9,29,49]:
    print(i+1,eigValues[i])


# In[256]:


sumOfEigvalues = np.sum(np.array(eigValues))
print(sumOfEigvalues)


# **(2) Plot of fractional reconstruction error vs. the number of iterations**

# In[257]:


errorValues = []
for k in range(1,50):
    errorValues.append(1 - np.sum(eigValues[:k])/sumOfEigvalues)
plt.title("Fractional reconstruction error vs. number of components")
plt.xlabel("No. of components")
plt.ylabel("Fractional reconstruction error")
_ = plt.plot(list(range(1,50)), errorValues)
plt.savefig("Reconstruction_error_plot.jpg")


# As seen from the above plot, the fractional reconstruction error decreases with increase in the number of components

# **(3)** First eigenvalue, eigenvector pair captures the maximum variance in the data. The first eigenvalue captures the variance in the projected direction, which captues the maximum variance in the data. Hence, since this eigen value is corresponding to the eigen vector capturing maximum variance in the data, it will be the largest

# In[258]:


plt.imshow(np.array(list(map(float, eigVectors[0]))).reshape(84,96).T)


# **(c) Visualization of Eigen-directions**

# **(1) Displaying the first 10 eigenvectors**

# In[259]:


plt.figure(figsize = (15,7))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(np.array(list(map(float, eigVectors[i]))).reshape(84,96).T)
plt.savefig("First_10_eigvectors.jpg")


# As seen from the above images, each eigen vector captures linear combination of the original features capturing the maximum variance
# 
# Here, as we can see the first eigen vector is a combination of the position sof the eyes, nose and mouth
# 
# The second eigen vector appears to capture the right side face features
# 
# The third eigen vector appears to capture the shape of the face
# 
# The fourth eigen vector appears to capture the position of the nose
# 
# The fifth eigen vector appears to capture the bottom face cut
# 
# The sixth eigen vector appears to capture the position of the eyebrows
# 
# The seventh eigen vector appears to capture the right side of the nose and forehead
# 
# The eighth eigen vector appears to capture the position of the eyebrows
# 
# The ninth eigen vector appears to capture the outline of the nose
# 
# The tenth eigen vector appears to capture the depth of the nose
# 

# **(2) Brief description of each eigenvalue**

# Each eigenvalue correspoding to the eigen vector indicates the strength of the eigen vector to capture the variance in the data. Here, the top 10 eigen vectors are visualized.
# * First eigen vector, eigenvalue pair captures the direction and the magnitude of the maximum variance in the data
# * Second eigen vector, eigenvalue pair captures the direction and the magnitude of the remaining maximum variance in the data with the current eigen vector orthogonal the previous eigen vector
# * And so on...
# 

# **(d) Visualization and reconstruction**

# In[260]:


required_images = [0,23,64,67,256]
required_table = [[0 for i in range(6)] for i in range(5)]
for i,v in enumerate(required_images):
    required_table[i][0] = np.array(faces_data.iloc[v])


# In[263]:


for j,k in enumerate([1,2,5,10,50]):
    eigVectorsHere = eigVectors[:k]
    reconstructed_sigma = np.array(eigVectorsHere).T.dot(np.array(eigVectorsHere))
    for i,v in enumerate(required_images):
        required_table[i][j+1] = np.array(faces_data.iloc[v]).dot(reconstructed_sigma)


# In[264]:


plt.figure(figsize = (18,17))
for i in range(len(required_table)):
    for j in range(len(required_table[i])):
        plt.subplot(5,6,6*i+j+1)
        plt.imshow(np.array(list(map(float, required_table[i][j]))).reshape(84,96).T, vmin=0, vmax=1)
plt.savefig("face_reconstructed.jpg")


# **(2) Brief interpretation of the reconstructions**

# As visible from the reconstruction of the required images, each row represents one image. The first column represents the original image, the second represents the one reconstructed with only 1 principal component, and the subsequent ones represents the images reconstructed with 2,5,10 and 50 pricipal components.
# 
# * As visible from the images. The first column doesn't reconstruct the original image in the best way. But, as we go on increasing the number of principal components, the reconstructed image gets closer and closer to the original image. The images constructed with 50 principal components resemles quite a lot to the original image and doesn't lose much of the information as in the original image

# In[ ]:





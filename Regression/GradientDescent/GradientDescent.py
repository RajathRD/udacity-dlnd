import numpy as np
import matplotlib.pyplot as plt

N = 100
num_w = 4 #6
# Gradient Descent Algorithm
def gradient_desc(x,w,y):
    rate = 0.000001 # 9+1
    augmented_x = np.array([x**0])
    for i in range(1,num_w+1):
        augmented_x = np.vstack([augmented_x, x**i ])
    iterations = 1000000 #6

    for i in range(0,iterations):
        ybar = np.dot(w,augmented_x)
        w = w - ( rate * ( np.dot( ( ybar - y ),augmented_x.T ) )/ (2*N) )
    return w
# Generate Data
samples = np.random.normal(0, 0.01, N)
x = np.linspace(0,2*np.pi,N)

#y = np.sin(x/3) + np.cos(2*x)
y = 3*x**2 + 2*x**3 + samples

w = np.ones(num_w+1)
weights = gradient_desc(x,w,y)
print (weights)

augmented_x = np.array([x**0])
for i in range(1,num_w+1):
    augmented_x = np.vstack([augmented_x, x**i ])

print ("Weights: ",weights) # actual weights

prediction = np.dot(weights,augmented_x)
print ("Prediction: ",prediction[0]) # to check if NaN
plt.plot (x,y,x,prediction)
plt.show()

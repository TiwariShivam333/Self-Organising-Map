import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#Functions for a single neuron
class singleneuron(object):
    def __init__(self,xx,yy,count):
        self.x=xx
        self.y=yy
        self.w=np.random.normal(scale=0.1,size=count)


    #Return coordinates of corresponding neuron
    def coordinates(self):
        return (self.x,self.y)

    #Return weight value of corresponding neuron
    def wt(self):
        return self.w

    #Weight update of corresponding neuron
    #Using W(t) = W(t) + α(t, T )h(t)(X − W(t))
    #h(t)=theta=Neighborhood function
    def weightupdate(self,lr,theta,inputv):
        self.w=self.w+theta*lr*(inputv-self.w)


#Dataset reading
data = pd.read_csv('Iris.csv')
#print(data.head())
data=data.drop('Id',axis=1) #Deleting 1st column which contains line number
"""print(ndata.head())
ndata.plot(kind='box', subplots=True)
plt.show()
"""
arr=data.values
X=arr[:,:-1] #Taking training data i.e. data ignoring labels
#Y=array[:,-1]
#print(X)
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y)


#Defining Feature Map
#Map(4,4) ====>> 16 clusters (assumption as unsupervised)
shapeh=4
shapew=4
inputd=4 #No. of features of dataset
#defining coordinates and weight for each neuron or each node of feature map
maparr=[singleneuron(i,j,inputd) for i in range(shapeh) for j in range(shapew)]

#Min-Max Normalization of data
for i in range(len(X)):
    x=X[i]
    X[i] = (x-min(x))/(max(x)-min(x))


#Euclidean distance between two data points
def euclid_dist(i,inputv):
    return np.sqrt(sum((i-inputv)**2))


#For finding Best MAtching Unit
def findbmu(inputv):
    dists=[euclid_dist(i.wt(),inputv) for i in maparr]
    #print(dists)
    bmu=np.argmin(dists)
    return maparr[bmu]

#bmu=findbmu(X[2]]

#Checking if a node is in neighborhood of Best MAtching Unit
def innbrhood(i,r,bmux,bmuy):
    x,y=i.coordinates()
    if (x-bmux)**2 + (y-bmuy)**2 <= r**2:
        return 1
    else:
        return 0


#Finding nodes coordinates that are in neighborhood of Best MAtching Unit
def bmu_nbrhood(bmu,r):
    bmux,bmuy=bmu.coordinates()
    return[i for i in maparr if innbrhood(i,r,bmux,bmuy)]



def train():
    epochs=1500
    lr=0.07 #Learning rate
    r=shapeh/2 #Initial neighborhood radius
    time_const = epochs / np.log(r)
    start=time.time()
    for iteration in range(epochs):
        it=time.time()

        #Selecting random datapoint
        random_data=np.random.choice(len(X)-1,1)[0]
        inputv=X[random_data]
        bm=findbmu(inputv)
                #print(bm)
                #print('\n')
        for i in bmu_nbrhood(bm,r):
            distan=euclid_dist(i.wt(),inputv)
            theta=np.exp(-(distan**2)/(2*r**2)) #Neighborhood function same as h(t) = exp(−d/2σ^2)
            i.weightupdate(lr,theta,inputv)
        lr = lr* np.exp(-iteration / epochs)  #Updating learning rate (Decreasing exponentially)
        if iteration < 1100:
            r=r * np.exp(-iteration / time_const) #Neighborhood radius decay
        et=time.time()
        print('Iteration: '+str(iteration)+'  Time required: '+str(et-it))
    end=time.time()
    print('Total time:  '+str(end-start))


train()


#For calculating final output
#Final output file contains valid clusters and corresponding data point indexes ('None' in feature map denotes invalid cluster)
shape=(shapeh,shapew)
indexes=np.empty(shape,dtype=tuple)
#result=np.empty(shape,dtype=tuple)
res=np.empty(shapeh*shapew,dtype=tuple)
result=[]
c=0
cl=0
for x in X:
    c=c+1
    bmu=findbmu(x)
    x, y = bmu.coordinates()
    if not indexes[x][y]:
        indexes[x][y]=[x,y]
        cl=cl+1
        res[int(x*shapeh+y)]=cl
    result.append(str(res[x*shapew+y]))
for x in range(shapeh):
    for y in range(shapew):
        print(indexes[x][y]) #Printing coordinates of valid clusters in map
thefile = open('resultsomiris.txt', 'w')
for item in result:
  thefile.write("%s\n" % item)

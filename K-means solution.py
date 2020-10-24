import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def initializeMeans(df):
    """
    Take two random points as the means
    """
    IDXmean1 = random.randint(1,len(df))
    IDXmean2 = random.randint(1,len(df))

    mean1 = df.iloc[IDXmean1]
    mean2 = df.iloc[IDXmean2]
    
    return mean1,mean2
    

def euclideanDist(df,pointIDX,mean1,mean2):
    """
    Take the index of the point in the dataframe you want to calculate the 
    distance from and calculate the euclidean distance to both means.
    
    return the dataframe with the closest mean assigned to column 'class'
    """
    point = df.iloc[pointIDX][['eruptions','waiting']].values
    mean1 = mean1[['eruptions','waiting']].values
    mean2 = mean2[['eruptions','waiting']].values
    
    distTo1 = sum([a-b for a,b in zip(point,mean1)])**2 # np.linalg.norm()
    distTo2 = sum([a-b for a,b in zip(point,mean2)])**2
        
    if distTo1 < distTo2:
        df.loc[pointIDX,'class'] = 1
    else:
        df.loc[pointIDX,'class'] = 2
    
    return df

def updateMean(df):
    """
    df is the dataframe of points containing the assigned classes
    return updated mean1 and mean2
    """
    dfClass1 = df.loc[df['class'] == 1]
    dfClass2 = df.loc[df['class'] == 2]
    
    mean1 = dfClass1[['eruptions','waiting']].mean()
    mean2 = dfClass2[['eruptions','waiting']].mean()
    
    return mean1,mean2
    
def Kmeans(df,iterations):
    
    prevPredictedClasses = list() # Initialize a container for predicted classes for early stopping
    mean1,mean2 = initializeMeans(df)
        
    plt.title('Init')

    plt.scatter(df.loc[df['class'] == 1]['eruptions'],df.loc[df['class'] == 1]['waiting'],color='g',label='1')
    plt.scatter(df.loc[df['class'] == 2]['eruptions'],df.loc[df['class'] == 2]['waiting'],color='b',label='2')

    plt.scatter(mean1['eruptions'],mean1['waiting'],s=70,label='mean1',marker='s',color='r')
    plt.scatter(mean2['eruptions'],mean2['waiting'],s=70,label='mean2',marker='s',color='y')
    
    plt.legend()
    plt.show()
    
    c = 1 # Counter for plotting
    while prevPredictedClasses != list(df['class'].values):
        prevPredictedClasses = list(df['class'].values)
        
        for i in range(len(df)):
            df = euclideanDist(df,i,mean1,mean2)
            
        mean1,mean2 = updateMean(df)
        
        plt.title('Iteration = {}'.format(c))
        
        plt.scatter(df.loc[df['class'] == 1]['eruptions'],df.loc[df['class'] == 1]['waiting'],color='g',label='1')
        plt.scatter(df.loc[df['class'] == 2]['eruptions'],df.loc[df['class'] == 2]['waiting'],color='b',label='2')
        
        plt.scatter(mean1['eruptions'],mean1['waiting'],s=70,label='mean1',marker='s',color='r')
        plt.scatter(mean2['eruptions'],mean2['waiting'],s=70,label='mean2',marker='s',color='y')
        
        plt.legend()
        plt.show()
        
        c += 1 # increment count
    
    return df,mean1,mean2

def predict(point,mean1,mean2):
    mean1 = mean1[['eruptions','waiting']].values
    mean2 = mean2[['eruptions','waiting']].values

    distTo1 = sum([a-b for a,b in zip(point,mean1)])**2
    distTo2 = sum([a-b for a,b in zip(point,mean2)])**2

    if distTo1 < distTo2:
        print("This point is likely to be of class 1")
    else:
        print("This point is likely to be of class 2")


df = pd.read_csv('geyserData.csv')
df = df.reset_index(drop=True)
df['class'] = 1
df['eruptions'] = df['eruptions'] / max(df['eruptions']) 
df['waiting'] = df['waiting'] / max(df['waiting'])
df,mean1,mean2 = Kmeans(df,5)

# Predict
point = np.array([0.3,0.5])
predict(point,mean1,mean2)
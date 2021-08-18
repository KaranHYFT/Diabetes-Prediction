#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def findMean(data):
    rows = data.shape[0]
    cols = data.shape[1]
    dataTobeProceessed = data.copy()
    # transpose data 
    transposedData = dataTobeProceessed.transpose()
    finalSumList = []
    countlst =[]
    # finding data without 0 Value
    for i in range (len(transposedData)):
        total= 0
        count = 0
        for j in range (len(transposedData[0])):
            if transposedData[i][j]!=0:
                total= total+ transposedData[i][j]
                count = count + 1
        finalSumList.append(float(total))
        if(count==0):
            count = 1
        countlst.append(float(count))    
    meanVal = [i / j for i, j in zip(finalSumList, countlst)] 
    inds = np.where(data == 0) 
    final_data = data.copy()
    final_data[inds] = np.take(meanVal, inds[1])
    
    final =  np.array(final_data).tolist()
    return final

class NeuralNetwork:
    def __init__(self,x,y,learnrate,epoch):
        self.input      = x                 
        self.y          = y
        self.learnrate  = learnrate
        self.epoch      = epoch
        self.weights1   = np.random.rand(self.input.shape[1],6) 
        self.weights2   = np.random.rand(6,1)
        self.outcome    = np.zeros(self.y.shape)
        self.costDataList   = []

    # Calculating sigmoid
    @staticmethod    
    def sigmoidVal(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    # Calculating Derivative of Sigmoid x(1-x)
    @staticmethod    
    def sigmoidDerivative(x): return x * (1 - x)      
    
    # Calculating cost 
    @staticmethod
    def findVals(targetData,outputData):
        return 0.5*np.sum(np.square(np.subtract(targetData,outputData)))
    
    # Apply Sigmoid on Layer and output
    def forwardMovement(self):
        self.HiddenLayer = self.sigmoidVal(np.dot(self.input, self.weights1))
        self.outcome = self.sigmoidVal(np.dot(self.HiddenLayer, self.weights2))

    def backpropogation(self):
        # derivative of the cost function 
        WeightVal2 = np.dot(self.HiddenLayer.T, ((self.y - self.outcome) * self.sigmoidDerivative(self.outcome)))
        WeightVal1 = np.dot(self.input.T,  (np.dot((self.y - self.outcome) * self.sigmoidDerivative(self.outcome), self.weights2.T) * self.sigmoidDerivative(self.HiddenLayer)))

        # Updating weights
        self.weights1 += WeightVal1 * self.learnrate
        self.weights2 += WeightVal2 * self.learnrate     
        
    # Training in data set    
    def dataTraining(self):
        for i in range (self.epoch):
            self.forwardMovement()
            self.backpropogation()
            self.costDataList.append(self.findVals(self.y,self.outcome))

    # Testing in data set        
    def dataPrediction(self,inputData):
        self.input=inputData
        self.forwardMovement()
        return self.outcome
                     

if __name__ == "__main__":

    #Reading CSV file of Training
    data = pd.read_csv("diabetes.csv",header=0)
    Trainingdata=[]
    ClassValue=[]

    #adding data to List for features and its associated output(Whether Diabetic or non-Diabetic)
    for i in range(0,len(data)):
        Trainingdata.append([data.values[i,0],data.values[i,1],data.values[i,2],data.values[i,3],data.values[i,4],data.values[i,5],data.values[i,6],data.values[i,7]])
        if data.values[i,8]==0:
            ClassValue.append([0])
        elif data.values[i,8]==1:
            ClassValue.append([1])
    Trainingdata = findMean(np.array(Trainingdata))        
    #Normalising using sklearn
    scalerData = StandardScaler()
    scalerData.fit(Trainingdata)
    scalerTrainingData=scalerData.transform(Trainingdata)
    

    network = NeuralNetwork(x=scalerTrainingData,y=np.array(ClassValue),learnrate=0.3,epoch=2000)
    #Data Training on Training Data
    network.dataTraining()

    # Reading data for Testing
    testdata = pd.read_csv("diabetes_test.csv",header=0)
    testData=[]

    #Normalising the inputs of test data
    for i in range(0,len(testdata)):
        testData.append([testdata.values[i,0],testdata.values[i,1],testdata.values[i,2],testdata.values[i,3],testdata.values[i,4],testdata.values[i,5],testdata.values[i,6],testdata.values[i,7]])
    
    testData = findMean(np.array(testData))        
    # Data Normalizing  
    xtest=scalerData.transform(testData)
    out=network.dataPrediction(xtest)
    
    ##Labeling the test data based on the output of the Neuralnet
    diabeticType=[]
    print("Output :")
    print("\nPregnancies  Glucose  BloodPressure  SkinThickness  Insulin  BMI  DiabetesPedigreeFunction  Age  Output  DiabetesReport")
    for i in range(0,len(xtest)):
        if(out[i]<0.5):
            #diabeticType.append('Non_Diabetic')
            diabeticType.append('0')
        else:
            #diabeticType.append('Diabetic')
            diabeticType.append('1')
        print(testData[i][0],"    ",testData[i][1],"    ",testData[i][2],"    ",testData[i][3],"    ",testData[i][4],"    ",testData[i][5],"    ",testData[i][6],"    ",testData[i][7],"    ",out[i],"    ",diabeticType[i])


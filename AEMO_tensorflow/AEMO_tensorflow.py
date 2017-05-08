### https://www.tensorflow.org/get_started/input_fn


import pandas as pd
import numpy as  np
import time
import tensorflow as tf
import tensorflow.contrib.learn as lr
import itertools


start_time = time.time()


xls_file = pd.ExcelFile(r'C:\Users\T\OneDrive\My Research\Tamer_Ehab_STLF\Programs For Local Machine\AEMO1_data.xlsx')
df = xls_file.parse('AEMO1_data')
#~~~~~~~~~~~~~~ Normalization from (-1 to 1) ~~~~~~~~~~~~
#OldData=df.loc[df['month'].isin([1,2,3,4,5,6,7,8,9,10,11,12])].values[:,7];
OldData=df['load'].values
MaxLoad  =np.amax(OldData,axis=0)
MinLoad  =np.amin(OldData,axis=0)   
OldData[:] = (2 *  (OldData[:] -MinLoad) /(MaxLoad-MinLoad))  -1

#~~~~~~~~~~~~~~~~Phase space~~~~~~~~~~~~~~~~~~~~~
N=len(OldData);
dim=7;   # dimenition 
tau=1   # time delay
f=(dim-1)*tau
T=N-f;
T=T-1 #***************
newData= np.zeros((T,dim+2)); # the two addition colunms is for id and Y 
for i in range(T):
    newData[i,0]=i+1;#   to construct an "id" column
    for j in np.arange(1,dim+1):
        newData[i,j]=OldData[i+j*tau];  # Phase Space for load
       
 
for i in range(T-1): 
    newData[i,dim +1]=newData[i+1,dim]; # predicted value

    

columnsNames= columnsNames=  ['id'] + [ 'x'+str(i) for i in range(1, dim+1)] + ['Y']
newDataFrame=pd.DataFrame(newData,columns =columnsNames);


TestSize=31
Train = newDataFrame[0:T-TestSize-2].drop('id',1)
Test=newDataFrame[T-TestSize-1:T-1].drop('id',1)
writer = pd.ExcelWriter('D:\\output.xlsx')
newDataFrame.to_excel(writer,'Sheet1')
#~~~~~~~~~~~~~~~~~~~~~~~ tensorFlow ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FEATURES=columnsNames[1:7]
LABEL="Y"

regressor = lr.DNNRegressor(
    feature_columns=[tf.contrib.layers.real_valued_column(k) for k in FEATURES],
    hidden_units=[1024, 512, 256] , model_dir=r"C:\Users\T\AppData\Local\Temp\tmphju2fe9s\model.ckpt" ) #1024, 512, 256 # model_dir="/tmp/The_model"

# Input builders
def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels




regressor.fit(input_fn=lambda: input_fn(Train), steps=400)


#ev = regressor.evaluate(input_fn=lambda: input_fn(Test), steps=1)
#loss_score = ev["loss"]
#print("Loss: {0:f}".format(loss_score))

# Print out predictions
y_real=Test['Y']
#Test['Y']=0 # to ensure that Y not included in prediction process
y = regressor.predict(input_fn=lambda: input_fn(Test))
# .predict() returns an iterator; convert to a list and print predictions
predictions = list(y)
print("Predictions: {}".format(str(predictions)))


##### summary #####

result=pd.DataFrame({'y_real':y_real, 'y_predicted':predictions})
res=result.values
res[:,0]= (res[:,0] +1)*((MaxLoad-MinLoad)/2) + MinLoad
res[:,1]= (res[:,1] +1)*((MaxLoad-MinLoad)/2) + MinLoad

result
MAPE= np.mean(abs(res[:,0]-res[:,1])/res[:,0])*100
print({'MAPE =':MAPE})
print("--- %s seconds ---" % (time.time() - start_time))
 
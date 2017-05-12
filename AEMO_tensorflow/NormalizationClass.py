import numpy as  np

class NormalizationClass(object):

    def Normalize(self , Un_array):
        Un_array=np.asarray(Un_array)
        global MaxValues
        global MinValues
        
        MaxValues=np.amax(Un_array,axis=0);
        MinValues=np.amin(Un_array,axis=0);
        
        rows , cols= Un_array.shape;
        
        for j in np.arange(cols):
            N_array[:,j] = (2 *  (Un_array[:,j] -MinValues[j]) /(MaxValues[j]-MinValues[j]))  -1
            
        
        return N_array


    def DeNormalize(self , N_array, ref_Col_num):
        rows , cols= N_array.shape;
        for j in np.arange(cols):
            Un_array[:,j]= (N_array[:,j] +1)*((MaxValues[j]-MinValues[j])/2) + MinValues[j]

    
        
        





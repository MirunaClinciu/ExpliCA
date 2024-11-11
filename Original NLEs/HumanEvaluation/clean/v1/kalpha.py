import pandas as pd

import krippendorff


Data1 = pd.read_csv('informativeness.csv') 
Data2 = pd.read_csv('clarity.csv') 
Data3 = pd.read_csv('effectiveness.csv') 


Data1 = Data1.dropna(axis = 0, how = 'all')
Data2 = Data2.dropna(axis = 0, how = 'all')
Data3 = Data3.dropna(axis = 0, how = 'all')


Data1  = Data1[Data1 .std(1)!=0]
Data2  = Data2[Data2 .std(1)!=0]
Data3  = Data3[Data3 .std(1)!=0]



low = .05
high = .95
quant_df = Data1.quantile([low, high])
Data1 = Data1.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
                                    (x < quant_df.loc[high,x.name])], axis=0)


low = .05
high = .95
quant_df = Data2.quantile([low, high])
Data2 = Data2.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
                                    (x < quant_df.loc[high,x.name])], axis=0)

low = .05
high = .95
quant_df = Data3.quantile([low, high])
Data3 = Data3.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
                                    (x < quant_df.loc[high,x.name])], axis=0)


print(krippendorff.alpha(reliability_data=Data1))
print(krippendorff.alpha(reliability_data=Data2))
print(krippendorff.alpha(reliability_data=Data3))


#To find the Resistance percentage using the eucliedean distance:
import Utilities as Ult
filePath = "E:\Project_works\Dataset\ProjectDataset.csv" colordf = Ult.getDataFrame(filePath)
Euclidist = colordf['Euclidean Distance']
Euclidistlist = colordf['Euclidean Distance'].values.tolist()
Heatpctlist = list()
maxeuclidist = max(Euclidistlist)
Heatpctlist = Ult.findHeatResistancePercentage(Euclidistlist,maxeuclidist)
colordf['Heat Resistance Percentage'] = Heatpctlist
Ult.saveDataFrame(colordf,filePath)

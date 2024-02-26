import Utilities as Ult
#Variable Declaration:
variableDict = Ult.getInputsForColorIdentification()
ImgPath = variableDict['ImgPath']
Trained_Model_Path =variableDict['Trained_Model_Path']
ThrshForCrop = variableDict['ThrshForCrop']
IntrstImgListReg = variableDict['IntrstImgListReg']
maxeuclidist = variableDict['maxeuclidist']
Trained_Cluster_Model_Path = variableDict['Trained_Cluster_Model_Path']

#Function Calls:
img = Ult.readImage(ImgPath)
Trained_Classifier_Model = Ult.getTrainedModel(Trained_Model_Path)
Trained_Cluster_Model = Ult.getTrainedModel(Trained_Cluster_Model_Path)
resizedImg = Ult.ResizeImg(img)
Ult.CropImgandSave(resizedImg,ThrshForCrop)
MostusedColorList = Ult.getDominantColor(IntrstImgListReg)
#Euclidist = Ult.EuclideanDist(MostusedColorList)
#Heatpctlist = Ult.findHeatResistancePercentage(Euclidist,maxeuclidist)
#colorforRegion = Ult.getColorForRegion(Trained_Classifier_Model,Euclidist,Heatpctlist,IntrstImgListReg)
#clustersList = Ult.findClusters(Trained_Cluster_Model,Euclidist,Heatpctlist)
print(MostusedColorList)

import cv2
from colorthief import ColorThief
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
def getInputsForColorIdentification():
inputDict = dict()
ImgPath = "E:\Project_works\Images\Wall_12.jpg" ThrshForCrop = 70
IntrstImgListReg = ['BottomLeft.jpg', 'BottomRight.jpg', 'Center.jpg', 'TopLeft.jpg',
'TopRight.jpg']
Trained_Model_Path = 'E:\Project_works\Dataset\ColorPrediction.sav' Trained_Cluster_Model_Path = 'E:\Project_works\Dataset\ColorClustering.sav' colordf = pd.read_csv("E:\Project_works\Dataset\ProjectDataset.csv")
maxeuclidist = max(colordf['Euclidean Distance'].values.tolist())
inputDict['ImgPath'] = ImgPath
inputDict['ThrshForCrop'] = ThrshForCrop
inputDict['IntrstImgListReg'] = IntrstImgListReg
inputDict['Trained_Model_Path'] = Trained_Model_Path
inputDict['Trained_Cluster_Model_Path'] = Trained_Cluster_Model_Path
inputDict['colordf'] = colordf
inputDict['maxeuclidist'] = maxeuclidist
return inputDict
def readImage(imagePath):
img = cv2.imread(imagePath)
return img
def getDataFrame(filePath):
df = pd.read_csv(filePath)
return df
def saveDataFrame(dataFrame,filePath):
dataFrame.to_csv(filePath, index=False)
def getTrainedModel(trainedModelPath):
Trained_Model = pickle.load(open(trainedModelPath, 'rb'))
return Trained_Model
def ResizeImg(inpimg):
height,width = inpimg.shape[:2]
newwidth = (width*3)//4
newheight = (height*3)//4
newDmn = (newwidth,newheight)
resizedImg = cv2.resize(inpimg,newDmn)
return resizedImg
def CropImgandSave(inpimg,threshold):
TopRight = inpimg[0:threshold,-threshold:]
TopLeft = inpimg[0:threshold,0:threshold]
BottomRight = inpimg[-threshold:,-threshold:]
BottomLeft = inpimg[-threshold:,0:threshold]
height,width = inpimg.shape[:2]
halfht = height//2
halfwd = width//2
Center = inpimg[halfht:halfht+threshold:,halfwd:halfwd+threshold:]
cv2.imwrite("E:\Project_works\CroppedImage\TopLeft.jpg",TopLeft)
cv2.imwrite("E:\Project_works\CroppedImage\BottomLeft.jpg", BottomLeft)
cv2.imwrite("E:\Project_works\CroppedImage\BottomRight.jpg", BottomRight)
cv2.imwrite("E:\Project_works\CroppedImage\TopRight.jpg", TopRight)
cv2.imwrite("E:\Project_works\CroppedImage\Center.jpg", Center)
def getDominantColor(imgPathList):
dom_Color_list = list()
prefixImgPath = 'E:\Project_works\CroppedImage/' for imgpath in imgPathList:
colorTheif = ColorThief(prefixImgPath+imgpath)
dom_Color = colorTheif.get_color(quality=1)
dom_Color_list.append(dom_Color)
return dom_Color_list
def getColorForRegion(trainedMdl,euclidist,Heatpctlist,regofintimglist):
ColorsforRegions = dict()
for index in range(len(regofintimglist)):
valueslist = [euclidist[index],Heatpctlist[index]]
predColor = trainedMdl.predict([valueslist]).item()
currentRegion = regofintimglist[index].replace('.jpg','')
ColorsforRegions[currentRegion] = predColor
return ColorsforRegions
def EuclideanDist(MostusedColorList):
euclideanList = list()
whiteColor = np.array((255,255,255))
j = 0
for i in range(len(MostusedColorList)):
red = MostusedColorList[i][j]
green = MostusedColorList[i][j+1]
blue = MostusedColorList[i][j+2]
othercolor = np.array((red,green,blue))
dist = np.linalg.norm(whiteColor-othercolor)
euclideanList.append(dist)
return euclideanList
def findHeatResistancePercentage(Euclidistlist,maxeuclidist):
Heatpctlist = list()
for dist in Euclidistlist:
percentage = (dist / maxeuclidist) * 100
heatpercentage = 100 - percentage
Heatpctlist.append(heatpercentage)
return Heatpctlist
def saveTrainedModel(csvFile,filePath):
dataframe = pd.read_csv(csvFile)
x = dataframe[['Euclidean Distance','Heat Percentage']]
y = dataframe['Color']
rfc = RandomForestClassifier(random_state=0,max_depth=2)
rfc.fit(x,y)
pickle.dump(rfc,open(filePath,'wb'))
def convertHexToRGB(hexaValue):
value = hexaValue.lstrip('#')
lv = len(value)
rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
return rgb[0], rgb[1], rgb [2]
def getDerivedDataSet(baseDatasetPath):
color_df = pd.read_excel()
rgbDict = dict()
redList = list()
blueList = list()
greenList = list()
euclideanList = list()
hexCodeList = color_df['#RRGGBB (Hex Code)'].values.tolist()
colorsList = color_df['Name'].values.tolist()
for hexCode in hexCodeList:
red, green, blue = convertHexToRGB(hexCode)
whiteColor = np.array((255, 255, 255))
otherColors = np.array((red, green, blue))
dist = np.linalg.norm(whiteColor - otherColors)
redList.append(red)
blueList.append(blue)
greenList.append(green)
euclideanList.append(dist)
rgbDict['Color'] = colorsList
rgbDict['Hex Code'] = hexCodeList
rgbDict['Red'] = redList
rgbDict['Green'] = greenList
rgbDict['Blue'] = blueList
rgbDict['Euclidean Distance'] = euclideanList
rgbDF = pd.DataFrame.from_dict(rgbDict)
rgbDF.sort_values("Euclidean Distance", inplace=True)
return rgbDF
def saveClusterLabels(dataPath,numberOfClusters,modelPath):
df = getDataFrame(dataPath)
X = df[['Euclidean Distance', 'Heat Resistance Percentage']]
kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(X)
clusterLabels = list(kmeans.labels_)
df['Cluster Label'] = clusterLabels
saveDataFrame(df, dataPath)
pickle.dump(kmeans, open(modelPath, 'wb'))
def findClusters(Trained_Cluster_Model,euclidist,Heatpctlist):
clustersforRegions = list()
for index in range(len(euclidist)):
valueslist = [euclidist[index], Heatpctlist[index]]
predCluster = Trained_Cluster_Model.predict([valueslist]).item()
clustersforRegions.append(predCluster)
return clustersforRegions

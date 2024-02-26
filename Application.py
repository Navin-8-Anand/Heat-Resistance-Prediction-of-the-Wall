from flask import Flask,render_template, request
import Utilities as Ult
app = Flask('A Small Try')
@app.route('/output',methods=['POST'])
def getOutput():
if request.method == 'POST':
ImgPath = request.files['filename']
inputImagePath = 'static/InputImage.jpg' ImgPath.save(inputImagePath)
# Variable Declaration:
variableDict = Ult.getInputsForColorIdentification()
Trained_Model_Path = variableDict['Trained_Model_Path']
ThrshForCrop = variableDict['ThrshForCrop']
IntrstImgListReg = variableDict['IntrstImgListReg']
maxeuclidist = variableDict['maxeuclidist']
Trained_Cluster_Model_Path = variableDict['Trained_Cluster_Model_Path']
# Function Calls:
img = Ult.readImage(inputImagePath)
Trained_Model = Ult.getTrainedModel(Trained_Model_Path)
Trained_Cluster_Model = Ult.getTrainedModel(Trained_Cluster_Model_Path)
resizedImg = Ult.ResizeImg(img)
Ult.CropImgandSave(resizedImg, ThrshForCrop)
MostusedColorList = Ult.getDominantColor(IntrstImgListReg)
Euclidist = Ult.EuclideanDist(MostusedColorList)
Heatpctlist = Ult.findHeatResistancePercentage(Euclidist, maxeuclidist)
colorforRegion = Ult.getColorForRegion(Trained_Model, Euclidist, Heatpctlist, IntrstImgListReg)
clustersList = Ult.findClusters(Trained_Cluster_Model, Euclidist, Heatpctlist)
keyIndex = 0
finalList = list()
for key in colorforRegion.keys():
currentList = [key]
currentList.append(IntrstImgListReg[keyIndex].split(".")[0])
currentList.append(colorforRegion[key])
currentList.append(Heatpctlist[keyIndex])
currentList.append(clustersList[keyIndex])
#currentList.append(healthPctList[keyIndex])
keyIndex = keyIndex + 1
finalList.append(currentList)
print(finalList)
return render_template("outputPage.html",result = finalList)
return render_template("inputPage.html")
@app.route('/')
def hello():
return render_template("inputPage.html")
app.run(debug=True)

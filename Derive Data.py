import Utilities as Ult
baseDataPath = "E:\Project_works\Dataset\Colors.xlsx" derivedDataPath = "E:\Project_works\Dataset\ProjectDataset.csv" derivedDf = Ult.getDerivedDataSet(baseDataPath)
Ult.saveDataFrame(derivedDf,derivedDataPath)

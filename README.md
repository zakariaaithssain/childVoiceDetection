# ADULT/CHILD VOICE DETECTOR

### steps: 
1 - build dataset, used CommonVoice ASR dataset to try to  create a dataset for child/adult classification, although children presentation in this dataset is minor, and data is not dedicated for such task.  
2 - compare feature selection methods to finally use RFECV to go from 60 acoustic features to only 26 with improved ROC-AUC.   
3 - Randomized CV Search to fine-tune an XGBoost Classifier hyperparams.  
4 - train and test the model.  
5 - fine tune prob threshold.  
6 - calibrate probs and train over all the dataset.  

### raw audios:  
CommonVoice dataset can also be found in this kaggle URL:  
https://www.kaggle.com/datasets/mozillaorg/common-voice 
### run with python: 
```bash 
python -m venv .venv 
source .venv/bin/activate 
python -m pip install -r requirements.txt
streamlit run app.py 
```
### or with docker: 
```bash 
docker build -t detector . 
#then 
docker run detector 
```

### metrics:  
metrics for each step are available in the notebooks 

### final conclusion:  

the task of child/adult detection requires deeper knowledge of audio suitable features, dedicated datasets, and can't be perfectly done with this project circumstances, especially distinguishing children/women voices. 

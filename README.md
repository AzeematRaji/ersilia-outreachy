## Bioavailability Prediction Model
A ML model that predicts whether a drug molecule is bioavailable or non-bioavailable using XGBoost and Ersilia compound embeddings.

### Table of Contents
1. [Project Description](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#project-description)
1. [Setting the Environment](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#setting-the-environment)
2. [Project Structure](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#project-structure)
3. [Download a dataset](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#download-a-dataset)
4. [Featurise the data](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#featurising-the-data)
5. [Training the model](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#training-the-model)
6. [Model Evaluation](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#evaluating-the-model)
8. [Model Summary](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#model-summary)
9. [Using the Model Later](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#using-the-model-later)
11. [Extra Model Validation](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#extra-model-validation)
12. [Apply Model to Public Dataset](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#apply-models-to-public-dataset)
13. [Conclusion](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#conclusion)

### Project Description

This project aims to classify drug molecules based on their oral bioavailability which is a key property in drug development. The workflow leverages the Ersilia Model for generating compound embeddings as molecular features, and trains an XGBoost classifier for prediction.

### Setting the Environment:

#### Prerequisites
- linux OS
- gcc compiler, `sudo apt install build-essential`
- python 3.8 and above, [install here](https://www.python.org/).
- conda installed, use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Installer
- git installed, [install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- github cli installed, [Github CLI](https://cli.github.com/)
- gitlfs installed and activated, [Git LFS](https://git-lfs.github.com/)
- docker installed, [Docker](https://docs.docker.com/engine/install/)

#### Installing Ersilia

All the prerequisites must be satisfied. 

create a conda environment:
  
`conda create -n ersilia python=3.12`

activate the environment:
  
`conda activate ersilia`

clone ersilia repo:
```
git clone https://github.com/ersilia-os/ersilia.git
cd ersilia
```

install with pip:

`pip install -e .`

confirm Erisilia:

`ersilia --help`

### Project Structure

``` text
├── data/                      
│   ├── bioavailability.csv
│   └── featurised_bioavailability.csv
├── models/                       
│   ├── bioavailability.pkl
│   ├── random_forest_bioavailability.pkl
│   └── svm_bioavailability.pkl
├── notebooks/                     
│   ├── bio_data_handling.ipynb
│   ├── bioavailability_train.ipynb
│   └── model_validation.ipynb
├── results/                    
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   └── roc_curve.png
├── scripts/                     
│   ├── bio_data_featurising.py
│   ├── bio_data_handling.py
│   └── bioavailability.py
└── README.md
```


### Download a dataset

#### Background of Data

__Dataset__: _Bioavailability, Ma et al._

Oral bioavailability is the fraction of an orally administered drug that reaches site of action in an unchanged form.
It is influenced by factors like absorption, metabolism and solubility.

__Task__: Given a drug ("SMILES"), predict the activity of bioavailability in Binary (0 or 1)

- Bioavailable - 1 
- Not Bioavailable - 0

__Size__: 640 drug molecules

__Source__: TDC (Therapeutics Data Commons), a collection of curated datasets and tools to apply machine learning in drug discovery

#### Steps to downloading dataset from TDC

1- To retrieve dataset from TDC, install its python package.

Install the package with:

`pip install pytdc` 

To avoid potential dependency conflicts, use:

`pip install "pytdc" "aiobotocore>=2.5,<3" "boto3>=1.37,<2" "botocore>=1.37,<2" "s3transfer>=0.5,<1"` 

2- Use Jupyter Notebook for Interactive Exploration

Check the notebook: notebooks/bio_data_handling.ipynb

Set up Jupyter Notebook if not installed already:
```
# using conda
conda install -c conda-forge notebook

# using pip
pip install notebook
```
Launch Jupyter:

`jupyter notebook`

Then run the following code to download and explore the dataset

retrieve the dataset from TDC:
```
from tdc.single_pred import ADME
data = ADME(name = 'Bioavailability_Ma')
```
load the dataset in pandas Dataframe for handling structured data in python:

  `df = data.get_data()`

save dataset in .csv format in data/ to keep it organized, .tab format saves by default in the working directory
```
df.to_csv("../data/bioavailability.csv", index=False)
```

explore features and basic info about dataset

`df.head()  # to show the first 5 rows`

`df.dtypes # to check data types`

`df.columns`

`df['Y'].dtype  # to check data type of target column`

```
split = data.get_split()
split.keys()  # check splits available
```

Save this notebook as bio_data_handling.ipynb in the notebooks/ directory.

3- Use Python Script for Automation

To make workflow reproducible, use the script at: scripts/data_handling.py.

To run it:

`python ./scripts/data_handling.py`

This script will:

Download and save the dataset to data/

Keep workflow easily reusable and automated

### Featurising the data

__Featuriser__: _Ersilia Compound Embeddings_ (eos2gw4)

This is useful in predicting bioavailability, because it encodes chemical and bioactivity information of drug molecules not just the structure. Therefore providing comprehensive molecular representation and since bioavailability is influenced by chemical structure, physicochemical properties and biology activity, its better suited. Also it is pretrained on bioactive molecules from FS-Mol and chEMBL, which ensures it captures meaningful patterns from well-established datasets and similarities between drugs which make machine learning generalize better.

#### Steps to featurise the data:

Since the featuriser is a representation model from ersilia hub, and previously installed [ersilia](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#installing-ersilia)

1- fetch the model:

`ersilia fetch eos2gw4`

2- serve the model:

`ersilia serve eos2gw4`

3- run, passing the saved dataset as the input, specify the output in a file:

`ersilia run -i ./data/bioavailability.csv -o ./data/featurised_bioavailability.csv`

this will take your dataset and return a featurised dataset in the file specified.

4- Using python scripts for automation and easily reproducible, can be found /scripts. run:
`python ./scripts/bio_data_featurising.py`

this will return a featurised dataset in the data/ successfully.

### Training the model

For training the model, use XGBoost due to its optimization for performance with structured data. Additionally, scikit-learn is used for preprocessing, evaluation, and saving the model.

#### Steps to train a model

1-  Install the Required Packages

Ensure all required dependencies installed:

`pip install xgboost scikit-learn matpotlib` 

confirm installation with:

`pip list`

2-  Merge the Featurised Data with Raw Data

Combine the featurised and raw data to include the target column (Y) in the dataset. The following code does that:

```
# load raw and featurised data
raw_df = pd.read_csv("../data/bioavailabity.csv")
featurized_df = pd.read_csv("../data/featurised_bioavailability.csv")

# merge both data
merged_df = featurized_df.merge(raw_df[["Drug_ID", "Y"]], left_on="key", right_on="Drug_ID")

# drop unneccessary columns
merged_df = merged_df.drop(columns=["key", "input", "Drug_ID"])
merged_df.to_csv("../data/merged_ft_bioavailability.csv", index=False)
```
Now, the dataframe will have both features and the target column (Y), making it ready for training.

3- Separate Features and Target

Split the data into features (x) and target (y):

```
x = merged_df.drop(columns=["Y"])
y = merged_df["Y"]
```

4- Split data into training and testing set

Use scikit-learn train_test_split to split the data into training and testing sets:

```
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
```

5- Create model and train

```
import xgboost as xgb  

model = xgb.XGBClassifier(
    learning_rate=0.2,
    n_estimators=100,
    max_depth=6,
    scale_pos_weight=2,
    random_state=42  
)  

model.fit(x_train, y_train)
```
This will train the model on the data and prepare it for predictions.

### Evaluating the model

Once the model is trained, it’s crucial to evaluate its performance using different metrics. Accuracy, precision, recall, F1 score, and AUC is used. Also, visualized the results with a confusion matrix, ROC curve, and precision-recall curve.

#### Steps to evaluate the model

1. Model Evaluation Metrics

```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  

# predict model performance
y_pred_prob = model.predict_proba(x_test)[:, 1]  

threshold = 0.9
y_pred_custom = (y_pred_prob >= threshold).astype(int)

# calculate metrics
accuracy = accuracy_score(y_test, y_pred_custom)
precision = precision_score(y_test, y_pred_custom)
recall = recall_score(y_test, y_pred_custom)
f1 = f1_score(y_test, y_pred_custom)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc:.4f}")
```

2. Visualizing results with matplotlib and seaborn

- Confusion matrix: To evaluate the true vs. predicted classes.

```
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_custom)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.show()
```

- ROC curve: To evaluate the performance at various thresholds.

```
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray") 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

- Precision-recall curve: To evaluate precision vs. recall at various thresholds.

```
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
```

3. Save the trained the model

```
import joblib
joblib.dump(model, "../models/bioavailability.pkl")
```

4. Saving Evaluation Metrics as Images

Visualizations of the confusion matrix, ROC curve, and precision-recall curve are saved in the results/ folder. Also code preview can be in the /notebooks//bioavailability_train.ipynb 

5. Automate the Process

To automate model training, evaluation, and prediction, run the following script:

`python ./scripts/bioavailability_train.py`

This script will build, train, and predict the bioavailability model automatically, making it reproducible and easier to run in the future.

### Model Summary

After training the XGBoost model, it achieved:

Accuracy: 79.69%

Precision: 83.96%

Recall: 90.82%

ROC-AUC Score: 0.7078

This ROC-AUC score is close to the TDC benchmark, which reports around 0.706 ± 0.031 for similar tasks.

To address class imbalance, the following adjustments were made:
- Stratified sampling while splitting the dataset
- Increased scale_pos_weight to 2
- Adjusted the decision threshold to 0.9

#### Improvement:
- **False Positive Bioavailable Predictions**: The confusion matrix indicated the model struggled with false bioavailable. This could be due to class imbalance, which was handled with methods like stratified sampling and adjusting the scale_pos_weight.
- **Threshold Adjustment**: The current threshold of 0.9 led to a good recall but possibly at the cost of precision. Further tuning of the threshold or experimenting with different models may improve performance.


### Using the model later

To use the model, the saved model has to be loaded;

```
import joblib
model = joblib.load("../models/bioavailability.pkl")
```

Make predictions

`y_pred = model.predict(x_test)`

### Extra Model Validation

#### Tried a different featurisation
_Morgan counts fingerprints_ (eos5axz) as [above](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#featurising-the-data)

#### Test other ML architectures using the fingerprints descriptors:

- Random Forest
- svm (Support Vector Machine)

Following the steps to [training](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#steps-to-build-a-model) a model

Alternative Models: Random Forest & SVM:
```
from sklearn.ensemble import RandomForestClassifier
import joblib

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
joblib.dump(rf_model, "../models/random_forest_bioavailability.pkl")

y_pred_rf = rf_model.predict(x_test)
y_pred_rf_prob = rf_model.predict_proba(x_test)[:, 1]
```
```
from sklearn.svm import SVC
svm_model = SVC(probability=True, kernel='rbf', random_state=42)
svm_model.fit(x_train, y_train)

joblib.dump(svm_model, "../models/svm_bioavailability.pkl")

y_pred_svm = svm_model.predict(x_test)
y_pred_svm_prob = svm_model.predict_proba(x_test)[:, 1]
```

#### Evaluating how each model performed and the AUROC score:

_xgboost model trained with ersilia embeddings_

AUROC score = 0.7078 and model performance is generally good.

_Random forest model trained with fingerprints descriptors_

AUROC score = 0.7162 and model classification is not good, misclassified more non-bioavailable compounds as bioavailable.
_svm model trained with fingerprints descriptor_

AUROC score = 0.7378 but model classification is poor, misclassified more non-bioavailable compounds as bioavailable and vice versa.

Confusion matrix and visualization results can be found in /notebooks/model_validation.ipynb

#### Apply models to public dataset

Downloaded a new drug molecules dataset _HIA_Hou_ for prediction using my models because both datasets are for an ADME task:

_xgboost model trained with ersilia embeddings_ : AUC Score: 0.7975 but classification is fairly better when compared to other two models

_Random forest model trained with fingerprints descriptors_ : AUC Score: 0.8228, model classification is suboptimal, as it misclassified many non-bioavailable compounds as bioavailable.

_svm model trained with fingerprints descriptor_ : AUC Score: 0.8688 but model exhibited poor classification performance.

### Conclusion

XGBoost model trained with descriptors using ersilia compound embeddings performed better considering the AUROC score and how well the model classified bioavailables and non bioavailables.










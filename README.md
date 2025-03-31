## Apply Ersilia Models to a modelling task

### Project steps:
1. [Download a dataset](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#download-a-dataset)
1. [Featurise the data](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#featurising-the-data)
1. [Build an ML model](https://github.com/AzeematRaji/ersilia-outreachy/edit/main/README.md#build-an-ml-model)

Setting the environment:

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

confirm erisilia:

`ersilia --help`

### Download a dataset

#### Background of Data

__Dataset__: _Bioavailability, Ma et al._

Oral bioavailability is the fraction of an orally administered drug that reaches site of action in an unchanged form.
It is influenced by factors like absorption, metabolism and solubility

__Task__: Given a drug ("SMILES"), predict the activity of bioavailability in Binary (0 or 1)

- Bioavailable - 1 
- Not Bioavailable - 0

__Size__: 640 drug molecules

__Source__: TDC (Therapeutics Data Commons), a collection of curated datasets and tools to apply machine learning in drug discovery

#### Steps to downloading dataset from TDC

1- To retreive dataset from TDC, install its python package:

`pip install pytdc` normal installation

`pip install "pytdc" "aiobotocore>=2.5,<3" "boto3>=1.37,<2" "botocore>=1.37,<2" "s3transfer>=0.5,<1"` to avoid any dependencies conflict

2- Using Notebook for an interactive session, can be found in notebooks/

Set up notebook:
- install jupyter notebook via conda `conda install -c conda-forge notebook` or
- install jupyter notebook via pip `pip install notebook`
- launch notebook from your root/ `jupyter notebook`

retrieve the dataset from TDC:
```
from tdc.single_pred import ADME
data = ADME(name = 'Bioavailability_Ma')
```
load the dataset in pandas Dataframe for handling structured data in python:

  `df = data.get_data()`

save dataset in .csv format in data/ to keep it organized, .tab saves by default in the working directory
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
split.keys()  # splits are available
```

save notebook `data_handling.ipynb` in the notebooks/

3- Using python scripts for automation and easily reproducible, this can be found in scripts/ 

to download, load and save dataset, the script can easily be run using:

`python ./scripts/data_handling.py`


Dataset has been successfully downloaded, loaded and saved into the data/, notebook saved in the notebooks/ and script saved in scripts/.

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

`ersilia run -i ./data/bioavailabilty.csv -o ./data/featurised_bioavailability.csv`

this will take your dataset and return a featurised dataset in the file specified.

4- Using python scripts for automation and easily reproducible, can be found /scripts. run:
`python ./scripts/bio_data_featurising.py`

this will return a featurised dataset in the data/ successfully.

### Build an ML model

- XGBoost for model training because it handles structured data well and optimized for performance.
- scikit-learn provides utilities for preprocesing, evaluation and saving the model.

#### Steps to build a model

1- Install/confirm required packages, xgboost, sklearn, matpotlib.

`pip install xgboost scikit-learn matpotlib` `pip list`

2- Merged featurised data and raw data to have the y column in the dataframe, this is because of the featuriser that was used

```
raw_df = pd.read_csv("../data/bioavailabity.csv")
featurized_df = pd.read_csv("../data/featurised_bioavailability.csv")
merged_df = featurized_df.merge(original_df[["Drug_ID", "Y"]], left_on="key", right_on="Drug_ID")
merged_df = merged_df.drop(columns=["key", "input", "Drug_ID"])
merged_df.to_csv("../data/merged_ft_bioavailability.csv", index=False)
```
the dataframe is now features and target column ready to be use for training

3- Seperate data into x and y

```
x = merged_df.drop(columns=["Y"])
y = merged_df["Y"]
```

4- Split data into training and testing

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

6- Making predictions

```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  

y_pred_prob = model.predict_proba(x_test)[:, 1]  

threshold = 0.9
y_pred_custom = (y_pred_prob >= threshold).astype(int)
```

7- Evaluating the model

```
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
8- Visualizing results with matplotlib

- Confusion matrix

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

- ROC curve

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

- Precision-recall curve

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

9- Save the trained the model

```
import joblib
joblib.dump(model, "../models/bioavailability.pkl")
```

10- Images of evaluation metrics can be found /results, also code preview can be in the /notebooks//bioavailability_train.ipynb 

### Model hypothesis

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

The confusion matrix indicated that the model still struggled slightly with false positives, likely due to the imbalance in the dataset. However, compared to similar tasks in TDC benchmarks, where models typically achieve ~70% ROC-AUC, our model performed reasonably well but could be improved.












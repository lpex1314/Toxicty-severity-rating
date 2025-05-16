This Repository contains the jupyter notebook for severity of toxic commets rating nlp task, 
jigsaw-rate-severity-of-toxic-comments-nlp.ipynb

Jigsaw rate severity of toxic comments is a competition hosted by kaggle at 
https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating

The notebook is structured as follows:
- Setup
- Data preparation
- Bert Encoding Method
  - Feature Engineering Improvements
  - Hyperparameters selection
  - Training
  - Validation
- TF-IDF (sparse, bag-of-words model)
  - Training
  - Validation
- Test and Submission
  
For reproducing results, the most convenient way to run this notebook in kaggle-provided server(because not only kaggle cloud server already has all necessary dependcies installed, but it also provides accelerators). Plus, only by submiting in kaggle competition can we obtain final test result. 

Simply open the competition website, under `code` section, click `+ New Notebook` and upload this notebook.

Turn on Internet, choose accelerator as GPU P100 and click `run all`, the notebook will run all code cells and output all necessary results and visualizations.

For submission to test, it's required to turn off internet and use locally pre-downloaded bert-base-uncase moded.
- download bert-base-uncase model and upload to kaggle via 'upload -- new model'
- then in 
```
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
bert = transformers.AutoModel.from_pretrained(model_name)
bert.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert.to(device)
```

change  `model_name` to `model_path`, which might be `'/kaggle/input/bert_base_uncased/pytorch/default/1/bert_base_uncased'`

Or it's also feasible to run locally but with downloading datasets manually and change the paths of datasets

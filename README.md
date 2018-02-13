# SETuP
Summary Evaluation Tool using Preferences

# Installation
All dependecies can be installed by running the following command:

```
pip install -r requirements.txt
```

# Config
The project includes a file named Config.py. It is used to configure the paths to external tools. These include the Stanford NLP tagger and the word2vec dataset from Google. Stanfords NLP tagger is already configured and included in the project. You can download the Google word2vec news dataset from the following url: <https://github.com/nishankmahore/word2vec-flask-api>. After downloading the dataset you need to configure the path to the folder in the Config.py file.

# Usage
```
python SETuP.py 
Usage: SETuP.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  classify
  train
```

For more information on the exact parameters look at the examples below.

# Example
## Training a new model
To test how the training of a new model works you can run the script "test-training.sh", it includes the following command:
```
 python SETuP.py train 
    --model="training-test"
    --path-source-documents="Training Example Data/Source Documents/"
    --path-system-summaries="Training Example Data/System Summaries/"
    --path-reference-summaries="Training Example Data/Reference Summaries/"
    --path-system-summary-scores="Training Example Data/System Summary Scores/"
```

## Classifying data
To test how the classification works you can run the script "test-classification.sh", it includes the following command:

```
 python SETuP.py classify  
    --model="randomforest-tac-2008-2009.pkl"
    --path-source-documents="Classification Example Data/Source Documents/"
    --path-system-summaries="Classification Example Data/System Summaries/"
    --path-reference-summaries="Classification Example Data/Reference Summaries/"
```



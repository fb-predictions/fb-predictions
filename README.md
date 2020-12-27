#fb-predictions

In this project, we've tried to predict popularity of Facebook posts. 
We focused on anonymous posts from confession pages of universities in the UK, USA and Canada. 

The popularity of a post was measured by the post *engagement rate*.

All code files have been built to use with a specific data format. If you use your own data, please make sure it's in the same format.

## Installation
* Clone this repo:
```
git clone https://github.com/TavRotenberg/ai-project-fb
```

* Install requirements:
```
pip install -r requirements.txt
```
## Datasets
The dataset used in this project consists of approximately 100K posts:
* About 75K belong to [Unitruths](https://www.facebook.com/unitruthz/) and are not available here.
* About 25K were collected by us through scrapping. It is available here, along with the scrapping script.

### Our Dataset
Our dataset is available in the [data](data) folder. 
For each FB group, there are two csv files:
* group_name.csv - includes the post text, date and time posted and more.
* group_name_reactions.csv - includes the amount of reactions the post received and the tree dominant reactions.

In addition, there are two json files:
* followers.json - specifies how many followers each group had when the data eas collected
* timezones.csv - specifies the timezone of the group geographical location

### Collection Dataset through Scrapping
To collecting data through scrapping, go to [facebook-scraper](https://github.com/fb-predictions/facebook-scraper) repository.

It should be noted that this method collects about 15% of the group posts.

## Data Preprocessing
Preprocess the data using the following script:
```bash
python data_preproccessing.py -p path/to/dataset/folder -s path/to/saving/folder
```
Arguments:
```
-p      --path 	                Path to the input directory  (default data)

-s      --saving_path 	        Path to the output directory  (default processed_data)
```

The data directory should contain files as specified under [Our Dataset](#our-dataset).


At the end of preprocessing, the saving_path folder will include the files:
* all_train.csv
* all_validation.csv
* all_test.csv

In the [supplementary](supplamentary) folder, you can find the preprocessing script we used with the complete data. There are syntax differences as it arrived in a different format. Unfortunately you can't use this file as you don't have the complete dataset. 

## Data Statistics
Collect data statistics using the following script:
```bash
python data_statistics.py -p path/to/dataset/folder -s path/to/saving/folder -t stat_type -f feature -t target
```
Arguments:
```
-p      --path 	                Path to the input directory  (default data)

-s      --saving_path 	        Path to the output directory  (default data_stats)

-ts     --type 	                Type of statstic (hist (default) /correlation/trendline/groups/trigger/pancuations)

-f      --feature               Feture (year/month/word_count etc. depend on data fetures)

-t      --target                Target (likes/eng_rate/presiction/abs_errot etc. depend on data)

-b      --balance               Balance data (yes/no)
    
-y      --by_year               Compute statistic by year
```

## Regressors
We used simple regressors (MLP/Linear/Random Forest), and a more complicated regressor that includes BERT.
### Simple Regressors
Run the simple regressors using the following script:
```bash
python baseline_regressors.py -p path/to/dataset/folder -s path/to/saving/folder -r regressor
```
Arguments:
```
-p      --path 	                Path to the input directory  (default data)

-s      --saving_path 	        Path to the output directory  (default regressors_out)

-r      --regressor             The regressor to use (MLP (default) /Linear/Decision_Tree)

-b      --balance               Balance data (yes/ no (default))
```
At the end of prediction, the saving_path folder will include the following file:
* regressor_name_predictions.csv

The file contains the validation csv you provided with the predictions of the regressor
### BERT
Run the regresssor containing BERT using the following script:
```bash
python bert.py -p path/to/dataset/folder -s path/to/saving/folder -m model -lr learning_rate -e epochs -b batch_size -n name
```
Arguments:
```
-p      --path 	                Path to the input directory  (default data)

-s      --saving_path 	        Path to the output directory  (default regressors_out)

-m      --model                 Bert model to use (1 - bert-cased , 2 - distil-bert-cased (default))

-lr      --lr                   Learining rate to use (default 2e-5)

-l      --max_len               Maximum length of posts to run through BERT (default 256, maximum 512)

-e      --epoch                 Number of epochs (default 10)

-f      --freeze                Wether to freeze BERT's weights (default no)

-b      --batch_size            Batch size to use (default 16)

-t      --train                 Whether to trian model (yes (default) / no)

-w      --pretrained_weights    Path to pretrained model wights (no - no pretrined weights (defualt))

```
At the end of prediction, the saving_path folder will include the following files:
* BERT_prdictions.csv
* BERT_weights.csv

#### Download Pretrained Weights 
You can download our pretrained weights from [here](https://drive.google.com/file/d/1xg85UJCBRyz2NMpt40YOA4GOETNctJTK/view?usp=sharing).
You can use them, or your own weights, to continue training using the *bert.py* script with parameter *-w path/to/weights* or get predictions without further training using the *-t no* parameter.

### Get Predictions Measures
Run the following script to get predictions measures:
```bash
python get_results.py -p path/to/prediction/file -s path/to/saving/folder -m model
```
Arguments:
```
-p      --path 	                Path to the input file  

-s      --saving_path 	        Path to the output directory  

-r      --regressor             The regressor used (MLP/Linear/Decision_Tree/BERT)
```

The script provides the following measures about the prediction:
* Mean absolute error
* Accuracy
* Precision
* Recall
* F1
* Confusion Matrix

## Sample Posts
Run the following script to sample posts:
```bash
python get_results.py -p path/to/prediction/file -s path/to/saving/folder -t uniform -n num_posts
```
Arguments:
```
-p      --path 	                Path to the input file (default - prcossed_data/all_validation.csv)

-s      --saving_path 	        name of output file (default - data_exp_out/sample.csv)  

-t      --type                  How to sample posts (uniform - all posts have the same probability/ abs_error - posts with higher abs_error have higher probability to be chosen)

-n      --size                  The sample size
```
This script will produce a file containing the sampled posts.

---
layout: default
---
[Learners](../../Infer.NET Learners.md) : [Matchbox recommender](../Matchbox recommender.md)

## Command-line runners

The Infer.NET command-line runners are defined in the Learner.exe which can be found in the bin directory of the Infer.NET release. They are intended to simplify the usage of the learners. In order to use the recommender from the command line, simply invoke the Learner.exe with the "Recommender" argument followed by the name of the operation to apply. This can be one of the following: SplitData, GenerateNegativeData, Train, PredictRatings, RecommendItems, FindRelatedUsers, FindRelatedItems, EvaluateRatingPrediction, EvaluateItemRecommendation, EvaluateFindRelatedUsers, EvaluateFindRelatedItems. Each one of these can be followed by a number of operation-specific options which are prefixed by `--` (dash, dash). Here is an example sequence of commands which splits the data into a training and a test set, trains the recommender and then for each one of the four modes of prediction makes predictions and evaluates these predictions.

### Example

```
Learner.exe Recommender SplitData --input-data Dataset.dat --output-data-train Train.dat   
                        --output-data-test Test.dat --ignored-users 0.8 --training-users 0.1  
Learner.exe Recommender Train --training-data Train.dat --trained-model Model.bin --traits 4   
                        --iterations 20 --batches 2 --use-user-features --use-item-features  

Learner.exe Recommender PredictRatings --data Test.dat --model Model.bin  
                        --predictions Predictions.dat  
Learner.exe Recommender EvaluateRatingPrediction --test-data Test.dat   
                        --predictions Predictions.dat --report Report.dat  

Learner.exe Recommender RecommendItems --data Test.dat --model Model.bin  
                        --predictions Predictions.dat  
Learner.exe Recommender EvaluateItemRecommendation --test-data Test.dat   
                        --predictions Predictions.dat --report Report.dat  

Learner.exe Recommender FindRelatedUsers --data Test.dat --model Model.bin   
                        --predictions Predictions.dat  
Learner.exe Recommender EvaluateFindRelatedUsers --test-data Test.dat  
                        --predictions Predictions.dat --report Report.dat  

Learner.exe Recommender FindRelatedItems --data Test.dat --model Model.bin  
                        --predictions Predictions.dat  
Learner.exe Recommender EvaluateFindRelatedItems --test-data Test.dat   
                        --predictions Predictions.dat --report Report.dat
```

### Data format

The data format for the command-line runners is fixed. That is because here we do not have the flexibility of user defined data mappings. This means that in order to use the command-line runners, the user first needs to convert their data into the required format.

The data is specified in a single file which to a certain extent resembles the data mapping interface. The file contains multiple lines, where each line is one of the following records:

*   A descriptor of the ratings. This entry starts with the letter "R", and then specifies the minimum and the maximum rating. These are comma-separated.
*   A rating instance. This is a user-item-rating triple, where the three entities are comma-separated.
*   A descriptor of the user features (optional). This entry starts with the letter "U" followed by a unique user identifier, and then the user features. These three are comma-separated. The features, in turn, are given in the sparse form of \[feature id : feature value\] and are separated by "\|" (a vertical bar). Feature ids that do not appear in this list are assumed to have value 0.
*   A descriptor of the item features (optional). Same as the user feature descriptor, but starts with "I" instance of "U".

Note that the rating descriptor has to be the first line in the file. The order of the other records does not matter.

For example, here is a sample file which defines the ratings given by two users to four items. It first specifies that the minimum rating in the dataset is 0, and the maximum rating is 5. It then enumerates all instances. This is followed by the user features - each user in this dataset has a five-dimensional feature vector specified in a sparse format. Finally, the features for the items are given - each item has seven features.

```
R,0,5  
Bob,Movie 1,0  
Alice,Movie 2,1  
Bob,Movie 3,2  
Alice,Movie 4,3  
Alice,Movie 3,4  
Bob,Movie 4,5  
U,Alice,0:1|3:0.1|4:0.6  
U,Bob,1:0.2|2:1|3:0.5  
I,Movie 2,5:0.8|6:1  
I,Movie 1,0:0.2|1:0.3|2:0.3  
I,Movie 3,3:0.4|6:0.9  
I,Movie 4,3:0.01|4:1|5:0.33
```

Command-line runners: [Negative data generator](../Matchbox/Runners/Negative data generator.md) \| [Data splitter](../Matchbox/Runners/Data splitter.md) \| [Trainer](../Matchbox/Runners/Trainer.md) \| [Predictors](../Matchbox/Runners/Predictors.md) \| [Evaluators](../Matchbox/Runners/Evaluators.md)

---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Command-line runners](../../Matchbox recommender/Command-line runners.md)

## Predictors

The Matchbox recommender in Infer.NET supports 4 modes of prediction - rating prediction, item recommendation, find related users, and find related items. There is a command-line runner for each one of these.

### Rating predictor

Rating prediction is performed using the `PredictRatings` argument to Learner Recommender. It takes in a test set and a model, and produces a file of predictions.

Required parameters

*   **data** \- dataset to make predictions for; typically produced by the [command-line data splitter](Data splitter.md)
*   **model** \- file containing a trained model produced by the [command-lined trainer](Trainer.md)
*   **predictions** \- the file containing the generated predictions

Example

```
Learner Recommender PredictRatings --data TestSet.dat   
                                   --model TrainedRecommender.bin   
                                   --predictions RatingPredictions.dat
```
### Item recommender

Item recommendation is performed using the `RecommendItems` argument to Learner Recommender. It takes in a test set and a model, and produces a file of predictions. Note that items are recommended with the intention to be later evaluated, so this module follows the prediction procedure explained here. That is, items are recommended for each unique user in the test set, and predictions are restricted to the items that each user has rated.

Required parameters

*   **data** \- dataset to make predictions for; typically produced by the [command-line data splitter](Data splitter.md)
*   **model** \- file containing a trained model produced by the [command-lined trainer](Trainer.md)
*   **predictions** \- the file containing the generated predictions

Optional parameters

*   **max-items** \- maximum number of items to recommend; defaults to 5
*   **min-pool-size** \- minimum size of the recommendation pool for a single user; defaults to 5

Example

```
Learner Recommender RecommendItems --data TestSet.dat   
                                   --model TrainedRecommender.bin  
                                   --predictions ItemRecommendations.dat   
                                   --max-items 10
```
### Related user finder

Related users can be found using the `FindRelatedUsers` argument to Learner Recommender. It takes in a test set and a model, and produces a file of predictions. Note that related users are found with the intention to be later evaluated, so this module follows the prediction procedure explained here. That is, related users are found for users who have rated a given number of items in common.

Required parameters

*   **data** \- dataset to make predictions for; typically produced by the [command-line data splitter](Data splitter.md)
*   **model** \- file containing a trained model produced by the [command-lined trainer](Trainer.md)
*   **predictions** \- the file containing the generated predictions

Optional parameters

*   **max-users** \- maximum number of related users for a single user; defaults to 5
*   **min-common-items** \- minimum number of items that the query user and the related user should have rated in common; defaults to 5
*   **min-pool-size** \- minimum size of the related user pool for a single user; defaults to 5

Example

```
Learner Recommender FindRelatedUsers --data TestSet.dat  
                                     --model TrainedRecommender.bin  
                                     --predictions RelatedUsers.dat  
                                     --max-users 10  
                                     --min-common-items 2
```
### Related item finder

Related items can be found using the `FindRelatedItems` argument to Learner Recommender. It takes in a test set and a model, and produces a file of predictions. Note that related items are found with the intention to be later evaluated, so this module follows the prediction procedure explained here. That is, related items are found for items which have been rated by a given number of users in common.

Required parameters

*   **data** \- dataset to make predictions for; typically produced by the [command-line data splitter](Data splitter.md)
*   **model** \- file containing a trained model produced by the [command-lined trainer](Trainer.md)
*   **predictions** \- the file containing the generated predictions

Optional parameters

*   **max-items** \- maximum number of related items for a single item; defaults to 5
*   **min-common-users** \- minimum number of users that the query item and the related item should have been rated by in common; defaults to 5
*   **min-pool-size** \- minimum size of the related item pool for a single item; defaults to 5

Example

```
Learner Recommender FindRelatedItems --data TestSet.dat   
                                     --model TrainedRecommender.bin  
                                     --predictions RelatedItems.dat  
                                     --max-items 10   
                                     --min-common-users 2
```

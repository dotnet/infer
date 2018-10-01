---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Command-line runners](../../Matchbox recommender/Command-line runners.md)

## Evaluators

Evaluators exist for each one of the four prediction modes - rating prediction, item recommendation, find related users and find related items.

### Rating prediction evaluator

Rating prediction can be evaluated using the EvaluateRatingPrediction argument to Learner Recommender. It takes in a test set and predictions for this test set, and produces a report of evaluation metrics, which include mean absolute error (MAE), and root mean squared error (RMSE). These metrics as well as the way the evaluator works are explained in the [corresponding evaluation API section](../API/Evaluation/Rating prediction.md).

Required parameters

*   **test-data** \- dataset to make predictions for; typically produced by the [command-line data splitter](Data splitter.md)
*   **predictions** \- predictions to evaluate; typically produced by Learner Recommender PredictRatings
*   **report** \- evaluation report file

Example
```
Learner Recommender EvaluateRatingPrediction --test-data TestSet.dat  
                                             --predictions RatingPredictions.dat   
                                             --report RatingPredictionEvaluation.txt
```
### Item recommendation evaluator

Item recommendations can be evaluated using EvaluateItemRecommendation argument to Learner Recommender. It takes in a test set and predictions for this test set, and produces a report which specifies the computed normalized discounted cumulative gain (NDCG). This metric as well as the way the evaluator works is explained in the [corresponding evaluation API section](../API/Evaluation/Item recommendation.md).

Required parameters

*   **test-data** \- dataset to make predictions for; typically produced by the [command-line data splitter](Data splitter.md)
*   **predictions** \- predictions to evaluate; typically produced by Learner Recommender RecommendItems
*   **report** \- evaluation report file

Example
```
Learner Recommender EvaluateItemRecommendation --test-data TestSet.dat  
                                               --predictions ItemRecommendations.dat   
                                               --report ItemRecommendationEvaluation.txt
```
### Related user evaluator

Related users can be evaluated using the `EvaluateFindRelatedUsers` argument to Learner Recommender. It takes in a test set and predictions for this test set, and produces a report of evaluation metrics, which include NDCG with L1 and L2 similarities as gains. These metrics as well as the way the evaluator works are explained in the [corresponding evaluation API section](../API/Evaluation/Related users.md).

Required parameters

*   **test-data** \- dataset to make predictions for; typically produced by the [command-line data splitter](Data splitter.md)
*   **predictions** \- predictions to evaluate; typically produced by Learner Recommender FindRelatedUsers
*   **report** \- evaluation report file

Optional parameters

*   **min-common-items** \- minimum number of items that the query user and the related user should have rated in common; defaults to 5

Example
```
Learner Recommender EvaluateFindRelatedUsers --test-data TestSet.dat  
                                             --predictions RelatedUsers.dat   
                                             --report RelatedUserEvaluation.txt
```
### Related item evaluator

Related items can be evaluated using the EvaluateFindRelatedItems argument to Learner Recommender. It takes in a test set and predictions for this test set, and produces a report of evaluation metrics, which include NDCG with L1 and L2 similarities as gains. These metrics as well as the way the evaluator works are explained in the [corresponding evaluation API section](../API/Evaluation/Related items.md).

Required parameters

*   **test-data** \- dataset to make predictions for; typically produced by the [command-line data splitter](Data splitter.md)
*   **predictions** \- predictions to evaluate; typically produced by Learner Recommender FindRelatedItems
*   **report** \- evaluation report file

Optional parameters

*   **min-common-items** \- minimum number of users that the query item and the related item should have been rated by in common; defaults to 5

Example
```
Learner Recommender EvaluateFindRelatedItems --test-data TestSet.dat  
                                             --predictions RelatedItems.dat   
                                             --report RelatedItemEvaluation.txt
```

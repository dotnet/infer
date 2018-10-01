---
layout: default
---
[Learners](../../../../Infer.NET Learners.md) : [Matchbox recommender](../../../Matchbox recommender.md) : [Learner API](../../../Matchbox recommender/Learner API.md) : [Evaluation](../../Evaluation.md)

## Item recommendation

In order to evaluate item recommendation, there needs to be a ground truth rating in the test set for each recommended item to a given user. As we do not usually have ratings for all user-item pairs, the set of items that can possibly be recommended to a user must be restricted to those for which we have a rating in the dataset. We will cover here how to make such predictions and how to evaluate them. 

### Recommending items for evaluation

Items can be recommended for evaluation using the `RecommendRatedItems` method of the evaluator:

```csharp
var evaluator = new  RecommenderEvaluator
    <Dataset, User, Item, int, int, Discrete>(dataMapping.ForEvaluation());  
var itemRecommendationsForEvaluation = evaluator.RecommendRatedItems(
    trainedRecommender, testDataset, maxRecommendedItemCount, minRecommendationPoolSize); 
```

In this example `trainedRecommender` is a trained recommender and `testDataset` is the instance source of the test set. Recommendations will be made to each unique user in this dataset. The item in each instance will also be queried by the data mapping. This allows a collection of items to be constructed that can be recommended to each user. Item recommendations are then made only from this collection. Ratings will not be queried by the data mapping at this stage. They will only be needed during evaluation. The parameter `maxRecommendedItemCount` specifies the number of items to be recommended to each user. But if the number of possible items to recommend to a user is less than the value of `minRecommendationPoolSize`, then the user is skipped. The last parameter allows for easy removal of users from the predictions for whom there is not sufficient information for later evaluation.

### Evaluating item recommendation

Once the restricted recommendations are produced, they can be evaluated using the `ItemRecommendationMetric` method of the evaluator:

```csharp
var ndcg = evaluator.ItemRecommendationMetric(  
    testDataset, itemRecommendationsForEvaluation, Metrics.Ndcg);
```

`testDataset` is the test instance source of user-item-rating triples to use for evaluation. The item recommendations need to have been produced using the user and item pairs of the same dataset. The ranking metric can be one of the following:

*   [Discounted Cumulative Gain](http://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain) (DCG)
*   Linear DCG - same as DCG, but uses a linear discount function instead of a logarithmic one
*   [Normalized Discounted Cumulative Gain](http://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG) (NDCG)
*   Linear NDCG - same as NDCG, but uses a linear discount function instead of a logarithmic one
*   [Graded Average Precision](http://dl.acm.org/citation.cfm?id=1835550)

The ratings are used as gains for the computation of the first two metrics above. However, one can specify a custom gain function as a fourth parameter of the `ItemRecommendationMetric` method. For example, here is how to deal with zero-based ratings:

```csharp
var ndcg = evaluator.ItemRecommendationMetric(  
    testDataset,  
    itemRecommendationsForEvaluation, Metrics.Ndcg, rating => Convert.ToDouble(rating) - minRating + 1));
```

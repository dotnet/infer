---
layout: default
---
[Learners](../../../../Infer.NET Learners.md) : [Matchbox recommender](../../../Matchbox recommender.md) : [Learner API](../../../Matchbox recommender/Learner API.md) : [Evaluation](../../Evaluation.md)

## Related users

In order to evaluate finding related users, the predicted set of related users needs to be restricted to such users who have rated some items in common with the query user. Here we cover how to make such predictions and how to evaluate them.

### Finding related users for evaluation

Related users for evaluation can be found using the `FindRelatedUsersWhoRatedSameItems` method of the evaluator:

```csharp
var evaluator = new  RecommenderEvaluator<Dataset, User, Item, int, int, Discrete>(  
    dataMapping.ForEvaluation());  
var relatedUsersForEvaluation = evaluator.FindRelatedUsersWhoRatedSameItems(  
    trainedRecommender,  
    testDataset,  
    maxRelatedUserCount,  
    minCommonRatingCount,   
    minRelatedUserPoolSize);
```

In this example `trainedRecommender` is a trained recommender and `testDataset` is the instance source of the test set. Related users will be found for each unique user in the dataset.

The item in each instance will also be queried by the data mapping. This allows a list of rated items to be constructed for this user which can then be examined to see which other users have also rated these items. This gives a list of "potentially related" users, and then predictions are made from this list. Ratings will not be queried by the data mapping at this stage. They will only be needed during evaluation. The parameter `maxRelatedUserCount` specifies the number of related users to find for each user. But if the number of possible related users for a given user is less than the value of `minRelatedUserPoolSize`, then the user is skipped. The last parameter allows for easy removal of users from the predictions for whom there is not sufficient information for later evaluation. Another parameter which is used to control this is `minCommonRatingCount` \- it is guaranteed that all related users have rated at least that many items in common with the query user.

### Evaluating related users

Once the restricted related user predictions are produced, they can be evaluated using the `RelatedUsersMetric` method of the evaluator:

```csharp
var l1SimNdcg = evaluator.RelatedUsersMetric( testDataset,  
    relatedUsersForEvaluation,  
    minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity));
```

The ranking metric used to evaluate related users is one of the following:

*   [Discounted Cumulative Gain](http://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain) (DCG)
*   Linear DCG - same as DCG, but uses a linear discount function instead of a logarithmic one
*   [Normalized Discounted Cumulative Gain](http://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG) (NDCG)
*   Linear NDCG - same as NDCG, but uses a linear discount function instead of a logarithmic one

The fifth argument to the `RelatedUsersMetric` method is the rating similarity function, the value of which used as gain in the computation of the metrics above. It takes in two vectors and returns a real number. Pre-defined similarity functions include:

*   [Cosine similarity](http://en.wikipedia.org/wiki/Cosine_similarity)
*   [Normalized Manhattan similarity](http://simeon.wikia.com/wiki/Manhattan_distance) (also called L1)
*   [Normalized Euclidean similarity](http://simeon.wikia.com/wiki/Euclidean_distance) (also called L2)

The way evaluation works is the following. First, the sets of rated items for each user are extracted from the input data. These sets are then reduced to the items that both users rated in common. Then, for each item in these sets the rating given by the corresponding user is taken. This forms two user rating vectors, which are used as inputs to the functions listed above.

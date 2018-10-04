---
layout: default
---
[Learners](../../../../Infer.NET Learners.md) : [Matchbox recommender](../../../Matchbox recommender.md) : [Learner API](../../../Matchbox recommender/Learner API.md) : [Evaluation](../Evaluation.md)

## Related items

In order to evaluate finding related items, the predicted set of related items needs to be restricted to such items which have been rated by some users in common with the query item. Here we cover how to make such predictions and how to evaluate them.

### Finding related items for evaluation

Related items for evaluation can be found using the `FindRelatedItemsRatedBySameUsers` method of the evaluator: 

```csharp
var evaluator = new  RecommenderEvaluator<Dataset, User, Item, int, int, Discrete>(  
    dataMapping.ForEvaluation());  
var relatedItemsForEvaluation = evaluator.FindRelatedItemsRatedBySameUsers(  
    trainedRecommender,  
    testDataset,  
    maxRelatedItemCount,  
    minCommonRatingCount,   
    minRelatedItemPoolSize);
```

In this example `trainedRecommender` is a trained recommender and `testDataset` is the instance source of the test set. Related items will be found for each unique item in the dataset.

The user in each instance will also be queried by the data mapping. This allows a list of users to be constructed who have rated this item which can then be examined to see which other items have been rated by these users. This gives a list of "potentially related" items, and then predictions are made from this list. Ratings will not be queried by the data mapping at this stage. They will only be needed during evaluation. The parameter `maxRelatedItemCount` specifies the number of related items to find for each item. But if the number of possible related items for a given item is less than the value of `minRelatedItemPoolSize`, then the item is skipped. The last parameter allows for easy removal of items from the predictions for which there is not sufficient information for later evaluation. Another parameter which is used to control this is `minCommonRatingCount` \- it is guaranteed that all related items have been rated by at least that many users in common with the query item.

### Evaluating related items

Once the restricted related item predictions are produced, they can be evaluated using the `RelatedItemMetric` method of the evaluator:

```csharp
var l1SimNdcg = evaluator.RelatedItemsMetric( testDataset,  
    relatedItemsForEvaluation,  
    minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity));
```

The ranking metric used to evaluate related items is one of the following:

*   [Discounted Cumulative Gain](http://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain) (DCG)
*   Linear DCG - same as DCG, but uses a linear discount function instead of a logarithmic one
*   [Normalized Discounted Cumulative Gain](http://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG) (NDCG)
*   Linear NDCG - same as NDCG, but uses a linear discount function instead of a logarithmic one

The fifth argument to the `RelatedItemsMetric` method is the rating similarity function, the value of which used as gain in the computation of the metrics above. It takes in two vectors and returns a real number. Pre-defined similarity functions include:

*   [Cosine similarity](http://en.wikipedia.org/wiki/Cosine_similarity)
*   [Normalized Manhattan similarity](http://simeon.wikia.com/wiki/Manhattan_distance) (also called L1)
*   [Normalized Euclidean similarity](http://simeon.wikia.com/wiki/Euclidean_distance) (also called L2)

The way evaluation works is the following. First, the sets of users who rated each item are extracted from the input data. These sets are then reduced to the users who rated both items. Then, for each user in these sets the rating given to the corresponding item is taken. This forms two item rating vectors, which are used as inputs to the functions listed above.

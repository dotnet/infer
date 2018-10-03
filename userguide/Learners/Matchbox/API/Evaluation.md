---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Learner API](../../Matchbox recommender/Learner API.md)

## Evaluation

Infer.NET contains an implementation of a recommender evaluator, which is not bound to the Matchbox recommender. That is, it is designed to be able to evaluate rating prediction, item recommendation, related users, and related items of any recommender that implements the correct interfaces.

The evaluator, similarly to the Infer.NET learners, accesses data using a data mapping object. In particular, it requires an implementation of `IStarRatingRecommenderEvaluatorMapping`. This interface provides the evaluator with methods which return the users, the items, and the ratings, as well as a converter of ratings from data domain to integers. In addition, it declares functions for obtaining the items rated by a given user and the users who rated a given item. These methods are used for evaluating item recommendation, related users and related items.

The recommender evaluator data mapping can be defined using the standard data format recommender mapping. Therefore, we will not go into detail about the `IStarRatingRecommenderEvaluatorMapping` interface, but rather show how to use it. Particularly useful in this case is the `ForEvaluation` extension method, which performs the data mapping chaining - it takes in a standard recommender mapping (`IStarRatingRecommenderMapping`) and returns an evaluator mapping. Note that it cannot operate on the native recommender data format mapping, because we want the evaluator to be independent of concrete recommender implementations.

Once we have an evaluator mapping, we can instantiate the evaluator by calling into its constructor. When doing this, we need to specify four type parameters - instance collection, user, item, and rating in data domain. Here is an example:
```csharp
var recommenderMapping = new  StarRatingRecommenderMapping();
var evaluatorMapping = recommenderMapping.ForEvaluation();
var evaluator = new  StarRatingRecommenderEvaluator <IEnumerable<Instance>, string, string, double>(evaluatorMapping);
```
Evaluation is then performed by computing a metric for rating prediction, item recommendation, related users, or related items. For example:
```csharp
var predictions = recommender.Predict(testSet);
double mae = evaluator.RatingPredictionMetric(testSet, predictions, Metrics.AbsoluteError);
```
Evaluating: [Rating prediction](Evaluation/Rating prediction.md) \| [Item recommendation](Evaluation/Item recommendation.md) \| [Related users](Evaluation/Related users.md) \| [Related items](Evaluation/Related items.md)

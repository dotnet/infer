---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Learner API](../../Matchbox recommender/Learner API.md)

## Serialization

Both trained and untrained models can be serialized and deserialized.

Serialization is achieved through the Save extension method on the `ILearner` interface. There are two overloads of this method - one that takes in a file, and one that takes in a stream and a formatter. The former internally calls the latter:
```csharp
recommender.Save("Model.bin");
```
Deserialization is done using the `MatchboxRecommender.Load` static method. It also has two overloads which correspond to the ones of the Save method. In addition, Load requires 4 generic parameters to be specified: the type of an instance source, the type of a user, the type of an item, and the type of a feature source.
```csharp
var recommender =
  MatchboxRecommender.Load<StandardDataset, User, Item, FeatureSource>("Model.bin");
```
When a model is deserialized the generic parameters have to match exactly the ones of the recommender that had been serialized. Also, during deserialization a version check will be performed - in case the serialized learner does not match the target one.

Note that only the parameters of the model and the user-defined mapping are persisted. None of the training data will be serialized.

---
layout: default
---
[Learners](../../../../Infer.NET Learners.md) : [Matchbox recommender](../../../Matchbox recommender.md) : [Learner API](../../../Matchbox recommender/Learner API.md) : [Data mappings](../Data mappings.md)

## Standard data format mapping

The standard data format mapping is declared in the `IStarRatingRecommenderMapping` interface. It contains seven methods.

```csharp
IEnumerable<TInstance> GetInstances(TInstanceSource instanceSource);  
TUser GetUser(TInstanceSource instanceSource, TInstance instance);  
TItem GetItem(TInstanceSource instanceSource, TInstance instance);  
TRating GetRating(TInstanceSource instanceSource, TInstance instance); 
```

These four methods deal with the instance data. `GetInstances` returns the collection of instances, and then `GetUser`, `GetItem`, and `GetRating` know how to extract the user, item, and rating respectively from the corresponding instance. These methods are all called during training, and all of them except `GetRating` are called during bulk rating prediction. Note that `GetUser`, `GetItem`, and `GetRating` take as inputs not only the instance itself, but also an instance source. This is because one can choose to implement a mapping where the instance is a pointer in an instance source; for example, an index to a row in a collection. In the other case, when the instance actually contains the user, item and rating, then the instance source is not used in the implementation of these methods. 

```csharp
TFeatureValues GetUserFeatures(TFeatureSource featureSource, TUser user);
TFeatureValues GetItemFeatures(TFeatureSource featureSource, TItem item);
```

The `GetUserFeatures` and `GetItemFeatures` methods are optional and can be used when user or item features are available. These methods are meant to be generic and not bound to Infer.NET, so the return types were left open. However, in order to be able to use them with the Matchbox Recommender, you have to bind the return type to the Infer.NET Vector type (`Microsoft.ML.Probabilistic.Math.Vector`) which supports both sparse and dense data. Similar to the instance-related methods, the feature-related ones accept a feature source that can be useful if the user or item points into it, or otherwise omitted when the user or item object contains the features internally.

```csharp
IStarRatingInfo<TRating> GetRatingInfo(TInstanceSource instanceSource);
```

The `GetRatingInfo` method is used to convert the ratings from the data domain into the model domain. If your ratings are already consecutive integers, you can return an instance of `StarRatingInfo`. This class implements the `IStarRatingInfo` interface, and its constructor takes two integers - the minimum and the maximum rating values. If you need to convert the ratings into the model domain on the fly, simply provide your own implementation of `IStarRatingInfo`. It has to define the `MinStarRating` and `MaxStarRating` properties, as well as the `ToStarRating` method, which takes in a generic rating value and returns an integer. You might need to manually implement `IStarRatingInfo` when the ratings are floating-point numbers, for example.

Here is an example of a standard format data mapping implementation:

```csharp
[Serializable]
public  class  StandardRecommenderTestMapping :  IStarRatingRecommenderMapping <StandardDataset, Tuple<User, Item, int?>, User, Item, int, FeatureProvider, Vector>
{
  public  IEnumerable<Tuple<User, Item, int?>> GetInstances(StandardDataset instanceSource)
  { return instanceSource.Observations; }

  public  User GetUser(StandardDataset instanceSource, Tuple<User, Item, int?> instance)
  { return instance.Item1; }

  public  Item GetItem(StandardDataset instanceSource, Tuple<User, Item, int?> instance)
  { return instance.Item2; }

  public  int GetRating(StandardDataset instanceSource, Tuple<User, Item, int?> instance)
  {
    if (instance.Item3 == null)
    {
      throw  new  ArgumentException( "Rating is not contained in the given instance", "instance");
    }
    return instance.Item3.Value;
  }

  public  virtual  Vector GetUserFeatures(FeatureProvider featureSource, User user)
  { return featureSource.UserFeatures[user]; }

  public  virtual  Vector GetItemFeatures(FeatureProvider featureSource, Item item)
  { return featureSource.ItemFeatures[item]; }

  public  IStarRatingInfo<int> GetRatingInfo(StandardDataset instanceSource)
  { return  new  StarRatingInfo(0, 5); }
}
```

Also, here are the type definitions used in this mapping:

```csharp
public  class  StandardDataset
{
  public  List<Tuple<User, Item, int?>> Observations { get; set; }
}

public  class  FeatureProvider
{
  public  IDictionary<User, Vector> UserFeatures { get; set; }
  public  IDictionary<Item, Vector> ItemFeatures { get; set; }
}
```

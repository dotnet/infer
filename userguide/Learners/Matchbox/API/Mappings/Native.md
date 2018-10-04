---
layout: default
---
[Learners](../../../../Infer.NET Learners.md) : [Matchbox recommender](../../../Matchbox recommender.md) : [Learner API](../../../Matchbox recommender/Learner API.md) : [Data mappings](../Data mappings.md)

## Native data format mapping

The native data format mapping provides the recommender with data which is ready to be set as observed values for the model variables. It is declared in the `IMatchboxRecommenderMapping` interface and contains fourteen methods, eight of which are optional.

```csharp
IList<int> GetUserIds(NativeDataset instanceSource, int batchNumber = 0);  
IList<int> GetItemIds(NativeDataset instanceSource, int batchNumber = 0);  
IList<int> GetRatings(NativeDataset instanceSource, int batchNumber = 0);
```

These three methods provide the recommender with the data instances. All users, items and ratings have to be zero-based consecutive integer numbers. If the data is split into batches, then the corresponding batch has to be returned. Batching is explained in the modelling section and the settings.

```csharp
int GetUserCount(NativeDataset instanceSource);  
int GetItemCount(NativeDataset instanceSource);  
int GetRatingCount(NativeDataset instanceSource);
```

The three count methods return the user, item and rating counts. Note that ratings are assumed to start from zero, so the rating count is equal to the maximum rating value plus one.

```csharp
IList<IList<double>> GetAllUserNonZeroFeatureValues(NativeDataset featureSource);  
IList<IList<int>> GetAllUserNonZeroFeatureIndices(NativeDataset featureSource);  
IList<IList<double>> GetAllItemNonZeroFeatureValues(NativeDataset featureSource);  
IList<IList<int>> GetAllItemNonZeroFeatureIndices(NativeDataset featureSource);  
IList<double> GetSingleUserNonZeroFeatureValues(NativeDataset featureSource, int userId);  
IList<int> GetSingleUserNonZeroFeatureIndices(NativeDataset featureSource, int userId);  
IList<double> GetSingleItemNonZeroFeatureValues(NativeDataset featureSource, int itemId);  
IList<int> GetSingleItemNonZeroFeatureIndices(NativeDataset featureSource, int itemId);
```

These eight methods deal with the features and are therefore optional. The recommender always accepts the features in a sparse format, so these methods come in pairs - one that provides the features values and one that provides the feature indices. The first four methods are used in training. The list of lists represents the features for each user or item. The last four methods are used in prediction. Prediction is performed on per-user or- item basis, and therefore the corresponding user or item needs to be specified.

Here is a sample implementation of a native data format mapping:

```csharp
[Serializable]
private  class  NativeRecommenderTestMapping : IMatchboxRecommenderMapping<NativeDataset, NativeDataset>
{
  public  IList<int> GetUserIds(NativeDataset instanceSource, int batchNumber = 0)
  {
    return instanceSource.UserIds[batchNumber];
  }
  public  IList<int> GetItemIds(NativeDataset instanceSource, int batchNumber = 0)
  {
    return instanceSource.ItemIds[batchNumber];
  }
  public  IList<int> GetRatings(NativeDataset instanceSource, int batchNumber = 0)
  {
    return instanceSource.Ratings[batchNumber];
  }
  public  int GetUserCount(NativeDataset instanceSource)
  {
    return instanceSource.UserCount;
  }
  public  int GetItemCount(NativeDataset instanceSource)
  {
    return instanceSource.ItemCount;
  }
  public  int GetRatingCount(NativeDataset instanceSource)
  {
    return 6; // Rating values are from 0 to 5
  }
  public  IList<IList<double>> GetAllUserNonZeroFeatureValues(NativeDataset featureSource)
  {
    return featureSource.NonZeroUserFeatureValues;
  }
  public  IList<IList<int>> GetAllUserNonZeroFeatureIndices(NativeDataset featureSource)
  {
    return featureSource.NonZeroUserFeatureIndices;
  }
  public  IList<IList<double>> GetAllItemNonZeroFeatureValues(NativeDataset featureSource)
  {
    return featureSource.NonZeroItemFeatureValues;
  }
  public  IList<IList<int>> GetAllItemNonZeroFeatureIndices(NativeDataset featureSource)
  {
    return featureSource.NonZeroItemFeatureIndices;
  }
  public  IList<double> GetSingleUserNonZeroFeatureValues( NativeDataset featureSource, int userId)
  {
    return featureSource.NonZeroUserFeatureValues[userId];
  }
  public  IList<int> GetSingleUserNonZeroFeatureIndices( NativeDataset featureSource, int userId)
  {
    return featureSource.NonZeroUserFeatureIndices[userId];
  }
  public  IList<double> GetSingleItemNonZeroFeatureValues( NativeDataset featureSource, int itemId)
  {
    return featureSource.NonZeroItemFeatureValues[itemId];
  }
  public  IList<int> GetSingleItemNonZeroFeatureIndices( NativeDataset featureSource, int itemId)
  {
    return featureSource.NonZeroItemFeatureIndices[itemId];
  }
}
```

Note that the instances and the features can be stored in different classes - the two generic parameters of the mapping. But we simplified this example by storing them in the same class:

```csharp
public  class  NativeDataset
{
  public  int[][] UserIds { get; set; } // For each data batch
  public  int[][] ItemIds { get; set; } // For each data batch
  public  int[][] Ratings { get; set; } // For each data batch
  public  int UserCount { get; set; }
  public  int ItemCount { get; set; }
  public  double[][] NonZeroUserFeatureValues { get; set; } // For each user
  public  int[][] NonZeroUserFeatureIndices { get; set; } // For each user
  public  double[][] NonZeroItemFeatureValues { get; set; } // For each item
  public  int[][] NonZeroItemFeatureIndices { get; set; } // For each item
}
```

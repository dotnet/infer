---
layout: default
---

[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Learner API](../../Matchbox recommender/Learner API.md)

## Training

After a recommender is instantiated and set up, one can train it by calling the train method and passing the sources of instances and optionally features:

```csharp
recommender.Train(instanceSource, featureSource);
```

At this point the recommender will call the data mapping methods to obtain the training data. If you used a standard data format mapping to instantiate the recommender, then the `GetInstances` method will be called only once. This is regardless of the value of the `BatchCount` setting, because the training instances get cached in a native format. If you used a native data format mapping instead, then the `GetUserIds`, `GetItemIds`, and `GetRatings` methods will be called for each training iteration and for each batch.

The recommender currently cannot be retrained. That is, online or incremental training is not yet supported. Therefore, if the Train method is called a second time, it will throw an `InvalidOperationException`.

Once the recommender has been trained, one can examine the learned distributions over the model parameters. These can be obtained in the following way:
```csharp
recommender.GetPosteriorDistributions();
```
### Batched training

When the number of batches is set to more than 1, the so-called _batched training_ gets activated. Data is loaded into memory in portions (batches) in order to reduce the memory consumption. The batched training works as follows:

```
For each iteration:  
    For each data batch:  
        Load the data for the current batch into memory  
        Update the model parameters for the current batch from the loaded data  
    Update the shared model parameters by aggregating the parameters from all batches
```

Note that although the instance data is loaded to memory in batches, and inference is run on limited portions of data, the model parameters are copied in memory for each batch. The model parameters are dominated by the user and item traits. 

As an aside, bulk _prediction_ is never batched. Therefore, during prediction the `BatchCount` setting is implicitly set to 1, and only batch 0 is requested from the native data format mapping.​​

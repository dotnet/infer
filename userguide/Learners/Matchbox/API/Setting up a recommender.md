---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Learner API](../../Matchbox recommender/Learner API.md)

## Setting up a recommender

Once you have an instance of a data mapping, creating a recommender is as easy as calling one factory method: `MatchboxRecommender.Create`. This method has two overloads which take either a native data format mapping (an implementation of `IMatchboxRecommenderMapping`) or a standard data format mapping (an implementation of `IStarRatingRecommenderMapping`). Both creation routines are very lightweight - no data access or computations is performed at this stage. The return type of the factory is `IMatchboxRecommender`, which provides concrete settings and means to perform training and prediction.

```csharp
var dataMapping = new  DataMapping();  
var recommender = MatchboxRecommender.Create(dataMapping);
```

The settings are obtained through the `Settings` property of the newly instantiated recommender. They can refer either to the training algorithm, or the prediction one.

Training settings

*   **UseUserFeatures** specifies whether user features will be used. If this setting is set to true, then the user-related feature methods of the data mapping will be called during training. This should generally improve predictions and in particular cold user predictions. 

*   **UseItemFeatures** specifies whether item features will be used. If this setting is set to true, then the item-related feature methods of the data mapping will be called during training. This should generally improve predictions and in particular cold item predictions. 

*   **TraitCount** sets the number of traits to be used. This is the dimension of the latent space to which users and items are mapped to. More informally, it is the number of implicitly learned characteristics of the users and items. An intuition to the meaning of traits is given in the [Introduction](../../Matchbox recommender/Introduction.md), and more detailed explanation is given in the [Model](../../Matchbox recommender/Model.md) section of this documentation. The number of traits typically varies between 2 and 20, and is set to 4 by default. The way to tune this parameter is by measuring accuracy on a train-test split of the data.

*   **BatchCount** sets the number of batches that the training instance data is split into. It is set to 1 by default, but it should be increased if the system runs out of memory when training. Note that this only splits the instance data and not the feature data (as the features do no need to be split). If you use a standard data format mapping, then the data batching will be automatically performed by the recommender, but if you use a native data format mapping, you are expected to have the instance data split into batches prior to training. More about how batched training works is given in the [Training](../Training.md) section.

*   **IterationCount** sets the number of inference algorithm iterations to be run during training. Matchbox uses a mixture of the Expectation Propagation and Variational Message Passing approximate inference algorithms. The value of this parameter is set to 20 by default, but it strongly depends on the data. The way to tune it is by measuring accuracy on a train-test split of the data. Note that increasing the number of iterations is expected to increase the predictive accuracy, but also increase the training time.

*   Under Settings.Training there is the **Advanced** subsection, where one can set the prior variances of some random variables in the model. These include the user and item traits, the user and item biases, the user and item features, the user thresholds, as well as the noise of the affinity and the user thresholds. These are explained in detail in the [Model](../../Matchbox recommender/Model.md) section. The values of the prior variances are typically between 0 and 10. These need to be carefully set for good prediction results. Since they strongly depend on the data, we recommend tuning them by measuring accuracy on a train-test split of the data.

Prediction settings

**SetPredictionLossFunction** sets the loss function to be used during rating prediction for a given user-item pair when converting a probability distribution over the ratings into a point estimate. There are four options here.

    *   **ZeroOne** loss function is equivalent to choosing the mode of the predicted distribution as a point estimate. Use this loss function to minimize mean classification error.
    *   **Squared** (or quadratic) loss function is equivalent to choosing the mean of the predicted distribution as a point estimate. Use this loss function to minimize mean squared error.
    *   **Absolute** loss function is equivalent to choosing the median of the predicted distribution as a point estimate. Use this loss function to minimize mean absolute error.
    *   **Custom** loss function can be specified if none of the above loss functions satisfy your specific needs.

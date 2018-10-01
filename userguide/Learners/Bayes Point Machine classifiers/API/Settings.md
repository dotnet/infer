---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [The Learner API](../API.md)

## Settings & events

The settings of a Bayes Point Machine classifiers are obtained through its `Settings` property. There are separate settings for training and prediction.

### Training settings

*   `IterationCount` sets the number of training algorithm iterations to be run. The number of training iterations specifies how often the message-passing algorithm ([Expectation Propagation](http://research.microsoft.com/en-us/um/people/minka/papers/ep/) in the case of the Bayes Point Machine classifiers) iterates over the (entire) training data. The higher the number of training iterations, the more accurate the predictions; however, training will be slower. 

    For most data sets, the default setting of 30 training iterations is sufficient for the algorithm to make accurate predictions. Sometimes accurate predictions can be made using fewer iterations. On data sets with highly correlated features, one may benefit from more training iterations. Typically, the number of iterations varies between 5 and 100. 


*   `BatchCount` sets the number of batches that the training data is split into. It is set to 1 by default, but should be increased if the system runs out of memory during training. Batching allows to train the Bayes Point Machine classifier even if the training data does not fit into memory! If a [standard data format mapping](Mappings/Standard Data Format Mapping.md) is used to read data, then the batching will be automatically performed by the Bayes Point Machine classifiers; but if you use a [native data format mapping](Mappings/Native Data Format Mapping.md), you are expected to have split the data into batches prior to training. 

    The computational cost of batching increases markedly with the number of batches because the Bayes Point Machine classifiers needs to switch the batches in and out of memory as well as reconstruct messages for each batch. Batching should hence be avoided, if possible. 


*   `ComputeModelEvidence` determines whether or not the Bayes Point Machine classifier can compute model evidence (marginal likelihood), a measure of how likely the data is given the model. Model evidence is hence used in model comparison. As the computation of evidence decreases performance, `ComputeModelEvidence` is set to `false` by default. In contrast to the other settings, it cannot be changed once a Bayes Point Machine classifier has been trained. 



It is important to realize that the Bayes Point Machine classifiers do not require the specification of some parameters, such as prior distributions over weights. The Bayes Point Machines are practically _hyper-parameter free_. As mentioned in the tutorial introduction, this is achieved via heavy-tailed prior distributions over weights (see [The probabilistic model](../Modelling.md)) and has a number of immediate benefits:

*   It avoids a misspecification of the prior distributions (which typically results in suboptimal performance).
*   It entirely frees the Bayes Point Machine classifiers from the need for time-consuming hyper-parameter tuning.
*   It also avoids the need for feature values to be normalized in a pre-processing step, as the Bayes Point Machine classifiers automatically adapt to the different feature scales during training.

### Prediction settings

*   `SetPredictionLossFunction` sets the loss function based on which a precise classification (predictive point estimate) is computed from the predicted distributions over class labels (see [Prediction](Prediction.md)). There are four options:  

    *  `ZeroOne` \- the 0/1 loss function is equivalent to choosing the mode of the predicted distribution as a point estimate. Use this loss function to minimize mean classification error. `ZeroOne` is the _default_ loss function.
    *   `Squared` \- the squared or quadratic loss function is equivalent to choosing the mean of the predicted distribution as a point estimate. Use this loss function to minimize mean squared error on class labels which have an inherent ordering. Note, however, that such an ordering violates the assumptions underlying the Bayes Point Machine classifiers (more [here](../Modelling.md)). Moreover, the `Squared` option is available only if labels are supplied via a native data format mapping. Otherwise you need to specify a custom squared loss function.
    *   `Absolute` \- the absolute loss function is equivalent to choosing the median of the predicted distribution as a point estimate. Use this loss function to minimize mean absolute error on class labels which have an inherent ordering. Note, however, that such an ordering violates the assumptions underlying the Bayes Point Machine classifiers (more [here](../Modelling.md)). Moreover, the `Absolute` option is available only if labels are supplied via a native data format mapping. Otherwise you need to specify a custom absolute loss function.
    *   `Custom` \- a custom loss function can be specified if none of the above loss functions satisfy your specific needs (in medical diagnosis, for instance, classification loss is often judged to be asymmetric). Using a custom loss function may slow down point prediction (i.e. the Predict method).

The following prediction setting only applies to the _multi-class_ Bayes Point Machine classifier:

*   `IterationCount` sets the number of prediction algorithm iterations to be run. In multi-class classification, the message-passing algorithm ([Expectation Propagation](http://research.microsoft.com/en-us/um/people/minka/papers/ep/)) needs to iterate a small number of times over the prediction data. Typically, this number varies between 2 and 10. As in the case of the training algorithm, the higher the number of prediction iterations, the more accurate the predictions; however, prediction will be slower.

### Events

The Bayes Point Machine classifiers can provide (limited) diagnostic information about the progress of the training algorithm via the `IterationChanged` event. Listening to this event gives you access to the posterior distributions over weights after the training algorithm has completed an iteration over the training set. Especially if there are many features (and thus weights) or classes, attaching a handler to this event comes at a cost as the weights need to be copied. However, in rare cases the Bayes Point Machine classifiers may exhibit relatively poor performance. In such a case, it is often informative to look at the change in posterior weight distributions over time and see whether there is a problem with convergence of the training algorithm.

Handlers attached to the `IterationChanged` event will _not_ be serialized when saving the Bayes Point Machine classifiers.

---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Distributed inference

The techniques described in the [Online learning](Online learning.md) document can also be used to perform inference over a cluster of machines. Each machine runs the same model, but with different data and different incoming messages, then they exchange messages. The steps are as follows:

1. Determine how you want to partition the model, and identify the set of variables shared among multiple partitions.
2. For each shared variable, create a variable to hold incoming messages, and attach it via _ConstrainEqualRandom_. We will call this the _inbox_. Each machine will have a different value for the inbox. Initially, the inboxes are all uniform.
3. Each machine runs inference, obtaining the MarginalDividedByPrior for each shared variable, and divides it by the inbox. We will call this the upward _message_. If machines are indexed by m, then we will get an array of messages indexed by m.
4. On the next iteration, the inbox for a variable on machine m is equal to the product of messages from all other machines.
5. At convergence, the posterior for the variable is its marginal computed by any of the machines (they should all be equal).

An example of this procedure can be found in the Infer.NET distribution under Learners. The paths to the batched training methods for the classifier and recommender are:

*   Classifier\\BayesPointMachineClassifierInternal\\Classifiers\\NativeDataFormatBayesPointMachineClassifier.cs -> TrainOnMultipleBatches()
*   Recommender\\MatchboxRecommenderInternal\\NativeDataFormatMatchboxRecommender.cs -> Train()

To divide out the inbox in step 3, we can use a clever modelling trick instead of doing the division manually. The trick is to make a copy of the shared variable and infer the MarginalDividedByPrior of the copy. Since the inbox is part of the prior of the copy, it will be divided out for free. Here is an example: 

```csharp
Variable<double> weight = Variable.GaussianFromMeanAndVariance(0, 1);  
Variable<Gaussian> weightInbox = Variable.Observed(new Gaussian(3,4));  
Variable.ConstrainEqualRandom(weight, weightInbox);  
Variable<double> weightCopy = Variable.Copy(weight);  
weightCopy.AddAttribute(QueryTypes.MarginalDividedByPrior);  
InferenceEngine engine = new InferenceEngine();  
var message = engine.Infer<Gaussian>(weightCopy, QueryTypes.MarginalDividedByPrior);
```

This method is recommended since it not only avoids the cost of doing the division but also potential numerical inaccuracies due to round-off errors. (The cost of copying a variable in Infer.NET is negligible since the compiler will optimize it away in the generated code.)
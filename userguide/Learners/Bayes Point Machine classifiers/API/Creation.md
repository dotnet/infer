---
layout: default 
--- 
[Infer.NET user guide ](../../../index.md) : [Learners ](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers ](../../Bayes Point Machine classifiers.md) : [The Learner API ](../API.md)

## Creation & serialization

### Creating a Bayes Point Machine classifier

Once you have an instance of a [mapping ](Mappings.md), creating a Bayes Point Machine classifier is as easy as calling one of the following two factory methods, depending on the type of classification problem:

*   `BayesPointMachineClassifier.CreateBinaryClassifier` (for problems with _two_ classes)
*   `BayesPointMachineClassifier.CreateMulticlassClassifier` (for problems with _three or more_ classes)

Moreover, there are two versions for both methods, one which takes as input a mapping to the native data format (an implementation of `IBayesPointMachineClassifierMapping`) and one which takes as input a mapping to the standard data format (an implementation of `IClassifierMapping`). The factory methods return classifiers of type `IBayesPointMachineClassifier`, providing appropriate [settings ](Settings.md) and means for [training ](Training.md) and [prediction ](Prediction.md).

For instance, let's imagine we have implemented a mapping from some SQL database to the native format of the Bayes Point Machine classifier, called `SqlNativeMapping`. To create a classifier for a multi-class problem, we then simply write

```csharp
var mapping = new SqlNativeMapping();  
var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);
```

Creating a Bayes Point Machine classifier is a lightweight operation - no data are accessed and no computations are performed.

### Serialization

Serialization is achieved through the Save extension method on the `ILearner` interface. There are two overloads of this method - one that takes in a file, and one that takes in a stream and a formatter. Both trained and untrained classifiers can be serialized (and deserialized).

To save a Bayes Point Machine classifier to file, for instance, we write

```csharp
classifier.Save("bpm.bin");
```

This serializes the parameters of the classifier and the user-defined mapping. It does _not_ serialize any training data nor any attached event handler.

### Deserialization

By calling one of the static `Load` methods of `BayesPointMachineClassifier`, it is possible to load a previously serialized classifier. Just like the `Save` method, `Load` can deserialize a classifier from a file or a stream (and a formatter). Here is how to load a classifier from a file:

```csharp
var classifier = BayesPointMachineClassifier.Load  
        <SqlInstanceSource,   
        Instance,   
        SqlLabelSource,   
        int,   
        Discrete,     
        BayesPointMachineClassifierTrainingSettings,  
        MulticlassBayesPointMachineClassifierPredictionSettings<int>>("bpm.bin");
```

The generic types that this method requires are

*   `TInstanceSource` \- the type of the instance source;
*   `TInstance` \- the type of a single instance;
*   `TLabelSource` \- the type of the label source;
*   `TLabel` \- the type of a label (in native data format `bool` for a binary Bayes Point Machine classifier and `int` for a multi-class Bayes Point Machine classifier);
*   `TLabelDistribution` \- the type of the label distribution (`Bernoulli` or `Discrete` if a native data format mapping is used);
*   `TTrainingSettings` \- the type of the classifier's training settings (`BayesPointMachineClassifierTrainingSettings` in case of the Bayes Point Machine classifier);
*  `TPredictionSettings` \- the type of the classifier's prediction settings (`BinaryBayesPointMachineClassifierPredictionSettings` or `MulticlassBayesPointMachineClassifierPredictionSettings` for the respective Bayes Point Machine classifiers).

When a classifier is deserialized the generic parameters have to exactly match the ones used to serialize the classifier. Also, during deserialization a _version check_ will be performed. Should the version of the serialized learner not match the version of the target learner, an exception will be thrown.

Then, for convenience, a number of methods exist which are more specific than Load, and which decrease the number of generic types that need to be specified. There is, for instance,

```csharp
var classifier = BayesPointMachineClassifier.LoadMulticlassClassifier  
        <SqlInstanceSource, Instance, SqlLabelSource, int, Discrete>("bpm.bin");
```

which equally deserializes the previously saved multi-class Bayes Point Machine classifier from file.

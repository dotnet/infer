---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [The Learner API](../API.md)

## Training

Once a Bayes Point Machine classifier is instantiated and set up, it is trivial to train. Let's imagine your training data lives in an object called `trainingSet` of type `SqlInstanceSource` and provides both features and ground truth labels. Let's further imagine you have already implemented the [native data format mapping](Mappings/Native Data Format Mapping.md) for `SqlInstanceSource`. Then, to train the Bayes Point Machine classifier on `trainingSet`, we write

```csharp
classifier.Train(trainingSet);
```

Sometimes features and ground truth labels live in different objects. Let's say the labels are given by an instance of `SqlLabelSource` called `trainingSetLabels` and `SqlInstanceSource` only provides the corresponding features (via `trainingSetFeatures`). Given the appropriate mapping for `SqlInstanceSource` and `SqlLabelSource`, training looks as follows:

```csharp
classifier.Train(trainingSetFeatures, trainingSetLabels);
```

At this point, the classifier will call the mapping methods to obtain the instances, features and labels of the training set. Then, it will create and run the Infer.NET message-passing algorithm. At this stage, the Infer.NET compiler would typically generate the training algorithm from the underlying probabilistic model. In the Bayes Point Machine classifiers, however, all inference algorithms are [_precompiled_](../../../Using a precompiled inference algorithm.md) for fast execution of both training and prediction.

### Incremental training

Very often, new training data becomes available only after one has trained a classifier. Often, this happens repeatedly, meaning the classifier needs to be trained again and again. The naive approach to this problem is to merge all available training data and train a classifier on larger and larger training sets. Such an approach is very costly, of course.

Alternatively, one may train the classifier _incrementally_. Incremental training sequentially updates the classifier, using only the newly available information. It hence is a very efficient way to learn as it does not require to revisit previously seen data. In fact, this is a simple consequence of Bayes's theorem, which essentially tells us how to update our belief about a quantity of interest in the light of evidence.

More concretely, you call

```csharp
classifier.TrainIncremental(otherTrainingSet);
```

to incrementally train a Bayes Point Machine classifier on a new training set `otherTrainingSet`. Obviously, you can repeat this process every time new labelled data becomes available. The only caveat is that the feature set is determined by the [mapping](Mappings.md) and cannot be changed during training. This means that features which were unknown when the classifier was created will be _ignored_ later on during incremental training.

### Batched training

Batched training allows learning from data which is split into batches, mutually exclusive and collectively exhaustive subsets of training set instances. Since it is possible to learn from a batch of instances at a time, batching limits memory consumption and makes it possible to learn from very large datasets.

In contrast to incremental training, which does not revisit training sets, batched training assumes the availability of all batches at training time and cycles through all of them in each iteration of the training algorithm. While this guarantees that batched and conventional training arrive at precisely the same results, this comes at a price: Batches usually need to be moved in and out of memory, which is computationally costly. In addition, the training algorithms of the Bayes Point Machine classifiers need to reconstruct messages every time a batch is switched, which is often even more expensive. For these reasons it is recommended to avoid batched training, if possible.

To switch to batched training, simply change the default value of the BatchCount [setting](Settings.md) prior to training (or incremental training). To disable batched training, set BatchCount back to 1.

Of course, it is possible change both the IterationCount and the BatchCount [settings](Settings.md) between individual training runs. This means you can run 20 iterations of batched incremental training after 100 iterations of non-batched incremental training, say!

### Evidence computation

The [evidence](../../../Computing model evidence for model selection.md) or marginal likelihood is a measure of how likely the data is under the given the model. Model evidence is hence extensively used to compare or [select models](http://alumni.media.mit.edu/~tpminka/statlearn/demo/). In contrast to other approaches, evidence computation does not require a hold-out set for validation purposes, but can be computed for the training set.

To compute model evidence, you need to activate the ComputeModelEvidence [setting](Settings.md) prior to training:

```csharp
classifier.Settings.Training.ComputeModelEvidence = true;
```

Then, after training,

```csharp
double logEvidence = classifier.LogModelEvidence;
```

will get you the natural logarithm of the model evidence (for the specified training set). Note that model evidence can only be computed for the first training run and not during incremental training. It can be computed during batched training, however.

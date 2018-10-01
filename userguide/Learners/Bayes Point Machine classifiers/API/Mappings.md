---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [The Learner API](../API.md)

## Mappings

Data comes in a variety of different formats and types. For example, it may live in memory as objects, on disk as files or on a server as database records. To access all different kinds of data without introducing a lot of friction, the Infer.NET learners provide what is called a _mapping_. A mapping is a mechanism which allows the user to decide how to best deliver given data to a learner, providing the user with more flexibility than a fixed set of strongly-typed input arguments. Often, this avoids unnecessary or costly data conversions.

A mapping is an implementation of an interface which defines how a learner accesses input data to train and make predictions. The Bayes Point Machine classifier comes with two mapping interfaces, `IClassifierMapping` and `IBayesPointMachineClassifierMapping`, which serve distinct purposes. Whereas the former interface, `IClassifierMapping`, is easy to implement and use, but incurs a computational cost as the data ultimately needs to be converted, the latter interface does not require a data conversion and thus is computationally cheaper, but is slightly more work to implement.

The reason for this is that `IBayesPointMachineClassifierMapping` guarantees that data is supplied to the learner in the format that is _native_ to its underlying Infer.NET inference algorithms. Hence, when implementing this interface, the learner does not need to make any further conversion to the data provided by the mapping to run the internal algorithms. On the other hand, implementing the `IBayesPointMachineClassifierMapping` interface is more involved than an implementation of `IClassifierMapping` as more details need to be taken care of. Also, the native format is not very widely used, meaning original data is rarely available in the format expected by the learner (and `IBayesPointMachineClassifierMapping`). In many situations the mapping still provides computational benefits over strongly-typed inputs.

The format required by `IClassifierMapping`, referred to as the _standard_ format, is more general than the native format, often allowing for trivial implementations of `IClassifierMapping`. Internally, however, the classifier will eventually have to convert the standard format data into the native format understood by the inference algorithms. In fact, when you choose to implement the `IClassifierMapping` interface to deliver data to the Bayes Point Machine classifier, it will be wrapped using an implementation of `IBayesPointMachineClassifierMapping`. Thus, for ultimate performance, it is recommended to directly implement the latter interface. In most scenarios, however, it is perfectly sufficient to implement `IClassifierMapping`.

In the following subsections we shall briefly describe the three mapping interfaces used in classification:

*   [IClassifierMapping](Mappings/Standard Data Format Mapping.md) \- providing classifiers with data in standard format,
*   [IBayesPointMachineClassifierMapping](Mappings/Native Data Format Mapping.md) \- providing the Bayes Point Machine classifiers with data in their native format, and
*   [IClassifierEvaluatorMapping](Mappings/Evaluation Data Format Mapping.md) \- providing evaluators with data in standard format.

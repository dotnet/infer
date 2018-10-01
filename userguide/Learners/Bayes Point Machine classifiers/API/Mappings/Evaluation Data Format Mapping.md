---
layout: default 
--- 
[Infer.NET user guide](../../../../index.md) : [Learners](../../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../../Bayes Point Machine classifiers.md) : [The Learner API](../../API.md) : [Mappings](../Mappings.md)

## Evaluation Data Format Mapping

An easy way to assess the performance of a classifier is to use an [evaluator](../Evaluation.md). The evaluator reads the ground truth labels for some instances of interest (validation or test set) via a mapping which implements the `IClassifierEvaluatorMapping` interface. Since an evaluator should be independent of the concrete data formats required by specific classifier implementations such as the Bayes Point Machine, `IClassifierEvaluatorMapping` essentially declares the generic [standard data format mapping](Standard Data Format Mapping.md) of the `IClassifierMapping` interface, just without the `GetFeatures` method. Predictions are input arguments to the evaluation methods and do not get accessed via the mapping.

A concrete implementation of the `IClassifierEvaluatorMapping` interface can be defined based on a given standard data format mapping. In fact, there is an extension method, `ForEvaluation`, to do just that. It takes the given standard data format mapping and returns the corresponding classifier evaluator mapping, essentially producing what is called a _chained_ mapping.

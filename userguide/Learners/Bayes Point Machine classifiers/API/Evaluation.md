---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [The Learner API](../API.md)

## Evaluation

### Creating an evaluator

One simple way to assess the quality of the predictions of a classifier against some corresponding ground truth labels is to use an evaluator, an instance of the `ClassifierEvaluator` class that is part of Infer.NET. An evaluator provides a number of popular metrics to measure classification performance (see below) and can be applied to quantify the quality of predictions from any classifier, not just the Bayes Point Machine.

Like the Infer.NET learners, an evaluator accesses the ground truth labels through a mapping (more precisely, an implementation of the `IClassifierEvaluatorMapping` interface, as described [earlier](Mappings.md)). Predictions, on the other hand, are directly consumed - either as distributions or as point estimates.

Once we have such an evaluator mapping (it only requires the implementation of three methods: `GetInstances`, `GetLabel`, and `GetClassLabels`), we can instantiate a classifier evaluator by calling into its constructor. To do so, four type parameters need to be specified:

*   `TInstanceSource` \- the type of the instance source,
*   `TInstance` \- the type of an instance,
*   `TLabelSource` \- the type of the label source, and
*   `TLabel` \- the type of a label.

Clearly, to compare classifier predictions with ground truth labels provided by the evaluator mapping, `TLabel` needs to match the type that is used by the classifier for predictions. For instance, to utilize an evaluator for predictions from a Bayes Point Machine classifier which uses a standard data format mapping to read labels of type string, it is necessary for `TLabel` to be bound to string.

Here is an example of how to create an evaluator from a standard data format mapping:

```csharp
var classifierMapping = new ClassifierMapping();  
var evaluatorMapping = classifierMapping.ForEvaluation();  
var evaluator = new ClassifierEvaluator  
    <IEnumerable<Instance>,   
    Instance,   
    IEnumerable<Instance>,   
    string>(evaluatorMapping);
```

Without a standard data format mapping, you cannot use the `ForEvaluation` extension method and must instead implement the three methods declared by `IClassifierEvaluatorMapping`.

### Available performance metrics

`ClassifierEvaluator` provides a number of metrics to quantify the performance of a classifier on some given validation or test set, including:

*   Classification error and accuracy
*   Precision
*   Recall
*   F1-measure (the harmonic mean of precision and recall)
*   Confusion matrix
*   Logarithmic loss
*   ROC - receiver operating characteristic curve
*   AUC - area under the ROC curve
*   Precision-recall (PR) curve
*   Empirical probability calibration curve

Classification error and logarithmic loss, for instance, can be computed using the `Evaluate` method:

```csharp
var errorCount = evaluator.Evaluate(  
    groundTruth, predictedLabels, Metrics.ZeroOneError);
```

```csharp
var logLoss = evaluator.Evaluate(  
    groundTruth, uncertainPredictions, Metrics.NegativeLogProbability);
```

As you can see, it is possible to evaluate both predictive distributions (`uncertainPredictions`) as well as point estimates (`predictedLabels`). Evaluate also allows to implement and use your own classification metric of choice.

Many metrics are also available from the confusion matrix:

```csharp
ConfusionMatrix confusionMatrix =   
    evaluator.ConfusionMatrix(groundTruth, predictedLabels);  

double accuracy = confusionMatrix.MicroAccuracy;  
double macroF1 = confusionMatrix.MacroF1;
```

Moreover, to get ROC, PR and calibration curves, respectively, you write:

```csharp
var rocCurve = evaluator.ReceiverOperatingCharacteristicCurve(  
    positiveClassLabel, groundTruth, uncertainPredictions);  

var precisionRecallCurve = evaluator.PrecisionRecallCurve(  
    positiveClassLabel, groundTruth, uncertainPredictions);  

var calibrationCurve = evaluator.CalibrationCurve(  
    positiveClassLabel, groundTruth, uncertainPredictions);
```

where positiveClassLabel is the label of the class for which we wish to produce the curve. AUC can be computed the same way:

```csharp
var auc = evaluator.AreaUnderRocCurve(  
    positiveClassLabel, groundTruth, uncertainPredictions);
```

and matches the ROC curve (given identical inputs).

Note that some of these metrics are natural only when evaluating binary classification problems, even though it is always possible to extend them to multi-class settings using one-vs-all or one-vs-another approaches. Note further that an imbalanced number of instances per class may distort some metrics (accuracy, for example).

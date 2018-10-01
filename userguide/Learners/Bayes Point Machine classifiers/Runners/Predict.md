---
layout: default
---
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [Command-line runners](../Runners.md)

## Prediction

Predictions can be made from a trained Bayes Point Machine classifier using the `Predict` module, both in binary and multi-class classification. The `Predict` module reads a trained classifier and a test set and returns the predictions for the test set. The predictions can then be [evaluated](Evaluate.md) on the test set.

The `Predict` module has the following command-line arguments:

### Required arguments

*   `test-set`: The file with test data containing ground truth labels (ignored) and features in the format described [earlier](../Runners.md).
*   `model`: The file from which a previously trained Bayes Point Machine classifier will be loaded.
*   `predictions`: The file to which the predictions will be saved.

A more detailed explanation of prediction is available [here](../API/Prediction.md).

### Example

```
Learner Classifier BinaryBayesPointMachine Predict --test-set test-set.dat   
    --model trained-binary-bpm.bin --predictions predictions.dat  

Learner Classifier MulticlassBayesPointMachine Predict --test-set test-set.dat   
    --model trained-multiclass-bpm.bin --predictions predictions.dat
```

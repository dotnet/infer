---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [Command-line runners](../Runners.md)

## Training

A Bayes Point Machine is trained using the `Train` module, both in binary and multi-class classification. The `Train` module reads a training set and returns a serialized trained classifier, which can then be used to [make predictions](Predict.md) or [train incrementally](TrainIncremental.md).

The `Train` module has the following command-line arguments:

### Required arguments

*   `training-set`: The file with training data containing ground truth labels and features in the format described [earlier](../Runners.md).
*   `model`: The file to which the trained Bayes Point Machine classifier will be saved.

### Optional arguments

*   `iterations`: The number of training algorithm iterations (defaults to 30).
*   `batches`: The number of batches into which the training data is split (defaults to 1).
*   `compute-evidence`: If specified, the Bayes Point Machine classifier will compute model evidence on the training data (defaults to false).

For more information about the command-line arguments, see [Settings](../API/Settings.md). A more detailed explanation of training is available [here](../API/Training.md).

### Example

```
Learner Classifier BinaryBayesPointMachine Train   
    --training-set training-set.dat --model trained-binary-bpm.bin   
    --iterations 25 --batches 2 --compute-evidence  

Learner Classifier MulticlassBayesPointMachine Train   
    --training-set training-set.dat --model trained-multiclass-bpm.bin   
    --iterations 25 --batches 2 --compute-evidence
```

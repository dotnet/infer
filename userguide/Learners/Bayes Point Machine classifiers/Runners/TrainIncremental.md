---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [Command-line runners](../Runners.md)

## Incremental Training

A Bayes Point Machine is trained incrementally using the `TrainIncremental` module, both in binary and multi-class classification. The `TrainIncremental` module reads a serialized trained classifier and a training set and returns an incrementally trained classifier, which can then be used to [make predictions](Predict.md) or train once more (incrementally).

The `TrainIncremental` module has the following command-line arguments:

### Required arguments

*   `training-set`: The file with training data containing ground truth labels and features in the format described [earlier](../Runners.md).
*   `input-model`: The file from which a previously trained Bayes Point Machine classifier will be loaded.
*   `model`: The file to which the trained Bayes Point Machine classifier will be saved.

### Optional arguments

*   `iterations`: The number of training algorithm iterations (defaults to 30).
*   `batches`: The number of batches into which the training data is split (defaults to 1).

For more information about the command-line arguments, see [Settings](../API/Settings.md). A more detailed explanation of incremental training is available [here](../API/Training.md).

### Example

```
Learner Classifier BinaryBayesPointMachine TrainIncremental   
    --input-model trained-binary-bpm.bin --training-set training-set.dat   
    --model incrementally-trained-binary-bpm.bin --iterations 15 --batches 3   

Learner Classifier MulticlassBayesPointMachine TrainIncremental   
    --input-model trained-multiclass-bpm.bin --training-set training-set.dat   
    --model incrementally-trained-multiclass-bpm.bin --iterations 15 --batches 3
```

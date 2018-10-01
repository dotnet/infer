---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [Command-line runners](../Runners.md)

## Sampling weights

The `SampleWeights` module reads a serialized trained Bayes Point Machine classifier (binary or multi-class) and returns a sample from its posterior weight distribution.

The `SampleWeights` module has the following command-line arguments:

### Required arguments

*   `model`: The file from which a previously trained Bayes Point Machine classifier will be loaded.
*   `samples`: The file to which the weight sample will be saved. The file will contain the weights of the first ![K](../../BPM/ClassCount.png) \- 1 classes, one class per line.

### Example

```
Learner Classifier BinaryBayesPointMachine SampleWeights   
    --model trained-binary-bpm.bin --samples samples.txt  

Learner Classifier MulticlassBayesPointMachine SampleWeights   
    --model trained-multiclass-bpm.bin --samples samples.txt
```

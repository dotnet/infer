---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [Command-line runners](../Runners.md)

## Cross-validation

You can use the `CrossValidate` module to assess the generalization performance of the Bayes Point Machine, both in binary and multi-class classification. The `CrossValidate` module starts by reading a labelled data from a file and partitions its instances into ![K](../../BPM/ClassCount.png) subsets of equal size, known as folds. It then trains the Bayes Point Machine classifier on ![K](../../BPM/ClassCount.png) \- 1 folds and evaluates its performance on the withheld ![K](../../BPM/ClassCount.png)-th fold. It cycles through all ![K](../../BPM/ClassCount.png) combinations of splits into training and validation sets to finally report the overall performance results.

The `CrossValidate` module has the following command-line arguments:

### Required arguments

*   `data-set`: The file containing ground truth labels and features in the format described [earlier](../Runners.md).
*   `results`: The CSV file to which the cross-validation results will be saved.

### Optional arguments

*   `folds`: The number of cross-validation folds to use (defaults to 5).
*   `iterations`: The number of training algorithm iterations (defaults to 30).
*   `batches`: The number of batches into which the training data is split (defaults to 1).
*   `compute-evidence`: If specified, the Bayes Point Machine classifier will compute model evidence on the training data (defaults to false).

For more information about the command-line arguments, see [Settings](../API/Settings.md).

### Example

```
Learner Classifier BinaryBayesPointMachine CrossValidate   
    --data-set training.dat --results cross-validation-results.csv   
    --iterations 15 --batches 1 --compute-evidence  

Learner Classifier MulticlassBayesPointMachine CrossValidate   
    --data-set training.dat --results cross-validation-results.csv   
    --iterations 15 --batches 1 --compute-evidence
```

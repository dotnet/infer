---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [Command-line runners](../Runners.md)

## Convergence diagnosis

The `DiagnoseTrain` module is used to assess the convergence of the message-passing algorithms used to train the Bayes Point Machine classifiers. It is available for both binary and multi-class classification. The `DiagnoseTrain` module reads a training set and produces a CSV file containing the differences in the posterior weight distributions, computed for a specified number of consecutive iterations of the training algorithm. When the training algorithm is converging, the differences will become smaller and smaller (see sample output below).

The `DiagnoseTrain` module has the following command-line arguments:

### Required arguments

*   `training-set`: The file with training data containing ground truth labels and features in the format described [earlier](../Runners.md).

### Optional arguments

*   `results`: The CSV file to which the weight distribution differences will be saved.
*   `model`: The file to which the trained Bayes Point Machine classifier will be saved.
*   `iterations`: The number of training algorithm iterations (defaults to 30).
*   `batches`: The number of batches into which the training data is split (defaults to 1).

For more information about the command-line arguments, see [Settings](../API/Settings.md). A more detailed explanation of training is available [here](../API/Training.md).

### Example

```
Learner Classifier BinaryBayesPointMachine DiagnoseTrain   
    --training-set training.dat --model trained-binary-bpm.bin   
    --results results.csv --iterations 500 --batches 1  

Learner Classifier MulticlassBayesPointMachine DiagnoseTrain   
    --training-set training.dat --model trained-multiclass-bpm.bin   
    --results results.csv --iterations 500 --batches 1
```

### Sample output

This is what gets written to the command-line window (console):

```
Data set contains 10000 instances, 2 classes and 5888 features. 
[17:16:09] Starting training... 
[17:16:10] Iteration 1      dp = 37641.0378952995       dt =   583ms  
[17:16:10] Iteration 2      dp = 37616.6784075578       dt =   242ms  
[17:16:10] Iteration 3      dp = 17.7014780451184       dt =   298ms  
[17:16:10] Iteration 4      dp = 1.26924794894729       dt =   173ms  
[17:16:11] Iteration 5      dp = 0.769664011372179      dt =   293ms  
[17:16:11] Iteration 6      dp = 0.15651774182376       dt =   192ms  
[17:16:11] Iteration 7      dp = 0.164699512014064      dt =   267ms  
[17:16:11] Iteration 8      dp = 0.0835439679522345     dt =   228ms  
[17:16:12] Iteration 9      dp = 0.068138911019533      dt =   288ms  
[17:16:12] Iteration 10     dp = 0.0439520998092477     dt =   246ms  
[17:16:12] Iteration 11     dp = 0.0322686108618166     dt =   341ms  
[17:16:13] Iteration 12     dp = 0.0237455855736709     dt =   238ms  
[17:16:13] Iteration 13     dp = 0.0175368057738852     dt =   282ms  
[17:16:13] Iteration 14     dp = 0.0129868852183733     dt =   217ms  
[17:16:13] Iteration 15     dp = 0.00964343919792277    dt =   292ms  
[17:16:14] Iteration 16     dp = 0.0071772245476559     dt =   173ms  
[17:16:14] Iteration 17     dp = 0.00535275621234321    dt =   233ms  
                               . 
                               . 
                               .
```

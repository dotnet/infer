---
layout: default 
--- 
[Infer.NET user guide](../../../index.md) : [Learners](../../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../../Bayes Point Machine classifiers.md) : [Command-line runners](../Runners.md)

## Evaluation

The `Evaluate` module computes a number of performance metrics for given predictions and ground truth labels. The module can hence be used to evaluate predictions from any classifier, not just the Bayes Point Machine!

The `Evaluate` module has the following command-line arguments:

### Required arguments

*   `ground-truth`: The file containing the ground truth labels (in the format described [earlier](../Runners.md), features can be absent, however).
*   `predictions`: The file from which the predictions will be loaded.

### Optional arguments

*   `report`: The text file to which an evaluation report will be written, containing most of the classification metrics of interest.
*   `calibration-curve`: The CSV file to which the empirical calibration curve will be written.
*   `roc-curve`: The CSV file to which the receiver operating characteristic (ROC) curve will be written.
*   `precision-recall-curve`: The CSV file to which the precision-recall curve will be written.
*   `positive-class`: The label indicating the positive class in the computation of calibration, ROC, and precision-recall curves. If left unspecified, the first class label encountered in the file with ground truth labels will be used.

A more detailed explanation of classifier evaluation and performance metrics is available [here](../API/Evaluation.md).

### Example

```
Learner Classifier Evaluate --ground-truth iris-test-set.dat   
    --predictions iris-predictions.dat --report evaluation.txt   
    --calibration-curve calibration.csv --roc-curve roc.csv   
    --precision-recall-curve pr.csv --positive-class Iris-virginica
```

### Sample output

Here is an example of an evaluation report:

```
Classifier evaluation report   
******************************  

           Date:      14/10/2014 18:50:37  
   Ground truth:      test-set.dat  
    Predictions:      predictions.dat  

 Instance-averaged performance (micro-averages)  
================================================  

                Precision =     0.9429  
                   Recall =     0.9427  
                       F1 =     0.9427  

                 #Correct =       1118  
                   #Total =       1186  
                 Accuracy =     0.9427  
                    Error =     0.0573  

                      AUC =     0.9915  

                 Log-loss =     0.2487  

 Class-averaged performance (macro-averages)  
=============================================  

                Precision =     0.9352  
                   Recall =     0.9383  
                       F1 =     0.9366  

                 Accuracy =     0.9383  
                    Error =     0.0617  

                      AUC =     0.9917  

         M (pairwise AUC) =     0.9952  

 Performance on individual classes  
===================================  

 Index           Label     #Truth  #Predicted  #Correct  Precision     Recall         F1        AUC  
---------------------------------------------------------------------------------------------------  
     1               3        603         596       575     0.9648     0.9536     0.9591     0.9908  
     2               2        280         277       255     0.9206     0.9107     0.9156     0.9910  
     3               1        303         313       288     0.9201     0.9505     0.9351     0.9935  

 Confusion matrix  
==================  

Truth \ Prediction ->  
       3    2    1  
  3  575   15   13  
  2   13  255   12  
  1    8    7  288  


 Pairwise AUC matrix  
=====================  

Truth \ Prediction ->  
          3       2       1  
  3       .  0.9942  0.9963  
  2  0.9942       .  0.9950  
  1  0.9963  0.9950       .
```

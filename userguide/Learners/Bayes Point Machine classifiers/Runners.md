---
layout: default 
--- 
[Infer.NET user guide](../../index.md) : [Learners](../../Infer.NET Learners.md) : [Bayes Point Machine classifiers](../Bayes Point Machine classifiers.md)

## Command-line runners

`Learner.exe` (which can be found in the `bin` directory of the Infer.NET release) provides access to the Infer.NET learners via modules that can be run from command-line. While the command-line modules make some assumptions about the data format (see below) and do not allow you to provide data via custom [mappings](../Bayes Point Machine classifiers/API/Mappings.md) tailored to your data, they are easy-to-use and do not require any implementation.

The command-line modules are organized hierarchically. There are two top-level modules: `Recommender` and `Classifier`. In this section we are only interested in the modules registered under `Classifier`. These are:

*   `BinaryBayesPointMachine` (for classification problems of _two_ classes)
*   `MulticlassBayesPointMachine` (for classification problems of _three or more_ classes)
*   [`Evaluate`](../Bayes Point Machine classifiers/Runners/Evaluate.md)

Both Bayes Point Machine classifier modules, `BinaryBayesPointMachine` and `MulticlassBayesPointMachine`, provide a number of additional modules defining operations:

*   [Train](../Bayes Point Machine classifiers/Runners/Train.md)
*   [TrainIncremental](../Bayes Point Machine classifiers/Runners/TrainIncremental.md)
*   [Predict](../Bayes Point Machine classifiers/Runners/Predict.md)
*   [CrossValidate](../Bayes Point Machine classifiers/Runners/CrossValidate.md)
*   [SampleWeights](../Bayes Point Machine classifiers/Runners/SampleWeights.md)
*   [DiagnoseTrain](../Bayes Point Machine classifiers/Runners/DiagnoseTrain.md)

The [Evaluate](../Bayes Point Machine classifiers/Runners/Evaluate.md) module itself has no sub-modules and can be used to evaluate the performance of _any_ classifier.

### Example

In this section and its subsections we describe the aforementioned modules together with their operation-specific options, all of which are prefixed by \-\- (dash, dash). Before we explain all these options in more detail, let us begin with a simple example sequence of commands to train, test and evaluate a multi-class Bayes Point Machine classifier.

```
Learner Classifier MulticlassBayesPointMachine Train   
    --training-set dna.train --model dna.mdl   

Learner Classifier MulticlassBayesPointMachine Predict   
    --test-set dna.test --model dna.mdl --predictions dna.predictions  

Learner Classifier Evaluate --ground-truth dna.test   
    --predictions dna.predictions --report dna.evaluation.txt
```

### Data format

The data format for the command-line modules is fixed since it is impossible to have the user define data [mappings](API/Mappings.md). This means that the classification data needs to be converted into the format required by the command-line runners.

Internally, the Bayes Point Machine command-line modules implement a standard data format mapping based on a _sparse feature representation_. _No bias_ is added by default, so that it may be necessary to add an additional feature with constant value to all instances.

The command-line classifier modules expect classification data in a single text file. The file's format is as follows:

*   There is one instance per line, first the label, then the features.
*   Empty lines or lines starting with "#", "//", or "%" are ignored.
*   The label and all features are separated from each other by an empty space.
*   A label can be any contiguous string, but must not contain a colon ":" or whitespace.
*   A single feature consists of a feature identifier and an optional feature value, separated by a colon ":" if the latter is specified. If a feature does not specify a feature value, it is implicitly set to 1.
*   The order of the features does not matter.
*   A feature identifier can be any contiguous string, but must not contain a colon ":" or whitespace.
*   A feature identifier must not appear more than once on a single line.
*   A feature value can be any numeric value (any string that can be parsed to `double`). Feature values which are infinite or `NaN` will be correctly parsed, but are disallowed by the Bayes Point Machine classifiers.
*   An omitted feature is implicitly assumed to have feature value 0.

In short, the format specifies one instance per line, allows for a zero-sparse feature representation which separates feature identifiers from feature values using a colon, and specifies labels on the beginning of a line.

Here is an example of how such a file might look (illustrating some corner cases):

```
// Six instances:  

A/I first1:2 second-2 third_3:1.3e-10  
J/O      
R/Z second-2:3.1234511  
J/O first1:0.12 second-2:4   
A/I first1:2           third_3:1.45e-10  
J/O first1:0.22  

    %Four more instances:  
A/I second-2 third_3:2.762e-10 first1:1.97  
J/O      first1:2.32  
R/Z second-2:3.1234511  
R/Z second-2:2.519 third_3
```

The resulting labels and feature values are (in a dense representation for the purpose of clarity):

| Label | Feature 1 first1 | Feature 2 second-2 | Feature 3 third_3 |
|-------------------------------------------------------------------|
| A/I | 2 | 1 | 1.3e-10 |
| J/O | 0 | 0 | 0       |
| R/Z | 0 | 3.1234511 | 0 |
| J/O | 0.12 | 4 | 0 |
| A/I | 2 | 0 | 1.45e-10 |
| J/O | 0.22 | 0 | 0 |
| A/I | 1.97 | 1 | 2.762e-10 |
| J/O | 2.32 | 0 | 0 |
| R/Z | 0 | 3.1234511 | 0 |
| R/Z | 0 | 2.519 | 1 |

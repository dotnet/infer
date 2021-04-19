---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Command-line runners](../../Matchbox recommender/Command-line runners.md)

## Data splitter

The data splitter allows you to easily split the instance data into a training and a test set. It takes a number of parameters which control the splitting algorithm explained in the [corresponding data mapping section](../API/Mappings/Splitting.md).

Required parameters

*   **input-data** \- the dataset to split
*   **output-data-train** \- training part of the split dataset
*   **output-data-test** \- test part of the split dataset

Optional parameters

*   **training-users** \- fraction of training-only users; defaults to 0.5
*   **test-user-training-ratings** \- fraction of test user ratings for training; defaults to 0.25
*   **cold-users** \- fraction of cold (test-only) users; defaults to 0
*   **cold-items** \- minimum fraction of cold (test-only) items; defaults to 0
*   **ignored-users** \- fraction of users not included in either the training or the test set; defaults to 0
*   **ignored-items** \- fraction of items not included in either the training or the test set; defaults to 0
*   **remove-occasional-cold-items** \- remove occasionally produced cold items that would increase the fraction of cold items above the specified minimum

Example

```
Learner Recommender SplitData --input-data AllData.dat  
                              --output-data-train TrainingSet.dat   
                              --output-data-test TestSet.dat  
                              --ignored-users 0.8   
                              --training-users 0.1   
                              --remove-occasional-cold-items
```

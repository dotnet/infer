---
layout: default
---
[Learners](../../../Infer.NET Learners.md) : [Matchbox recommender](../../Matchbox recommender.md) : [Command-line runners](../../Matchbox recommender/Command-line runners.md)

## Trainer

Training is performed using the Train argument to Learner Recommender. It takes as input a training dataset and outputs a serialized trained model, which can be loaded later for making predictions. Training takes in a number of arguments, explained in the [Setting up a recommender](../API/Setting up a recommender.md) section. There is more detail on the training procedure in the [Training](../API/Training.md) section.  Features are not normalized by the algorithm.  You will need to do feature encoding and normalization beforehand.

Required parameters

*   **training-data** \- training dataset
*   **trained-model** \- trained model file

Optional parameters

*   **traits** \- number of traits (defaults to 4)
*   **iterations** \- number of inference iterations (defaults to 20)
*   **batches** \- number of batches to split the training data into (defaults to 1)
*   **use-user-features** \- use user features in the model (defaults to False)
*   **use-item-features** \- use item features in the model (defaults to False)

Example

```
Learner Recommender Train --training-data TrainingSet.dat  
                          --trained-model TrainedMatchbox.bin   
                          --traits 5  
                          --iterations 30  
                          --batches 4  
                          --use-user-features   
                          --use-item-features
```

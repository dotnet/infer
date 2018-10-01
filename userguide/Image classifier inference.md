---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

[Page 1](Image classifier example.md) \| [Page 2](Image classifier model.md) \| [Page 3](Image classifier test model.md) \| Page 4

## Inference in the image classifier example

Now we have the models in place, let's look what happens when a user of our application drags an image to the green or red area. When the user releases the left mouse button after doing the drag, a `MouseLeftButtonUp` event will occur. This event is handled by the `Rectangle_MouseLeftButtonUp` method in the `ClassifierView` class. This method checks to see if the state of the image items has changed (i.e. whether labels have changed) and if yes, then calls into the Reclassify method of the `ItemsModel` class. The Reclassify method, which is the glue between the user interface and the inference, does the following:

1.  Determines, from the UI state, which are the labeled images - i.e. which images form part of the training set, and what are their labels (red or green)
2.  Calls the BayesPointMachine Train method to infer the weights of the model
3.  Calls the BayesPointMachine Test method to get the probabilty of classifying each test image as True (i.e. green)

The user interface code then updates the positions of all the test images, so that their horizontal coordinate represents the probability calculated in step 3.

Now let's look at the Train and Test methods in steps 2 and 3.

The BayesPointMachine Train method looks as follows:

```csharp
nTrain.ObservedValue = data.Length;  
trainingItems.ObservedValue = data;  
trainingLabels.ObservedValue = labels;  
weightPosterior.ObservedValue = trainEngine.Infer<VectorGaussian>(weights);
```

The first three lines are setting the given values required by the model. The final line does the inference and retrieves the inferred posterior weights and sets the given posterior weights for the test model. The first time this is run there is a noticeable delay since the model is being compiled.  This could be avoided by pre-compiling the model as described in [Using a precompiled inference algorithm](Using a precompiled inference algorithm.md).

The BayesPointMachine Test method looks as follows:

```csharp
nTest.ObservedValue = data.Length;  
testItems.ObservedValue = data;  
Bernoulli[] labelPosteriors = Distribution.ToArray<Bernoulli[]>(testEngine.Infer(testLabels));  
double[] probs = new  double[data.Length];  
for (int i = 0; i < probs.Length; i++)  
    probs[i] = labelPosteriors[i].GetProbTrue();  
return probs;
```

Again, the first two lines are setting the given values required by the test model (the weightPosteriors given having already been set by the Train method). The third line infers the label posteriors as Bernoulli distributions. Then the probability of each test point being True (i.e. green) is recovered from the Bernoulli distributions, and these probabilities are returned and used by the user interface to reposition the images.

<br/>
[Page 1](Image classifier example.md) \| [Page 2](Image classifier model.md) \| [Page 3](Image classifier test model.md) \| Page 4

---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

[Page 1](Image classifier example.md) \| [Page 2](Image classifier model.md) \| Page 3 \| [Page 4](Image classifier inference.md)

## The image classifier test model 

Often we will want to train a model using a training set, and then deploy it for prediction or test. The parameters of the model inferred in training will be used to define the test model. The BayesPointMachine constructor provides a test model also:

```csharp
nTest = Variable.New<int>();  
Range testItem = new Range(nTest);  
testItems = Variable.Array<Vector>(testItem);  
testLabels = Variable.Array<bool>(testItem);  
weightPosterior = Variable.New<VectorGaussian>();  
Variable<Vector> testWeights = Variable<Vector>.Random(weightPosterior);  
testLabels[testItem] = Variable.IsPositive(Variable.GaussianFromMeanAndVariance  
  (Variable.InnerProduct(testWeights, testItems[testItem]), noise));
```

This is very similar to the training model. Apart from the names being prefixed by 'Test' instead of 'Train', the only differences are:

1.  `testLabels` are not observed. The job of the test inference is to infer these test labels, conditional on the inputs, whereas in the training case, these were observed.
2.  `testWeights` are distributed according to `weightPosterior` which is of type `Variable<VectorGaussian>`, and is set to the posterior weights from the training model. No updates are made to model weights in test mode.

In preparation for inference test outputs, our BayesPointMachine constructor also creates a test inference engine which contains the inference parameters we want to use for test:

```csharp
testEngine = new InferenceEngine(new ExpectationPropagation());  
testEngine.NumberOfIterations = 1;
```

For this type of model, only 1 iteration is need to update belief messages in the factor graph.

<br/>
[Page 1](Image classifier example.md) \| [Page 2](Image classifier model.md) \| Page 3 \| [Page 4](Image classifier inference.md)

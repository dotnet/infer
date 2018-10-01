---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

[Page 1](Click model example.md) \| [Page 2](Click Model 1.md) \|  Page 3 \| [Page 4](Click Model Prediction.md)

## Click model 2 example

Here we implement the same model as in [click model 1](Click Model 1.md) but with [shared variables](Sharing variables between models.md). There are a number of reasons why one might want to use shared variables including memory problems, parallelisation, and more control over the schedule which might be necessary if there are convergence problems. Infer.NET provides a **SharedVariable** class and a **Model** class which ensure that the correct messages get marshalled between the different models. This model is available as Model2 in the example code. It mirrors the Model1 code except for the following:

*   **SharedVariable** objects are created in place of **Variable** objects for all variables that we want to infer; these are initialised with the priors.
*   Model code must be changed to refer to the instance of the SharedVariable for the current chunk.
*   The data is divided into identically sized chunks.
*   We explicitly loop over chunks, and do inference on each chunk. We need to loop over all chunks several times, checking marginals between each pass to test for convergence.
*   For each chunk, we use **SharedVariable** and **Model** class methods to obtain the variables for each submodel, and to perform inference on these variables, respectively.

Let's look at each of these in turn.

#### Creating the shared variables

There is a concept called **Model** that takes care of tedious plumbing needed for sharing information between models. The **SharedVariable** class is a convenient wrapper class used to specify the variables that are shared between the models. First, we create an instance of the model. Simultaneously, we also specify the variables that are shared across all the models. For more detailed description check out this link on [shared variables](Sharing variables between models.md).

```csharp
Model model = new Model(numChunks);  
SharedVariable<double> scoreMean = SharedVariable<double>.Random(priorScoreMean).Named("scoreMean");  
SharedVariable<double> scorePrec = SharedVariable<double>.Random(priorScorePrec).Named("scorePrec");  
SharedVariable<double> judgePrec = SharedVariable<double>.Random(priorJudgePrec).Named("judgePrec");  
SharedVariable<double> clickPrec = SharedVariable<double>.Random(priorClickPrec).Named("clickPrec");  
SharedVariable<double>[] thresholds = new SharedVariable<double>[numThresholds];  
for (int t = 0; t < numThresholds; t++) {  
  thresholds[t] = SharedVariable<double>.Random(priorThresholds[t]).Named("threshold" + t);  
}
```

#### Changes in the model code

Changes are needed in the model code to take care of the fact that scoreMean etc. are no longer **Variable**s. These changes just require referring to the **GetCopyFor**method of the **SharedVariable** instance:

```csharp
scores[r] = Variable<double>.GaussianFromMeanAndPrecision(  
    scoreMean.GetCopyFor(model), scorePrec.GetCopyFor(model)).ForEach(r);  
scoresJ[r] = Variable<double>.GaussianFromMeanAndPrecision(  
    scores[r], judgePrec.GetCopyFor(model));  
scoresC[r] = Variable<double>.GaussianFromMeanAndPrecision(  
    scores[r], clickPrec.GetCopyFor(model));  
  
   ...  
Variable.ConstrainBetween(  
    scoresJ[r], thresholds[i].GetCopyFor(model), thresholds[i + 1].GetCopyFor(model));
```

#### Dividing the data into chunks

A method is provided for this example (i.e. not part of Infer.NET) which divides the data into chunks and returns an array of arrays of arrays of Gaussians. The first index of this triply-indexed array is the chunk index, the second is the label index, and the third is the query/document index:

```csharp
Gaussian[][][] allObs = getClickObservations(numLabels,chunkSize,labels,clicks,exams);  
int numChunks = allObs.Length;
```

When a given chunk, index c, is processed, the **numberOfObservations** and **observationDistrib** parameters must be set according to the corresonding data:

```csharp
for (int i = 0; i < numLabels; i++)  
{  
    numberOfObservations[i].ObservedValue = allObs[c][i].Length;  
    observationDistribs[i].ObservedValue = allObs[c][i];  
}
```

#### Looping over chunks, and convergence

We do several passes over the chunks. We can infer all the variables jointly by using **InferShared** method of Model class. After each full pass the marginals of key shared variables are tested against the marginals of the previous pass in order to determine convergence:

```csharp
for (int pass = 0; pass < maxPasses; pass++)  
{  
    prevMargScoreMean = marginals.marginalScoreMean;  
    prevMargJudgePrec = marginals.marginalJudgePrec;  
    prevMargClickPrec = marginals.marginalClickPrec;  
    for (int c = 0; c < numChunks; c++)  
    {  
        // set values  
          
        ...  
  
        // perform inference  
        model.InferShared(engine,c); // Retrieve marginals  
        marginals.marginalScoreMean = scoreMean.Marginal<Gaussian>();  
        marginals.marginalScorePrec = scorePrec.Marginal(Gamma);  
        marginals.marginalJudgePrec = judgePrec.Marginal(Gamma);  
        marginals.marginalClickPrec = clickPrec.Marginal(Gamma);  
        // Test for convergence if (marginals.marginalScoreMean.MaxDiff(prevMargScoreMean) < convergenceThresh &&  
        marginals.marginalJudgePrec.MaxDiff(prevMargJudgePrec) < convergenceThresh &&  
        marginals.marginalClickPrec.MaxDiff(prevMargClickPrec) < convergenceThresh)  
        break;  
    }  
}
```

#### Running Model 2

We can run model 2 using the same [prediction model](Click Model Prediction.md) as for model 1. We present the data in chunks of 200; the inference converges after 4 passes, and gets identical results (up to some small numeric tolerance) with model 1.

<br/>
[Page 1](Click model example.md) \| [Page 2](Click Model 1.md) \|  Page 3 \| [Page 4](Click Model Prediction.md)
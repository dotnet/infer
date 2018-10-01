---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Computing model evidence for model selection

The central quantity in [Bayesian model selection](http://alumni.media.mit.edu/~tpminka/statlearn/demo/) is the probability of the data with all parameters integrated out, also known as the model evidence. Infer.NET can compute model evidence as a special case of mixture modelling. Esssentially you create a mixture of your model with an empty model, and the learned mixing weight is then the model evidence. The following example illustrates:

```csharp
Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");  
IfBlock block = Variable.If(evidence);  
// start of model  
Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1);  
Variable.ConstrainTrue(x > 0.5);  
// end of model  
block.CloseBlock();  
InferenceEngine engine = new InferenceEngine();  
double logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;  
Console.WriteLine("The probability that a Gaussian(0,1) > 0.5 is {0}", Math.Exp(logEvidence));
```

The output of this code is:

```csharp
The probability that a Gaussian(0,1) > 0.5 is 0.308537538725987
```

To choose the number of components in a Gaussian mixture, for example, you set the number of components to 2, compute the model evidence, then change the number of components to 3, compute the new model evidence, and so on. Multiply these evidence values with the prior probability of each model, and the model with the highest number has the highest posterior probability of being correct for the data.
---
layout: default 
--- 
[Infer.NET user guide](index.md)

## A simple example

Here is an example of using Infer.NET to work out the probability of getting both heads when tossing two fair coins.

```csharp
Variable<bool> firstCoin = Variable.Bernoulli(0.5);
Variable<bool> secondCoin = Variable.Bernoulli(0.5);
Variable<bool> bothHeads = firstCoin & secondCoin;
InferenceEngine engine = new InferenceEngine();
Console.WriteLine("Probability both coins are heads: "+engine.Infer(bothHeads));
```

The output of this program is:

```csharp
Probability both coins are heads: Bernoulli(0.25)
```

which correctly gives the probability of two heads as 0.25 or 1/4.

This short example contains the three key elements of any Infer.NET program:

1.  **Definition of a probabilistic model**

    All Infer.NET programs need a probabilistic model to be defined. This is done in the first three lines above by defining the random variables **firstCoin** and **secondCoin** and the specifying the dependent variable **bothHeads** as a function of these. You can read more about defining models in [the Infer.NET modelling API](The Infer.NET modelling API.md).


2.  **Creation of an inference engine**

    All inference is achieved through the use of an inference engine. This must be created and configured before any inference is performed. The fourth line above creates an inference engine which uses the default inference algorithm (expectation propagation).


3.  **Execution of an inference query**

    Given an inference engine, you can query marginal distributions over variables using **Infer()**. In the last line of the example, the engine is used to infer the marginal distribution of **bothHeads** i.e. the probability that both coins turned up heads. The engine returns a Bernouilli distribution which is then printed to the console. You can read more about inference in the section on [running inference](Running inference.md).

Read on to find out [how Infer.NET works](how Infer.NET works.md).

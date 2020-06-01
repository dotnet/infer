---
layout: default
---

[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Causal inference with Infer.NET

_The paper [Causality with Gates](https://www.microsoft.com/en-us/research/publication/causality-with-gates/?from=http%3A%2F%2Fresearch.microsoft.com%2Fapps%2Fpubs%2Fdefault.aspx%3Fid%3D162692) is now available which describes the theory behind this page._

An oft quoted phrase is "correlation does not imply causation". It means that if A tends to be true when B is true (i.e. A and B are correlated), then it is not correct to assume that A causes B (or vice versa). 

For example, if you observe that people taking a new cancer drug are surviving longer than people taking the old drug, you cannot assume that the new drug is better. It may be that the new drug is more expensive and so the people who can afford to take it can afford better healthcare in general than those who can't. In order to detect a causal relationship, you need to have data about **interventions** i.e. where one of the variables is directly controlled. A randomized study is an example of this, where doctors control whether to give a person a new drug or not, according to a random assignment. Because people are randomly assigned to the treated group (given the new drug) or the control group (not given the drug), the only systematic difference between these two groups is whether they had the drug or not. Other factors, like how rich they are, will not vary systematically between the two groups and so misleading conclusions about whether the drug works better can be avoided. 

Inferring the probability that a causal link exists between two variables can be highly complex. I will show how you can use Infer.NET to do this inference automatically.

### Does A cause B or B cause A?

We'll consider a simple example with two binary variables **A** and **B**. The question we want to ask is whether **A** causes **B** or **B** causes **A** (in this example we won't consider any other possibilities e.g. that a third variable **C** causes both **A** and **B**).

We'll first look at how to model '**A** causes **B**'. For this model, we'll assume that **A** is selected to be true 50% of the time and false 50% of the time (a Bernoulli distribution with parameter 0.5). We're going to have N observations for each of A and B, so in Infer.NET we write this as:

```csharp
A[N] = Variable.Bernoulli(0.5);
```

We'll then assume that **B** is a noisy version of **A** such that **B** is set to the same value as **A** with probability _(1-q)_ and set to the opposite value with probability _q_. So large values of _q_ (e.g. 0.4) means that B is a very noisy version of **A**, whereas small values of _q_ (e.g. 0.01) mean that **B** is almost identical to **A**.

In Infer.NET we write this as:

```
B[N] = A[N] != Variable.Bernoulli(q);
```

So that's it for our first model. The complete code with array allocations and loop over N is:

```csharp
var A = Variable.Array<bool>(N).Named("A"); // Array allocation for A
var B = Variable.Array<bool>(N).Named("B"); // Array allocation for B
// First model code (A causes B)
using (Variable.ForEach(N)) // Loop over N
{
  A[N] = Variable.Bernoulli(0.5);
  B[N] = A[N] != Variable.Bernoulli(q);
}
```

### A tale of two models

We now want to consider the second model where **B** causes **A**. The model is going to be defined just as for the first model, but with **A** and **B** swapped. However, we don't just want to define this model, we want Infer.NET to work out which of the two models is the right one. To do this we need to introduce a binary switch variable, that I will call **AcausesB**. If this variable is true, then we will use the first model (**A** causes **B**) and if it is false, we will use the second model (**B** causes **A**).
The code looks like this:

```csharp
var AcausesB = Variable.Bernoulli(0.5); // The model switch variable
var A = Variable.Array<bool>(N).Named("A");  // Arrays are allocated once
var B = Variable.Array<bool>(N).Named("B");  // and used for both models

using (Variable.If(AcausesB))
{
  // First model code goes here (A causes B)
  // i.e. the ForEach loop above
}
using (Variable.IfNot(AcausesB))
{
  // Second model code goes here (B causes A) 
  // This is the same code as for the first model but with A and B swapped
}
```

To use this code to work out which model is true, we must: create an inference engine, attach data (boolean arrays) to the model by observing the values of **A** and **B**, and use the inference engine to infer the posterior distribution of **AcausesB**. Here's the code:

```csharp
var engine = new InferenceEngine();  // Create an inference engine 
A.ObservedValue = dataA;  // Attach data to A
B.ObservedValue = dataB;  // Attach data to B 
Bernoulli AcausesBdist = engine.Infer<Bernoulli>(AcausesB); // Infer posterior
```

If you run the whole program, you will find that the posterior distribution `AcausesBdist` is always Bernoulli(0.5) that is, a 50% chance that each model is true, no matter what data you attach to **A** and **B**. In other words, Infer.NET is saying "I don't know". This is because _without interventions_, it is impossible to say which of the two models is the true one.

### Interventions

To get this example to work, we're going to add in interventions on the variable `B`. For a subset of our data points, we will intervene to set the value of `B` directly. We will record which data points we intervene on using another variable `doB`. When `doB` is true, it indicates that we set the value of `B` directly, overriding the existing model above. When `doB` is false, `B` will be set according to the existing model.

Here is the code for an intervention where we set the value of **B** according to a coin flip i.e. Bernoulli(0.5). It should be placed above the two model definitions, since it is common to both. 

```csharp
var doB = Variable.Array<bool>(N).Named("doB"); // True if we intervene on B, false otherwise 
using (Variable.ForEach(N))
{
  using (Variable.If(doB[N]))
  {
    B[N] = Variable.Bernoulli(0.5); // If we intervene, set B at random.
  }
}
```

To make sure that we do not define **B** more than once, we also need to modify each model, so that `B[N]` is not set if an intervention happens. So the in the first model we need to add an `IfNot` statement around the line setting `B[N]`:

```csharp
using (Variable.IfNot(doB[N]))
{
  B[N] = A[N] != Variable.Bernoulli(q);
}
```

In the second model the `IfNot` statement is still placed around the line setting `B[N]`, but in this case it is the first line because `A[N]` and `B[N]` are swapped:

```csharp
using (Variable.IfNot(doB[N]))
{
  B[N] = Variable.Bernoulli(0.5);
}
```

Because the intervention affects the two models in different ways, we will now be able to tell the difference between them.

### The results

To test out the finished model, let's assume that the true model is the first one, so that **A** actually does cause **B**. We can create sampled data sets with interventions from this true model with various numbers of interventions _N_ and for varying noise levels _q_ (to see how to do this look at the full code attached). For each data set, we use the above code to compute the posterior of the true model P(**AcausesB**).

The following plot shows the resulting computed probability for varying _N_ and _q_. To account for random variation in datasets of the same size, the computed probability has been averaged over 1000 generated datasets. The plots show that Infer.NET has worked out correctly that the first model is the most probable one, but that the probability depends on the noise level and the number of interventions in the data. The less noise there is in the relationship between A and B, the fewer interventions are needed to be confident that **A** causes **B**. For example, when _q_=0.2 it takes about 20 interventions to be 90% sure that **A** causes **B**, whereas when _q_=0.1 it takes less than 10.

The complete code for this example is in [CausalityExample.cs](https://github.com/dotnet/infer/blob/master/src/Tutorials/CausalityExample.cs).  There is also an [F# version](https://github.com/caxelrud/Fsharp-and-infer.NET/tree/master/Examples_1/Causuality). 

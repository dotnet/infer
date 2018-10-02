---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

[Page 1](Click through model sample.md) \| Page 2

## Defining the model using Infer.Net:

#### Initializing variables:

In this example, we assume the probabilities which represent whether a user will examine the next document (conditioned on whether a user clicks and whether the document is relevant) are specified by the user. These values do not need to be specified during model specification, and are needed only during inference. Therefore, we just create them as new variables, and hold off on giving them values until we run the model:

```csharp
Variable<double>[] probNextIfNotClick = Variable.New<double>();  
Variable<double>[] probNextIfClickNotRel = Variable.New<double>();  
Variable<double>[] probNextIfClickRel = Variable.New<double>();
```

By the same argument, the number of users is also deferred to run-time:

```csharp
Variable<int> nUsers = Variable.New<int>();
```

For every document, we have its (user-independent) _relevance_ the summary _appeal_, which are created as .NET arrays over variables:

```csharp
Variable<double>[] appeal = new Variable<double>[nRanks];  
Variable<double>[] relevance = new Variable<double>[nRanks];
```

_Appeal_ and _relevance_ are given uniform priors for each document:

```csharp
for (int d = 0; d < nRanks; d++)  
{  
    appeal[d] = Variable.Beta(1, 1);  
    relevance[d] = Variable.Beta(1, 1);  
}
```

For every user, and for every document, we have the variables which represent whether the user examined the document, whether they clicked on it, and whether is was relevant for them. For each document, for each document rank we use VariableArrays to model across users:

```csharp
VariableArray<bool>[] examine = new VariableArray<bool>[nRanks];  
VariableArray<bool>[] click = new VariableArray<bool>[nRanks];  
VariableArray<bool>[] isRel = new VariableArray<bool>[nRanks];
```

The clicks are observed, and we will specify those observations for the click arrays at run time; however, we do not need to have the observations at the time we specify the model.

At each rank, these variables, _examine_, _click_, and _isRel_ follow the same conditional distribution across all users. So, we use the range variable, user, to specify the distribution across all users simultaneously. The range variable can be specified as: 

```csharp
Range u = new Range(nUsers).Named("User");
```

```csharp
examine[d] = Variable.Array<bool>(u);  
click[d] = Variable.Array<bool>(u);  
isRel[d] = Variable.Array<bool>(u);
```

#### Specifying the model:

In Infer.NET, we can specify the model in the same way as we would sample from it. In this section, we elaborate how to define our probabilistic model of clicks.

First, let's consider the _examine_ variable. We assume all users have examined the very first document (d==0) so that:

```csharp
examine[0][u] = Variable.Bernoulli(1).ForEach(u);
```

The documents at other ranks are examined according to the conditional distribution specified by:

```csharp
using (Variable.ForEach(u))  
{  
    var nextIfClick = Variable.New<bool>();  
    using (Variable.If(isRel[d-1][u]))  
        nextIfClick.SetTo(Variable.Bernoulli(probNextIfClickRel));  
    using (Variable.IfNot(isRel[d-1][u]))  
        nextIfClick.SetTo(Variable.Bernoulli(probNextIfClickNotRel));  
    var nextIfNotClick = Variable.Bernoulli(probNextIfNotClick);  
    var next = (((!click[d - 1][u]) & nextIfNotClick) | (click[d - 1][u] & nextIfClick));  
    examine[d][u] = examine[d - 1][u] & next;  
}
```

The conditional probability table for examine\[d\]\[u\] is specified by using [Variable.If and Variable.IfNot](Branching on variables to create mixture models.md) constructs, along with [“and” and “or” factors](Boolean and comparison operations.md) as shown in the factor graph. The variables _nextIfClick_, _nextIfNotClick_, and _next_ are represented by small circles in the factor graph on [Page 1](Click through model sample.md). Note that this whole piece of model code is embedded in a [ForEach block](ForEach blocks.md). This ensures that the calls to Variable.Bernoulli create a separate variable for each user.

The conditional distribution over the _click_ variable _isRel_ variables at each rank and across all users can now be specified as:

```csharp
using (Variable.ForEach(u))  
{  
    click[d][u] = examine[d][u] & Variable.Bernoulli(appeal[d]);  
    isRel[d][u] = click[d][u] & Variable.Bernoulli(relevance[d]);  
}
```

#### **Instantiation of the inference engine:**

We create an inference engine with EP as the inference algorithm:

```csharp
InferenceEngine ie = new InferenceEngine(new ExpectationPropagation());
```

#### Specifying the parameters and the observed variables:

In this code, all the required parameters and observations are provided by a 'user' object. Using this, we set the observed values as follows:

```csharp
nUsers.ObservedValue = user.nUsers;  
probNextIfNotClick.ObservedValue = user.probExamine[0];  
probNextIfClickNotRel.ObservedValue = user.probExamine[1];  
probNextIfClickRel.ObservedValue = user.probExamine[2];  
for (int d = 0; d < nRanks; d++)  
    click[d].ObservedValue = user.clicks[d];
```

#### **Performing inference:**

```csharp
for (int d = 0; d < nRanks; d++)  
{  
    docStats[d].inferredRelevance = ie.Infer<Beta>(relevance[d]);  
    docStats[d].inferredAppeal = ie.Infer<Beta>(appeal[d]);  
}
```

<br/>
[Page 1](Click through model sample.md) \| Page 2
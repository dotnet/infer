---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Strings Tutorial 1: Hello, Strings!

This page describes an experimental feature that is likely to change in future releases.

This tutorial introduces the basics of performing inference over string variables in Infer.NET. It shows how to define a generative process that includes strings and how to reason about variables involved in that process.

You can run the code in this tutorial either using the [Examples Browser](The examples browser.md) or by opening the Tutorials solution in Visual Studio and editing RunMe.cs to execute [HelloStrings.cs](https://github.com/dotnet/infer/blob/master/src/Tutorials/HelloStrings.cs).

### A generative model of text

Probabilistic models we've considered so far had variables of numeric types only, i.e. integers, floats and booleans. In principle, however, there is no reason to restrict model variables to these domains: as long as the inference engine is able to handle variables of a certain type, such variables should be allowed in the modelling code. The Infer.NET inference engine, in particular, also supports variables of collection types, such as strings or lists. In this tutorial we will focus on string variables.

One way to define a string random variable is to specify a prior distribution over it:

```csharp
Variable<string> str1 = Variable.StringUniform().Named("str1");  
Variable<string> str2 = Variable.StringUniform().Named("str2");
```

_**See also:** [Creating variables](Creating variables.md)_

**Variable.StringUniform** creates a string random variable from a uniform distribution over all possible strings. So, both **str1** and **str2** can potentially take any value, and all values are equally likely. It should be noted that since the number of all possible strings is infinite, this distribution is [improper](http://en.wikipedia.org/wiki/Prior_probability#Improper_priors). However, in many models improper priors don't constitute a problem since the posterior distribution over variables with an improper prior can still be proper.

Another way to obtain a string random variable is to invoke an operation that produces a string, such as, for instance, concatenation:

```csharp
Variable<string> text = (str1 + " " + str2).Named("text");
```

So, **text** is defined to be a concatenation of **str1**, a string containing a single space, and **str2**.

The model we've just defined can be thought of as the following generative process: take any two strings and concatenate them, putting a space in between. Another possible interpretation is a parsing process that accepts only strings that contain at least one space. If you are familiar with regular expressions, such a process can be represented by an expression of the form ".* .*".

### Uncertain segmentation

Now that we have a model, we are ready to observe some data and make an inference about its variables. In particular, let us observe the value of **text** and try to figure out what the values of **str1** and **str2** are. To observe the value of **text**, we, as before, need to set its **ObservedValue** property:

```csharp
text.ObservedValue = "Hello uncertain world";
```

Note that it's not clear from the value of **text** what **str1** and **str2** are: the whitespace between **str1** and **str2** can correspond to either the first or the second space in the observed string. The segmentation of **text** into **str1** and **str2** is, therefore, subject to uncertainty. And this is precisely the conclusion that the Infer.NET inference engine will reach if we run it on this model, as we will see soon.

```csharp
var engine = new InferenceEngine();  
engine.Compiler.RecommendedQuality = QualityBand.Experimental;  

Console.WriteLine("str1: {0}", engine.Infer(str1));  
Console.WriteLine("str2: {0}", engine.Infer(str2));
```

_**See also:** [Quality bands](Quality bands.md)_

A couple of important things to note is that a) inference over strings is currently supported only with the expectation propagation algorithm and b) it's currently considered to be an experimental feature, so, to prevent the model compiler from emitting warnings about using experimental components, the recommended quality level must be amended. Running this code will produce the following output:

```
str1: Hello[ uncertain]  
str2: world|uncertain world
```

Currently when a distribution over strings is printed to the console (or **ToString** is called on it), the result is a compact representation of the set of all strings that are possible under that distribution, also known as the support of the distribution. So, if we print the posterior distribution over **str1** and **str2**, we can immediately see that, given the observed text, the value of **str1** used to produce it could have been "Hello" or "Hello uncertain", while **str2** could have been "world" or "uncertain world", as discussed above. 

### StringAutomaton

Inspecting the support of the posterior is not, however, usually sufficient. For a more detailed analysis, say, seeing how likely a particular string is under the distribution, one needs to obtain a distribution object. For string random variables the corresponding object is always of type **StringDistribution**. The **StringDistribution** class is an implementation of a distribution over strings that represents uncertainty via a [weighted finite state automaton](http://en.wikipedia.org/wiki/Finite_state_transducer#Weighted_automata) internally. As with other distribution classes, it has methods for sampling and retrieving the probability of a given string. Thus, we can write the following code:

```csharp
var distOfStr1 = engine.Infer<StringDistribution>(str1);  
foreach (var s in new[] { "Hello", "Hello uncertain", "Hello uncertain world" })  
{  
    Console.WriteLine("P(str1 = '{0}') = {1}", s, distOfStr1.GetProb(s));  
}
```

And, as expected, it will produce this output:

```
P(str1 = 'Hello') = 0.5  
P(str1 = 'Hello uncertain') = 0.5  
P(str1 = 'Hello uncertain world') = 0
```

We are now ready to move to the [next tutorial](StringFormat operation.md), where we'll see some other supported operations over strings and use them to define a more sophisticated model.

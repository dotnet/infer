---
layout: default
---
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Tutorial 1: Two coins

This tutorial introduces the basics of Infer.NET: creating random variables, linking them together and inferring marginal distributions. It considers what happens when two fair coins are tossed. 

You can run the code in this tutorial either using the [Examples Browser](The examples browser.md) or by opening the Tutorials solution in Visual Studio and executing [FirstExample](https://github.com/dotnet/infer/blob/master/src/Tutorials/FirstExample.cs).

The code for this tutorial is also available in: **[F#](https://github.com/dotnet/infer/blob/master/test/TestFSharp/TwoCoins.fs)**, **[C++](CPlusPlus Two coins.md)**, **[IronPython](Two coins in IronPython.md)**, **[Matlab](Two coins in Matlab.md)**.

### Variables, random and otherwise

The most basic element in Infer.NET is the _variable_. Variables are represented using the Variable<T\> class in the **Microsoft.ML.Probabilistic.Models** namespace. Variables are strongly typed so that Variable<bool\> is a boolean variable and Variable<double\> is a double variable. An important feature of Infer.NET is that variables can be deterministic (with a single known value) or random (with an unknown or uncertain value). The Variable<T\> type is deliberately used for both cases, since we shouldn't have to care whether a variable is deterministic or random when we are using it.

To represent a coin, we can use a boolean Variable where **true** represents heads and **false** represents tails. As each coin is fair, it has a 50% chance of turning up heads and a 50% change of turning up tails. A distribution over a boolean value with some probability of being true is called a _Bernoulli_ distribution. So we can simulate a coin by creating a boolean random variable from a Bernoulli distribution with a 50% probability of being true. In Infer.NET, we can create a random variable with this distribution using `Variable.Bernoulli(0.5)`:

```csharp
Variable<bool> firstCoin = Variable.Bernoulli(0.5);  
Variable<bool> secondCoin = Variable.Bernoulli(0.5);
```

You can now think of **firstCoin** or **secondCoin** as having a distribution over values, rather than a single value. In general, a common way of creating a random variable is by specifying its prior distribution, such as a Bernoulli, Gaussian, Gamma or Discrete.

Another way of making a random variable is to derive it using an expression containing other random variables. When a random variable is used in an expression, the result of the expression will itself be a new random variable. For example, we can create a derived random variable called **bothHeads** like so:

```csharp
Variable<bool> bothHeads = firstCoin & secondCoin;
```

_**See also:** [Creating variables](Creating variables.md)_

Here, **bothHeads** is true only if both **firstCoin** and **secondCoin** are true and hence it represents the situation where both coins turn up heads. We have not directly given **bothHeads** a prior distribution, yet it is a random variable, because it is a function of random variables. So, we can ask the question "what is the distribution of this random variable?". To answer this question, we need to perform _inference_.

### Inferring distributions

The primary purpose of Infer.NET is to perform inference, in other words, to infer the posterior distribution of a particular random variable. In this example, we have created a variable **bothHeads** which is random, and would like to find out its distribution. To do this, we need to use an Infer.NET _inference engine_. All inference in Infer.NET is achieved through the use of an inference engine, using its **Infer()** method. This code creates an inference engine with default settings and uses it to infer the distribution over **bothHeads**.

```csharp
InferenceEngine engine = new InferenceEngine();  
Console.WriteLine("Probability both coins are heads: "+engine.Infer(bothHeads));
```

When run, this code prints out:

```csharp
Probability both coins are heads: Bernoulli(0.25)
```

_**See also:** [Running inference](Running inference.md)_

which is the correct answer of a quarter (=0.5 * 0.5). The **Infer()** method returns a distribution object that represents, either exactly or approximately, the posterior distribution over the random variable. How this is achieved under the hood depends on the settings of the inference engine, for example which inference algorithm is used. You can configure these settings, as described in [inference engine settings](inference engine settings.md).

### Going backwards

We can also use Infer.NET to do backwards reasoning, where we _observe_ the output to be a particular value and ask questions about the inputs. For example, we could observe **bothHeads** to be false. This is like someone tossing two coins in secret, and only telling us that the result was not two heads. We might then want to know what this tells us about the original coin tosses - for example, using this information what is the probability that the first coin was heads. This code adds the observation that we know **bothHeads** is false and uses the inference engine to infer the new distribution over **firstCoin**.

```csharp
bothHeads.ObservedValue=false;  
Console.WriteLine("Probability distribution over firstCoin: " + engine.Infer(firstCoin));
```

When run, this code prints out:

```csharp
Probability distribution over firstCoin: Bernoulli(0.3333)
```

which says that, given the result was not heads-heads, the probability that the first coin is heads is 1/3. This can be seen to be the case by noting that the three possible outcomes are tails-heads, heads-tails and tails-tails, only one of which has the first coin being heads.

Congratulations on running your first Infer.NET program! You can now try the [second tutorial](Truncated Gaussian tutorial.md).

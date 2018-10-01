---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Improving the speed and accuracy of inference](improving the speed and accuracy of inference.md)

## Using sparse messages

When performing inference, Infer.NET internally works with _messages._ These are distribution objects (like `Discrete`, `DiscreteChar` or `Dirichlet`) which are used when any inference algorithm is being applied. If you have a `Variable<int>` in your model which can take any of 1,000,000 values then this will cause the inference algorithm to use messages which are `Discrete` distributions with a size of 1,000,000. Normally, this would require storing a million doubles (at eight bytes each!) for every message relating to that variable. At 8MB per message, this model would have an enormous memory footprint and also it would take a very long time to process the individual messages making inference very slow. Often, these messages are very sparse, in that most of the values are the same or very nearly the same. We can exploit this to make inference much faster and require much less memory by using _sparse messages_ which just store the values which are different from the common value.

The `Discrete`, `DiscreteChar` and `Dirichlet` distribution classes both have support for sparsity, because they use the Vector class to represent parameters of the distribution. Instances of the Vector class can be dense, sparse (exactly) or sparse (with approximation), or piecewise, as described in [Vector and matrix types](Vector and matrix types.md). If you create a random variable from a vector**v** using `Variable.Discrete(v)`, `Variable.Char(v)` or `Variable.Dirichlet(v)` then the messages used for that variable will have the same sparseness as **v**. So if **v** is sparse, the messages relating to this variable will also be sparse.

#### Setting the sparsity of a variable

Normally the best way to make messages sparse is to use a sparse prior, as described above. If you want the sparsity of messages to be different from the prior, then you can use the `SetSparsity()` method on a variable. This encourages all messages into or out of the variable to be sparse. However, message operators can override this. If a message operator is not written to produce a sparse result when requested, then the message will be dense regardless of this setting. So it is important that the factors in your model are implemented to produce sparse messages.

For example, to encourage (exact) sparse messages for a variable `x`:

```csharp
x.SetSparsity(Sparsity.Sparse);
```

To encourage the messages for **x** to be sparse with approximation, so that any value within 0.001 of the common value is approximated by that common value:

```csharp
x.SetSparsity(Sparsity.ApproximateWithTolerance(0.001));
```

To set the tolerance level, a useful strategy is to run some test cases and find the largest value whose results are sufficiently similar to a tolerance of 0.

If the messages for **x** are automatically being made sparse, but you want them to be dense, you can write:

```csharp
x.SetSparsity(Sparsity.Dense);
```

This will override the default behaviour so that dense messages are used instead.

#### A real example

An example of a real world model where it is important to exploit sparsity for efficient inference is Latent Dirichlet Allocation (LDA). The LDA model attempts to uncover topics in a collection of documents by learning a different distribution over words for each topic. Since a document typically only uses a small subset of the words in the vocabulary, significant savings can be made by using sparse representations of many of the messages being sent.

In an example run of the LDA model, memory consumption is reduced by 20-30 times by using sparse messages and inference time is reduced by a factor of 10. For full details of the LDA model and how sparsity affects its speed/memory consumption, see [the LDA example page](Latent Dirichlet Allocation.md).
---
layout: default 
--- 
[Infer.NET development](index.md)

## The Infer.NET Distribution library

This library provides data structures for representing distributions and routines for standard mathematical operations such as multiplying two distributions, sampling from a distribution, and comparison of distributions. While any distribution type could be accommodated, the emphasis is on standard exponential families, for use within message-passing. Complex multivariate distributions are better handled via the factor graph library. Consequently, we often refer to these distributions as “message types”, to emphasize their purpose. However, distribution objects can be used outside of message passing. A demonstration of EM on a mixture of Gaussians is available at examples/Clustering.

The operations supported by a distribution are motivated by message-passing. The most common requirements are:

*   Set to uniform, test if uniform
*   Set to point mass, test if point mass, get location of point mass
*   Clone, Copy to another message
*   Equality testing, measuring distance between two messages
*   Cursor functionality
*   Multiplication of messages

Specific algorithms will require methods such as:

*   Evaluate the distribution at a point 
*   Sample 
*   Get the mean 

For model checking, the following are also needed:

*   Test compatibility of two messages (e.g. dimensions) 
*   Test compatibility of a message with a variable domain 

However, these do not need to be methods on the message object, but can be stored in a separate descriptor. Note that we do not require a message object to supply a description of its domain; it only needs to check compatibility with a given domain. 

The library uses a capability system to let message types advertise which operations they support. Each bulleted item above has its own interface, often with only one method in it. Algorithms use run-time type tests (preferably at model compile time) to determine if the message type has the capability it needs. Note that interfaces only apply to instance methods, so all the operations above must correspond to instance methods, not static methods.

Unfortunately, C# makes it cumbersome to use multiple interfaces on an object at once. For example, you cannot declare an array whose items support two interfaces at once. To alleviate this, frequent combinations of capabilities, such as the first list above, are collected into composite interfaces (which add no new methods). Unfortunately, this requires message types to declare the composite interface as well, since C# does not detect this automatically. Therefore we should not have too many composite interfaces. One proposal is IDistribution, which combines the first four items above. Almost every algorithm would require this interface, but not all. 

The current design includes compatibility checking as methods of the message objects themselves. Having separate descriptor types (as in version 0.5) makes it too cumbersome to add new message types. In cases where a descriptor is needed, a prototype message object (such as a uniform distribution) is used. Prototype message objects also represent variable domains, for the purpose of compatibility checking. For example, a categorical domain is represented by a uniform categorical distribution, and the domain of positive real numbers is represented by a uniform Gamma distribution. This is potentially confusing, but it eliminates the need for separate domain descriptor types. To make domain checking simple, each domain should have a unique standard distribution type which is used to represent it.

Each message type operates on a particular domain type. The domain type appears in the interfaces for PointMass, Evaluate, and Sample. The domain type can be a value type or reference type; however, it should always have value semantics. For example, Gaussian distributions have domain Vector, while some other distribution may have domain Int32. You can usually find out the domain type of a message object by looking at the signature of its PointMass interface, if it has one. 

#### Value types vs. Reference types

The memory size of a distribution can vary greatly, from a single integer to multiple megabytes (such as in junction tree algorithms). Thus the library must support in-place mutation. A good practice is to put the result of a method into the target, e.g. `a.SetTo(b)`, because this works regardless of whether the target is a value type. Consider `a.SetTo(b)` versus `b.CopyTo(a)`. The first version works regardless of whether a is a value type. But the second version will not work if `a` is a value type.

Unfortunately, the SetTo approach is not always possible. In particular, Sample must return a domain object, which cannot be the target since Sample is a method on a message. There are several possibilities for the signature of Sample:

1. `DomainType Sample();` 
2. `void Sample(DomainType result);` 
3. `DomainType Sample(DomainType result);` 
4. `void Sample(ref DomainType result);`

The first version works but can be inefficient if you always create a new object. The second version does not work if DomainType is a value type. The third version combines these two, and works efficiently for all types. The fourth version is also efficient, though it is clumsy to use because of the ref argument, e.g. you cannot pass a property into it. Therefore, the third version seems best.

The first version could be efficient if the message object keeps a reference to the result and re-uses it. This makes the result object volatile. Unfortunately, this approach requires a new field on the message object, which is a permanent cost every algorithm must pay, even if they never call the Sample method. 

A related issue arises when a method requires a temporary workspace (such as `Gaussian.Sample` and `Gaussian.GetMean`). Following the above pattern, we could require the caller to pass in a pre-allocated workspace. Unfortunately, this makes the method signature vary from class to class. It also creates problems for late-binding, since a general caller would not know what sort of workspace to provide. 

#### Partial evaluation

Partial evaluation provides a modular approach to preserving workspaces across calls to a subroutine. For each routine that takes a workspace, such as `Sample(result, workspace)`, we create a partially-evaluated routine `SamplePrep()` which returns a delegate. `SamplePrep` allocates the workspace and stores it in the delegate. Invoking the delegate then calls `Sample` with the workspace. When the user is done sampling, they free the delegate, which also cleans up the workspace.

The advantage of this approach is that it is dynamic; the user only allocates the workspaces that they need. It also works nicely with late-binding, since the user does not have to know what sort of workspace to create. However, if we only want to call `Sample` once, this approach is overkill. In that case, we can have a simplified version, `Sample(result)`, which just allocates a workspace and calls `Sample(result, workspace)`.

Using this approach, we would have the following methods on a distribution that uses a workspace:

1. `DomainType Sample(DomainType result);` 
2. `DomainType Sample(DomainType result, object workspace);` 
3. `Sampler<DistributionType,DomainType> SamplePrep();`

Sampler has the following signature:

`DomainType Sampler(DistributionType distribution, DomainType result);`

In this design, the distribution argument, which is implicit in (1), is explicit in the delegate. This way you can apply the same delegate to sample from multiple (compatible) distributions. Note that the partially-evaluated routine (3) can be constructed mechanically from (1).
---
layout: default
---
[Infer.NET user guide](index.md) : [Infer.NET development](Infer.NET development.md)

## Infer.NET compiler design

This page documents the design of the Infer.NET compiler. This consists of a series of code transformations.

| **Transform** | **Summary** |
|-----------------------------|
| [Gate](Compiler/Gate transform.md) | Handles `if` and `switch` statements with stochastic arguments. |
| [Indexing](Compiler/Indexing transform.md) | Converts constant array indexing such as a[0] into a separate variable a_item0. |
| [Depth cloning](Compiler/Depth cloning transform.md) | Clones variables to ensure that each is used at a fixed depth. |
| [Replication](Compiler/Replication transform.md) | Replicates any variables which are referenced in a loop, so that each reference is a unique variable. |
| [Variable](Compiler/Variable transform.md) | Inserts variable factors, splitting each variable into a definition, use, and marginal. |
| [If cutting](Compiler/If cutting transform.md) | Splits `if` statements. |
| [Channel](Compiler/Channel transform.md) | Converts variables so that each reference becomes a unique channel variable, corresponding to an edge in the factor graph. |
| [Group](Compiler/Group transform.md) | Converts GroupMember attributes into ChannelPath attributes |
| Hybrid algorithm | Allows different algorithms to be applied to different parts of the graph. Inserts operators at algorithm boundaries to transform the messages of one algorithm into messages suitable for another. |
| [Message](Compiler/Message transform.md) | Transforms a model specified in terms of channels into the set of message passing operations required to perform inference in that model. |
| [Message optimisation](Compiler/Message optimisation transform.md) | Removes duplicate messages or redundant message operations. |
| [Loop cutting](Compiler/Loop cutting transform.md) | Splits `for` loops and promotes variables declared in loops to the top level. |
| [Dependency analysis](Compiler/Dependency analysis transform.md) | Determines dependencies between statements in the code. |
| [Pruning](Compiler/Pruning transform.md) | Removes statements whose result is never used by an output, and removes updates whose result is uniform. |
| [Iteration](Compiler/Iteration transform.md) | Creates `while(true)` loops around statement blocks with cyclic dependencies. Statements not in loops, and the loops themselves, are topologically sorted. |
| [Scheduling](Compiler/Scheduling transform.md) | Re-orders statements within `while(true)` loops to respect their dependency requirements. |
| [Dead code elimination](Compiler/Dead code transform.md) | Post-optimization of the schedule. Removes statements whose result is never used. |
| [Iterative process](Compiler/Iterative process transform.md) | Converts an inference method into a class which implements `IGeneratedAlgorithm`. Promotes local variables to fields and places parts of the inference into different methods. |
| Loop merging | Optimizes the generated code by merging consecutive for loops over the same range into one for loop. |
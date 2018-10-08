---
layout: default
---
[Infer.NET development](index.md)

## Compiler overview

This text is intended for people who have some experience of using Infer.NET and wish to know more about how the compiler works. Delving into the compiler in any detail will require knowledge of programming language concepts such as expressions and statements.

The compiler is architected as a chain of about 40 transforms. The input and output of every transform is an [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree) (AST) that can represent the majority of the C# 2.0 language. The tree is type bound (expressions know their type) and has a backing [higher-order abstract syntax graph](https://en.wikipedia.org/wiki/Higher-order_abstract_syntax) (variable references contain pointers to the variable definition as opposed to using names). Although the output of each transform is syntactically correct C#, typically it will not compile. Successful compilation of the output is only a requirement of the final transform.

The Infer.NET API produces a subset of C#, called model specification language (MSL), as input to the chain. The output of the chain is a single C# class implementing `IGeneratedAlgorithm`. The Infer.NET compiler can optionally invoke the C# compiler on the generated code, load the resulting assembly and return a reference to the generated type.

Frequently, additional information about a particular element in the code model must be communicated between separate transforms in the compiler. When this information doesn't fit easily into the AST, compiler attributes are used. They can be thought of much like normal attributes but instead of inheriting from Attribute, they implement our empty [ICompilerAttribute](../apiguide/api/Microsoft.ML.Probabilistic.Compiler.ICompilerAttribute.html) interface. They are held in a separate data structure from the AST and queried in a different way. There are about 70 different compiler attributes. Some are marker attributes with no data e.g. OperatorStatement and some are more complex data structures e.g. [DependencyInformation](https://github.com/dotnet/infer/blob/master/src/Compiler/Infer/CompilerAttributes/DependencyInformation.cs).

To visualise the changes that the transforms makes as the program passes through the compiler, we have the [Transform browser](../userguide/Transform browser.md).

Transforms are implemented using a depth-first recursive pattern by inheriting from [ShallowCopyTransform](../apiguide/api/Microsoft.ML.Probabilistic.Compiler.Transforms.ShallowCopyTransform.html). Most transforms are single pass. Some transforms cannot be performed in a single pass so use a pattern whereby an analysis transform is implemented as an inner class and executed prior to the main transform.

Compiler phases (transforms with links have up-to-date descriptions in their class comments): 

1. Pre-processing
    *   Normalise and annotate MSL.
    *   _Transforms: LoopUnrolling, [ModelAnalysis](https://github.com/dotnet/infer/blob/master/src/Compiler/Infer/Transforms/ModelAnalysisTransform.cs)_
2. Factor graph construction
    *   Output is a single method representing a factor graph. Each edge is a local variable with one definition and one use. Each factor is a function call. There are no stochastic containers (TODO define). Uses must occur after definitions.
    *   _Essential transforms: [StocAnalysis](https://github.com/dotnet/infer/blob/master/src/Compiler/Infer/Transforms/StocAnalysisTransform.cs), Variable, Channel_
    *   _Model dependent transforms: EqualityPropagation, Gate, Indexing, DepthCloning, Replication, IfCutting, DerivedVariable, Power_
    *   _Algorithm dependent transforms: Group_
3. Message passing
    *   Output is a single method. Each message is a local variable. Initialisation of messages is explicit. Each update is a function call to an operator method. The ordering of updates is arbitrary.
    *   _Transforms: Message, LoopCutting_
4. Optional optimisation (compiler flag)
    *   Output is of the same format as the message passing phase.
    *   _Transforms: CopyPropagation, Hoisting, LoopCutting_
5. Scheduling
    *   Output is a single method with statements correctly ordered and redundant statements removed. Any iterative loop is contained within a while block.
    *   _Transforms: DependencyAnalysis, Pruning, Iteration, [ForwardBackward](https://github.com/dotnet/infer/blob/master/src/Compiler/Infer/Transforms/ForwardBackwardTransform.cs), Scheduling_
6. Code gen and optimisation
    *   Output is final code for C# compilation. IterativeProcess splits up the inference code and packages it into an `IGeneratedAlgorithm`.
    *   Note that LoopMerging could be considered part of scheduling as it is required for some models
    *   _Transforms: LoopMerging2, DeadCode, IterativeProcess, LoopMerging, Local, ParallelFor_
	
Also see slides 20+ of the slide deck from [NIPS 2008](nips2008.pdf)

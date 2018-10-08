---
layout: default
---
[Infer.NET development](../index.md) : [Compiler transforms](../Compiler transforms.md)

## Dead code transform

This transform removes unnecessary duplicate statements from the code. Specifically, it removes statements which are re-executed before their result is used by another statement, assuming a sequential order of execution. These statements are called "dead". The transform assumes that statements have been annotated with DependencyInformation attributes. 
 
The transform works by constructing a dependency graph and looping through the statements in reverse order. At every point, it maintains the set of statements whose result has been used. Initially this Used set is the set of outputs. For each statement, if the statement is not in the Used set, then it is pruned. Otherwise, the statement is removed from the Used set, and all of its dependencies are added to the Used set. 
 
In case of a `while` loop, this algorithm is applied repeatedly to the loop body until convergence (i.e. as if the loop had been unrolled). This leads to the possibility that some statements become dead only after iterating this algorithm multiple times on the body. In other words, some statements are used by code after the loop and not used by code inside the loop. The transform removes such statements from the loop and places them immediately after the loop.
 
As a side-effect, this transform identifies the statements whose result is used between iterations of a `while` loop. This is useful for later optimizations so it is attached to the `while` loop as an InitializerSet attribute. Intuitively, this is the set of variables which need to be initialized on entry to the `while` loop.
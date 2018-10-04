---
layout: default
---
[Infer.NET development](../index.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Pruning transform

This transform removes unnecessary statements from the code. Specifically, it removes statements whose result is never eventually used by an output statement, and it removes update statements whose result is inferred to be uniform. The transform assumes that statements have been annotated with DependencyInformation attributes. 
 
The transform works by constructing a dependency graph and performing a depth-first search backward from the output statements. Any statement not reached by this search is pruned.
 
To infer which statements are uniform, the transform puts all statements onto a priority queue, and repeatedly removes statements whose result is NOT uniform. A statement on the queue is determined to be non-uniform if it meets the following conditions:

*  It is not annotated as uniform by its DependencyInformation.
*  It has no Requirement dependencies waiting on the queue.

After convergence, any statements left on the queue are assumed to be uniform, and pruned. Note that when these statements are pruned from the code, they must also be pruned from the DependencyInformation attributes of all other statements that might refer to them.

The uniform statements are actually removed first, before doing the depth-first search. This is important because it allows dependencies of uniform statements to be pruned. For example, suppose A requires B and C, B is uniform. This means that A is uniform and A will be pruned. As a result, C doesn't need to be computed (even if it might be non-uniform).
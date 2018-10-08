---
layout: default
---
[Infer.NET development](../index.md) : [Compiler transforms](../Compiler transforms.md)

## Iteration transform

This transform finds statements with cyclic dependencies between them and places them in `while(true)` loops. Statements not in loops, and the loops themselves, are then topologically sorted. Statements within `while` loops have arbitrary order. The transform assumes that statements have been annotated with DependencyInformation attributes. 
 
The transform works by constructing a dependency graph and finding the strongly-connected components of the graph. Each component is either a single statement or a set of statements having a cyclic dependency. The latter are converted into `while` loops. Note that the output may contain more than one `while` loop, and the `while` loops are always at the top level (they are never nested). The components are topologically sorted by depth-first search from the outputs.
 
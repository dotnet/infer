---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Calling Infer.NET from C++/CLI

Calling Infer.NET from C++/CLI is generally straightforward, but there are a few things worth noting. See the tutorial examples for instances of all of these.

*   There are no Infer.NET header files. You only need to reference the Infer.NET assemblies (in Visual Studio Solution Explorer, right-click the project to add references).
*   Much of the model construction API makes use of static methods, and so should use the C++ static method syntax using :: to separate class and method
*   Many (but not all) of the distribution types are value types and can be allocated on the stack. All the model constructs (Variable, VariableArray, and Range) are reference types and must be allocated using gcnew
*   C++/CLI has no equivalent to the using statement for automatic call to the IDisposable Dispose method. Therefore Infer.NET block statements such as switch blocks, if blocks and for blocks must explicitly close their blocks.
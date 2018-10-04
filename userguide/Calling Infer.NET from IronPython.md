---
layout: default
---
[Infer.NET User Guide](index.md)

## Calling Infer.NET from IronPython

This release of Infer.NET has been tested with **[Iron Python](http://www.ironpython.net/) 2.7** and this is the recommended version. It is also recommended that you use [**Python Tools for Visual Studio**](https://microsoft.github.io/PTVS/) which provides integrated editing and debugging within Visual Studio 2010.

IronPython exposes .NET concepts as Python constructs, and the [IronPython .NET documentation](http://ironpython.net/documentation/dotnet/) shows how to deal with various issues such type system compatibility and generics. It is important to be familiar with these issues as Infer.NET makes heavy use of generics. In general there is a direct translation between C# code and IronPython code with the following equivalences (other than the obvious differences):

*   Generic type arguments use square brackets rather than angle brackets
*   The with statement is used rather than the using statement
*   float in IronPython is equivalent to double in C#
    

You will see from the tutorial examples (linked below) how closely C# models can be mapped to IronPython models. However, for a few constructs, IronPython has difficulty determining the correct generic overload - for example the `InitialiseTo<>` method on a `Variable<>` object. In order to ameliorate this, Infer.NET comes with a wrapper ([IronPythonWrapper.py](Initialising and Creating Variables with the Ironpython Wrapper.md)) which provides some static methods to facilitate creating and initialising Variables and VariableArrays of fixed size. This wrapper, which is optional, also serves as an example for how to deal with other similar scenarios related to generic overloads.

### Running from a Console

Running Infer.NET within an IronPython console requires [referencing the Infer.NET DLLs and importing the relevant namespaces](Running Infer.NET from IronPython.md); this can be most easily accomplished by using the InferNet package folder distributed with Infer.NET which can be copied to the relevant package folders for different consoles (for example the IronPython console or the [Sho](http://research.microsoft.com/sho/) console).

### Running within Visual Studio

[**Python Tools for Visual Studio**](https://microsoft.github.io/PTVS/) (PTVS) provides integrated editing and debugging within Visual Studio 2010. An example solution containing an IronPython project is distributed as part of the Infer.NET installation. It assumes that you have both IronPython and PTVS installed. The example solution contains many of the tutorial examples. The code for these examples is shown in the following links.

### Tutorial Examples

*   [Two Coins in IronPython](Two coins in IronPython.md)
*   [Truncated Gaussian in IronPython](Truncated Gaussian in IronPython.md)
*   [Learning a Gaussian in IronPython](Learning a Gaussian in IronPython.md)
*   [Bayes point Machine in IronPython](Bayes point Machine in IronPython.md)
*   [Clinical Trial in IronPython](Clinical Trial in IronPython.md)
*   [Mixture of Gaussians in IronPython](Mixture of Gaussians in IronPython.md)
---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Calling Infer.NET from F\#

This release of Infer.NET is tested with version 2.0 of F# as included with Visual Studio 2010. As Infer.NET is a .NET API, it can be called directly from F#. Infer.NET also ships with a wrapper DLL along with its source code which can be found in the bin and Source folders in the installation folder. The use of this wrapper is optional but it does allow for more succinct code.

#### Porting existing C# code to F\#

If you are taking existing C# Infer.NET example code and porting it to F# there are a couple of issues to bear in mind:

*   `float` in F# is equivalent to `double` in C#
*   The `params` keyword in C# has no equivalent in F#. So a list of variables in some API calls may need to be converted to an F# array.

### F# Wrapper

The F# wrapper consists of a set of type declarations and functionality to make the experience of using Infer.NET more user friendly for the F# programmer. Infer.NET makes heavy use of generic classes methods and overrides, and F#'s implicit type inference has some difficulty disambiguating some Infer.NET types. Calls to some methods therefore need explicit specification of type arguments, and these types can get quite complex, especially, for example, for distribution arrays over jagged array domains. Therefore many type declarations are included in the F# wrapper to make such explicit typing much easier.

Another issue with calling Infer.NET from F# is the use of imperative constructs which are counter-intuitive for an F# programmer. Hence the wrapper provides functions to call these methods in a more natural way.

Complex jagged arrays of data are also difficult to create and assign values to and methods are provided to make this easier and allow assignment of values in a non imperative fashion.

A final issue is comparison operator overriding for `Variable<T>` objects. In C# these can be used to create new variables (for example a < b) whose result types are `Variable<bool>`s. F# does not recognise such overrides, so the wrapper provides alternative operators (for example a << b). The Wrapper is divided into a set of modules as described below.

#### The Modules

The F# wrapper is accessed by referencing the FSharpWrapper.dll and including the namespace Microsoft.ML.Probabilistic.FSharp in an F# script file, or by including references in the Solution Project of an F# program file. It contains the following modules, which can be accessed by calling moduleName.methodName with appropriate arguments, from an F# file. Alternatively use open moduleName to avoid referring to the moduleName each time a module is used.  

*   **FloatDistribution**: Provides type declarations for distributions over float-based array domains (float arrays, Vector arrays, and PositiveDefiniteMatrix arrays). Also provides variable types for these distribution array types which are useful for providing observed priors to your model.

*   **IntDistribution**:  Provides type declarations for distributions over integer array domains. Also provides variable types for these distribution array types which are useful for providing observed priors to your model.
*   **BoolDistribution**: Provides type declarations for distributions over boolean array domains. Also provides variable types for these distribution array types which are useful for providing observed priors to your model.
*   **Variable**: Provides methods for calling imperative structures such as ForEach statements, If Blocks and Switch Blocks. It also provides methods for creating and initialising VariableArrays.
*   **Operators**:  Provides a set of operator overloads for the comparison operators which can be used to compare objects of type `Variable<'a> ` with `Variable<'a>` or with type 'a. The module name is not needed when using such operators.
*   **Inference**: Provides methods of calling Inference methods without the need for complex explicit typing.
*   **Factors:** Provides a mechanism for adding new Factors in F#, through a CreateDelegate method which can then be registered and used in a model.
*   Array2D: Provides a method for creating 2D .NET arrays from lists of lists

#### Using the Methods in the F# Wrapper

A following description of how to use the methods is divided up by commonly used functionality needed to build models in Infer.NET, which can make use of methods and type declarations from several modules.

*   [Arrays in F#](Arrays in FSharp.md)
*   [Variable Arrays in F#](Variable Arrays in FSharp.md)
*   [Imperative Statement Blocks in F#](Imperative Statement Blocks in FSharp.md)
*   [Inference in F#](Inference in FSharp.md)
*   [Operator Overloading in F#](Operator Overloading in FSharp.md)
*   [Creating delegates in F#](Creating delegates in FSharp.md)
*   [Adding New Factors in F#](Adding New Factors in FSharp.md)

#### Tutorials

These tutorials show how to use the F# Wrapper modules to rewrite the C# tutorials provided for Infer.NET in F#

*   [Two coins](Two coins.md)
*   [Efficient truncated Gaussian](Efficient truncated Gaussian.md)
*   [Learning a Gaussian with Ranges](Learning a Gaussian with Ranges.md)
*   [Clinical trial](Clinical trial.md)
*   [Bayes point machine](Bayes point machine.md)
*   [Mixture of Gaussians](Mixture of Gaussians.md)

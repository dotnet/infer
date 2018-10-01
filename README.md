**Infer&#46;NET** is a framework for running Bayesian inference in graphical models. It can also be used for probabilistic programming.

One can use Infer&#46;NET to solve many different kinds of machine learning problems - from standard problems like [classification](https://microsoft.github.io/Infer.NET/userguide/docs/Infer.NET%20Learners%20-%20Bayes%20Point%20Machine%20classifiers),
[recommendation](https://microsoft.github.io/Infer.NET/userguide/docs/Infer.NET%20Learners%20%20Matchbox%20recommender) or [clustering](https://microsoft.github.io/Infer.NET/userguide/docs/Mixture%20of%20Gaussians%20tutorial) through to [customised solutions to domain-specific problems](https://microsoft.github.io/Infer.NET/userguide/docs/Click%20through%20model%20sample). 

**Infer&#46;NET** has been used in a wide variety of domains including information retrieval, bioinformatics, epidemiology, vision, 
and many others.

# Contents

- [Structure of repository](#structure-of-repository)
- [Build and test](#build-and-test)
    - [Windows](##windows)
    - [Linux](##linux)

# Structure of repository

* The Visual Studio solution `Infer2.sln` in the root of the repository contains all Infer&#46;NET components, unit tests and sample programs from the folders described below.

* `src/`

  * `Compiler` contains the Infer&#46;NET Compiler project which takes model descriptions written using the Infer&#46;NET API, and converts them into inference code. The project also contains utility methods for visualization of the generated code.

  * `Csoft` is an experimental feature that allows to express probabilistic models in a subset of the C# language. You can find many unit tests of `Csoft` models in the `Tests` project marked with `Category: CsoftModel` trait.

  * `Examples` contains C# projects that illustrate how to use Infer&#46;NET to solve a variety of different problems. 

    * `ClickThroughModel` - a web search example of converting a sequence of clicks by the user into inferences about the relevance of documents.

    * `ClinicalTrial` - the clinical trial tutorial example with an interactive user interface.`

    * `InferNET101` - samples from Infer&#46;NET 101 introduction to the basics of Microsoft Infer&#46;NET programming.
  
    * `ImageClassifier` - an image search example of classifying tagged images.

    * `LDA` - this example provides Infer&#46;NET implementations of the popular LDA model for topic modeling. The implementations pay special attention to scalability with respect to vocabulary size, and with respect to the number of documents. As such, they provide good examples for how to scale Infer&#46;NET models in general.

    * `MontyHall` - an Infer&#46;NET implementation of the Monty Hall problem, along with a graphical user interface.

  * `FSharpWrapper` is a wrapper project that hides some of the generic constructs in the Infer&#46;NET API allowing simpler calls to the Infer&#46;NET API from standard F#.

  * `IronPythonWrapper` contains wrapper for calling Infer&#46;NET from the [IronPython](https://ironpython.net/) programming language and tests for the wrapper. Please refer to [README.md](IronPythonWrapper/README.md) for more information.

  * `Learners` folder contains Visual Studio projects for complete machine learning applications including classification and recommendation. You can read more about Learners [here](https://microsoft.github.io/Infer.NET/userguide/docs/Infer.NET%20Learners.md).

  * `Runtime` - is a C# project with classes and methods needed to execute the inference code.

  * `Tutorials` contains [Examples Browser](https://microsoft.github.io/Infer.NET/userguide/docs/The%20Example%20Browser.md) project with simple examples that provide a step-by-step introduction to Infer.NET.

* `test/`

  * `TestApp` contains C# console application for quick invocation and debugging of variouse Infer&#46;NET components.

  * `TestFSharp` is an F# console project for smoke testing of Infer&#46;NET F# wrapper.

  * `TestPublic` contains scenario tests for tutorial code. These tests are a part of the PR and nightly builds.

  * `Tests` - main unit test project containing thousands of tests. These tests are a part of the PR and nightly builds. The folder `Tests\Vibes` contains MATLab scripts that compare Infer&#46;NET to the [VIBES](https://vibes.sourceforge.net/) package. Running them requires `Vibes2_0.jar` (can be obtained on the [VIBES](https://vibes.sourceforge.net/) website) to be present in the same folder.

  * `Learners` folder contains the unit tests and the test application for `Learners` (see above).

* `docs` folder contains the scripts for bulding API documentation and for updating https://microsoft.github.io/Infer.NET. Please refer to [README.md](docs/README.md) for more details.

# Build and Test

Infer&#46;NET is cross platform and supports .NET Framework 4.5.2 and Mono 5.0. Unit tests are written using the [XUnit](https://xunit.github.io/) framework.

## Windows

### Prerequisites

**Visual Studio 2017.**
If you don't have Visual Studio 2017, you can install the free [Visual Studio 2015/2017 Community](http://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx).

### Build and test
You can load `Infer2.sln` solution located in the root of repository into Visual Studio and build all libraries and samples.

**NB!** The solution has a number of build configurations that allows building either for all supported frameworks simultaneously or only for a specific one, but in order for Visual Studio to behave correctly, the solution needs to be closed and re-opened after switching between such configurations.

Unit tests are available in `Test Explorer` window. Normally you should see tests from 3 projects: `Tests`, `PublicTests` and `LearnersTest`. Note, that some of the tests are categorized, and those falling in the `OpenBug` or `BadTest` categories are not supposed to succeed.

## Linux 

Almost all components of Infer&#46;NET run on Mono and/or .net core 2.0 except some visualizations in `Compiler` project and sample applications that use WPF.

### Prerequisites

1. **[Mono and MSBuild](https://www.mono-project.com/download/stable/#download-lin)** (version 5.0 and higher)
1. **[.NET Core 2.0 SDK](https://www.microsoft.com/net/download/linux-package-manager/ubuntu18-04/sdk-2.1.202)**
1. **[NuGet](https://docs.microsoft.com/en-us/nuget/install-nuget-client-tools)** package manager

### Build 

1. Restore required NuGet packages after cloning the repository. Execute the following command in the root directory of the repository:
    ```bash
    msbuild /p:MonoSupport=true /restore Infer2.sln
    ```

2. Then build the entire solution (or individual projects) using the following commands:
    ```bash
    msbuild /p:MonoSupport=true Infer2.sln
    ```
    or
    ```bash
    msbuild /p:MonoSupport=true src/Runtime/Runtime.csproj
    ```
    These commands set the `MonoSupport` property to `true`. It excludes code that uses WPF from build.

### Run unit tests

In order to run unit tests, build the test project and execute one of the following commands:
```bash
mono ~/.nuget/packages/xunit.runner.console/2.3.1/tools/net452/xunit.console.exe <path to net452 assembly with tests> <filter>
```
```bash
dotnet ~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.0/xunit.console.dll <path to netcoreapp2.0 assembly with tests> <filter>
```

There are three test assemblies in the solution:

- **Infer.Tests.dll** in the folder `test/Tests`. 
- **TestPublic.dll** in the folder `test/TestPublic`.
- **Infer.Learners.Tests.dll** in the folder `test/Learners/LearnersTests`. 

Depending on the build configuration, the assemblies will be located in the `bin/Debug` or `bin/Release` subdirectories
of the test project.

`<filter>` is a rule to chose what tests will be run. You can specify them
using `-trait Category=<category>` and `-notrait Category=<category>` parts
of `<filter>`. The former selects tests of
the given category, while the latter selects test that don't belong to the given
category. These can be combined: several `-trait` options mean that _at least one_ of the listed traits has to be present, while several `-notrait` options mean that _none_ of such traits can be present on the filtered tests.

Runner executes tests in parallel by default. However, some test category must be run
sequentially. Such categories are:
- _Performance_
- _DistributedTest_
- _CsoftModel_
- _ModifiesGlobals_

Add the `-parallel none` argument to run them.

_CompilerOptionsTest_ is a category for long running tests, so, for quick
testing you must filter these out by `-notrait`.
_BadTest_ is a category of tests that must fail.
_OpenBug_ is a category of tests that can fail.


An example of quick testing of `Infer.Tests.dll` in `Debug` configuration after changing working directory to
the `Tests` project looks like:
```bash
mono ~/.nuget/packages/xunit.runner.console/2.3.1/tools/net452/xunit.console.exe bin/Debug/net452/Infer.Tests.dll -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance

mono ~/.nuget/packages/xunit.runner.console/2.3.1/tools/net452/xunit.console.exe bin/Debug/net452/Infer.Tests.dll -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTests -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -parallel none
```

To run the same set of tests on .net core:
```bash
dotnet ~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.0/xunit.console.dll bin/Debug/netcoreapp2.0/Infer.Tests.dll -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance

dotnet ~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.0/xunit.console.dll bin/Debug/netcoreapp2.0/Infer.Tests.dll -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTests -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -parallel none
```

Helper scripts `monotest.sh` and `netcoretest.sh` for running unit tests on Mono and .net core respectively are located in the `test` folder.
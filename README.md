# Infer&#46;NET

**Infer&#46;NET** is a framework for running Bayesian inference in graphical models. It can also be used for probabilistic programming.

One can use Infer&#46;NET to solve many different kinds of machine learning problems - from standard problems like [classification](https://dotnet.github.io/infer/userguide/Learners/Bayes%20Point%20Machine%20classifiers.html),
[recommendation](https://dotnet.github.io/infer/userguide/Learners/Matchbox%20recommender.html) or [clustering](https://dotnet.github.io/infer/userguide/Mixture%20of%20Gaussians%20tutorial.html) through to [customised solutions to domain-specific problems](https://dotnet.github.io/infer/userguide/Click%20through%20model%20sample.html). 

**Infer&#46;NET** has been used in a wide variety of domains including information retrieval, bioinformatics, epidemiology, vision, 
and many others.

## Contents

- [Build Status](#build-status)
- [Installation](#installation)
- [Documentation](#documentation)
- [Structure of Repository](#structure-of-repository)
- [Build and Test](#build-and-test)
    - [Windows](##windows)
    - [Linux and macOS](##linux-and-macos)
- [Contributing](#contributing)
- [License](#license)
- [.NET Foundation](#.net-foundation)

## Build Status

|    | Debug | Release |
|:---|----------------:|------------------:|
|**Windows**|[![Win Debug](https://msrcambridge.visualstudio.com/Infer.NET/_apis/build/status/Nightly%20Windows%20Debug)](https://msrcambridge.visualstudio.com/Infer.NET/_build/latest?definitionId=135)|[![Win Release](https://msrcambridge.visualstudio.com/Infer.NET/_apis/build/status/Nightly%20Windows%20Release)](https://msrcambridge.visualstudio.com/Infer.NET/_build/latest?definitionId=134)|
|**Linux**|[![Linux Debug](https://msrcambridge.visualstudio.com/Infer.NET/_apis/build/status/Nightly%20Linux%20Debug)](https://msrcambridge.visualstudio.com/Infer.NET/_build/latest?definitionId=137)|[![Linux Release](https://msrcambridge.visualstudio.com/Infer.NET/_apis/build/status/Nightly%20Linux%20Release)](https://msrcambridge.visualstudio.com/Infer.NET/_build/latest?definitionId=136)|
|**macOS**|[![macOS Debug](https://msrcambridge.visualstudio.com/Infer.NET/_apis/build/status/Nightly%20macOS%20Debug)](https://msrcambridge.visualstudio.com/Infer.NET/_build/latest?definitionId=139)|[![macOS Release](https://msrcambridge.visualstudio.com/Infer.NET/_apis/build/status/Nightly%20macOS%20Release)](https://msrcambridge.visualstudio.com/Infer.NET/_build/latest?definitionId=138)|

## Installation

Infer&#46;NET runs on Windows, Linux, and macOS - any platform where [.NET Core 2.0](https://github.com/dotnet/core) is available.

First ensure you have installed [.NET Core 2.0](https://www.microsoft.com/net/download/dotnet-core/2.0). Infer&#46;NET also works on the .NET Framework 4.6.1 and above.

Once you have an app, you can install the Infer&#46;NET NuGet package(s) from the .NET Core CLI using:
```
dotnet add package Microsoft.ML.Probabilistic
dotnet add package Microsoft.ML.Probabilistic.Compiler
dotnet add package Microsoft.ML.Probabilistic.Learners
```

or from the NuGet package manager:
```powershell
Install-Package Microsoft.ML.Probabilistic
Install-Package Microsoft.ML.Probabilistic.Compiler
Install-Package Microsoft.ML.Probabilistic.Learners
Install-Package Microsoft.ML.Probabilistic.Visualizers.Windows
```

Or alternatively you can add the Microsoft.ML.Probabilistic.* package(s) from within Visual Studio's NuGet package manager or via [Paket](https://github.com/fsprojects/Paket).

There currently are [four maintained Infer.NET nuget packages](https://www.nuget.org/packages?q=Microsoft.ML.Probabilistic):

1. `Microsoft.ML.Probabilistic` contains classes and methods needed to execute the inference code.
1. `Microsoft.ML.Probabilistic.Compiler` contains the Infer&#46;NET Compiler, which takes model descriptions written using the Infer&#46;NET API and converts them into inference code. It also contains utilities for the visualization of the generated code.
1. `Microsoft.ML.Probabilistic.Learners` contains complete machine learning applications including a classifier and a recommender system.
1. `Microsoft.ML.Probabilistic.Visualizers.Windows` contains an alternative .NET Framework and Windows specific set of visualization tools for exploring and analyzing models.

## Documentation

Documentation can be found on the [Infer&#46;NET website](https://dotnet.github.io/infer/userguide/).

## Structure of Repository

* The Visual Studio solution `Infer2.sln` in the root of the repository contains all Infer&#46;NET components, unit tests and sample programs from the folders described below.

* `src/`

  * `Compiler` contains the Infer&#46;NET Compiler project which takes model descriptions written using the Infer&#46;NET API, and converts them into inference code. The project also contains utility methods for visualization of the generated code.

  * `Csoft` is an experimental feature that allows to express probabilistic models in a subset of the C# language. You can find many unit tests of `Csoft` models in the `Tests` project marked with `Category: CsoftModel` trait.

  * `Examples` contains C# projects that illustrate how to use Infer&#46;NET to solve a variety of different problems. 

    * `ClickThroughModel` - a web search example of converting a sequence of clicks by the user into inferences about the relevance of documents.

    * `ClinicalTrial` - the clinical trial tutorial example with an interactive user interface.

    * `InferNET101` - samples from Infer&#46;NET 101 introduction to the basics of Microsoft Infer&#46;NET programming.
  
    * `ImageClassifier` - an image search example of classifying tagged images.

    * `LDA` - this example provides Infer&#46;NET implementations of the popular LDA model for topic modeling. The implementations pay special attention to scalability with respect to vocabulary size, and with respect to the number of documents. As such, they provide good examples for how to scale Infer&#46;NET models in general.

    * `MontyHall` - an Infer&#46;NET implementation of the Monty Hall problem, along with a graphical user interface.

    * `MotifFinder` - an Infer&#46;NET implementation of a simple model for finding motifs in nucleotide sequences, which constitutes an important problem in bioinformatics.

  * `FSharpWrapper` is a wrapper project that hides some of the generic constructs in the Infer&#46;NET API allowing simpler calls to the Infer&#46;NET API from standard F#.

  * `IronPythonWrapper` contains wrapper for calling Infer&#46;NET from the [IronPython](https://ironpython.net/) programming language and tests for the wrapper. Please refer to [README.md](src/IronPythonWrapper/README.md) for more information.

  * `Learners` folder contains Visual Studio projects for complete machine learning applications including classification and recommendation. You can read more about Learners [here](https://dotnet.github.io/infer/userguide/Infer.NET%20Learners.html).

  * `Runtime` - is a C# project with classes and methods needed to execute the inference code.

  * `Tutorials` contains [Examples Browser](https://dotnet.github.io/infer/userguide/The%20examples%20browser.html) project with simple examples that provide a step-by-step introduction to Infer.NET.

  * `Visualizers/Windows` contains an alternative .NET Framework and Windows specific set of visualization tools for exploring and analyzing the code generated by the `Compiler`.

* `test/`

  * `TestApp` contains C# console application for quick invocation and debugging of variouse Infer&#46;NET components.

  * `TestFSharp` is an F# console project for smoke testing of Infer&#46;NET F# wrapper.

  * `TestPublic` contains scenario tests for tutorial code. These tests are a part of the PR and nightly builds.

  * `Tests` - main unit test project containing thousands of tests. These tests are a part of the PR and nightly builds. The folder `Tests\Vibes` contains MATLab scripts that compare Infer&#46;NET to the [VIBES](https://vibes.sourceforge.net/) package. Running them requires `Vibes2_0.jar` (can be obtained on the [VIBES](https://vibes.sourceforge.net/) website) to be present in the same folder.

  * `Learners` folder contains the unit tests and the test application for `Learners` (see above).

* `build` folder contains the YAML definitions for the Continuous Integration builds and the specification files for the nuget packages.

* `docs` folder contains the scripts for building API documentation and for updating https://dotnet.github.io/infer. Please refer to [README.md](docs/README.md) for more details.

## Build and Test

Infer&#46;NET is cross platform and supports .NET Framework 4.6.1, .NET Core 2.0, and Mono 5.0. Unit tests are written using the [XUnit](https://xunit.github.io/) framework.

All of the Infer&#46;NET libraries target .NET Standard 2.0. Projects that produce executables (including test projects) mostly target .NET Framework 4.6.1, .NET Core 2.0, or both depending on build configuration:

| Configurations | Targeted Frameworks |
|:---|---:|
| Debug, Release | both .NET Framework 4.6.1 and .NET Core 2.0 |
| DebugFull, ReleaseFull | .NET Framework 4.6.1 only |
| DebugCore, ReleaseCore | .NET Core 2.0 only |


### Windows

#### Prerequisites

**Visual Studio 2017.**
If you don't have Visual Studio 2017, you can install the free [Visual Studio 2017 Community](http://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx).

#### Build and test
You can load `Infer2.sln` solution located in the root of repository into Visual Studio and build all libraries and samples.

**NB!** The solution has a number of build configurations that allows building either for all supported frameworks simultaneously or only for a specific one, but in order for Visual Studio to behave correctly, the solution needs to be closed and re-opened after switching between such configurations.

Unit tests are available in `Test Explorer` window. Normally you should see tests from 3 projects: `Tests`, `PublicTests` and `LearnersTest`. Note, that some of the tests are categorized, and those falling in the `OpenBug` or `BadTest` categories are not supposed to succeed.

### Linux and macOS

All components of Infer&#46;NET and almost all sample projects run on .NET Core 2.0 and/or Mono except sample applications that use WPF.

#### Prerequisites

* **[.NET Core 2.0 SDK](https://www.microsoft.com/net/download/dotnet-core/2.0)** to build and run .NET Standard and .NET Core projects

  and, optionally,

* **[Mono](https://www.mono-project.com/download/stable/)** (version 5.0 and higher) and **[NuGet](https://docs.microsoft.com/en-us/nuget/install-nuget-client-tools)** package manager to build and run .NET Framework 4.6.1 projects that don't use WPF (there're some examples that use Win Forms and, therefore, don't run on .NET Core, but can be built and run with Mono; there's also Visualizers/Windows project mentioned above that can be built with Mono using `/p:MonoSupport=true`)

#### Build 

To build .NET Standard libraries and .NET Core executables, run in the root of the repository either
```bash
dotnet build -c DebugCore Infer2.sln
```
to build debug assemblies, or
```bash
dotnet build -c ReleaseCore Infer2.sln
```
to build release assemblies.

The corresponding commands to build .NET Standard libraries and .NET Framework executables with Mono are
```bash
msbuild /c:DebugFull /p:MonoSupport=true /restore Infer2.sln
```
and
```bash
msbuild /c:ReleaseFull /p:MonoSupport=true /restore Infer2.sln
```
Please, expect build failure messages about examples that use WPF GUI. Libraries and executables that don't reference WPF should build, though.

#### Run unit tests

In order to run unit tests, build the test project and execute one of the following commands:
```bash
dotnet ~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.0/xunit.console.dll <path to netcoreapp2.0 assembly with tests> <filter>
```
```bash
mono ~/.nuget/packages/xunit.runner.console/2.3.1/tools/net452/xunit.console.exe <path to net461 assembly with tests> <filter>
```

There are three test assemblies in the solution:

- **Microsoft.ML.Probabilistic.Tests.dll** in the folder `test/Tests`. 
- **TestPublic.dll** in the folder `test/TestPublic`.
- **Microsoft.ML.Probabilistic.Learners.Tests.dll** in the folder `test/Learners/LearnersTests`. 

Depending on the build configuration and targeted framework, the assemblies will be located in the `bin/Debug<Core|Full>/<netcoreapp2.0|net461>` or `bin/Release<Core|Full>/<netcoreapp2.0|net461>` subdirectories
of the test project.

`<filter>` is a rule to choose what tests will be run. You can specify them
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


An example of quick testing of `Microsoft.ML.Probabilistic.Tests.dll` in `Debug` configuration after changing working directory to
the `Tests` project looks like:
```bash
dotnet ~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.0/xunit.console.dll bin/DebugCore/netcoreapp2.0/Microsoft.ML.Probabilistic.Tests.dll -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance

dotnet ~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.0/xunit.console.dll bin/DebugCore/netcoreapp2.0/Microsoft.ML.Probabilistic.Tests.dll -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTests -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -parallel none
```
To run the same set of tests on Mono:

```bash
mono ~/.nuget/packages/xunit.runner.console/2.3.1/tools/net452/xunit.console.exe bin/DebugFull/net461/Microsoft.ML.Probabilistic.Tests.dll -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance

mono ~/.nuget/packages/xunit.runner.console/2.3.1/tools/net452/xunit.console.exe bin/DebugFull/net461/Microsoft.ML.Probabilistic.Tests.dll -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTests -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -parallel none
```

Helper scripts `netcoretest.sh` and `monotest.sh` for running unit tests on .NET Core and Mono respectively are located in the `test` folder.

## Contributing

We welcome contributions! Please review our [contribution guide](CONTRIBUTING.md).

## License

Infer&#46;NET is licensed under the [MIT license](LICENSE.txt).

## .NET Foundation

Infer&#46;NET is a [.NET Foundation](https://www.dotnetfoundation.org/projects) project.
It's also a part of [ML.NET](https://github.com/dotnet/machinelearning) machine learning framework.

There are many .NET related projects on GitHub.

- [.NET home repo](https://github.com/Microsoft/dotnet) - links to 100s of .NET projects, from Microsoft and the community.

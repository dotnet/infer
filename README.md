# Infer&#46;NET

**Infer&#46;NET** is a framework for running Bayesian inference in graphical models. It can also be used for probabilistic programming.

One can use Infer&#46;NET to solve many different kinds of machine learning problems - from standard problems like [classification](https://dotnet.github.io/infer/userguide/Learners/Bayes%20Point%20Machine%20classifiers.html),
[recommendation](https://dotnet.github.io/infer/userguide/Learners/Matchbox%20recommender.html) or [clustering](https://dotnet.github.io/infer/userguide/Mixture%20of%20Gaussians%20tutorial.html) through to [customised solutions to domain-specific problems](https://dotnet.github.io/infer/userguide/Click%20through%20model%20sample.html). 

**Infer&#46;NET** has been used in a wide variety of domains including information retrieval, bioinformatics, epidemiology, vision, 
and many others.

## Contents

- [Build Status](#build-status)
- [Installing pre-built binaries](#installing-pre-built-binaries)
- [Documentation](#documentation)
- [Structure of Repository](#structure-of-repository)
- [Building Infer.NET from its source code](#building-infernet-from-its-source-code)
- [Contributing](#contributing)
- [License](#license)
- [.NET Foundation](#.net-foundation)

## Build Status

|    | Release |
|:---|------------------:|
|**Windows**|[![Win Release](https://msrcambridge.visualstudio.com/Infer.NET/_apis/build/status/Nightly%20Windows%20Release)](https://msrcambridge.visualstudio.com/Infer.NET/_build/latest?definitionId=134)|
|**Linux**|[![Linux Release](https://msrcambridge.visualstudio.com/Infer.NET/_apis/build/status/Nightly%20Linux%20Release)](https://msrcambridge.visualstudio.com/Infer.NET/_build/latest?definitionId=136)|
|**macOS**|[![macOS Release](https://msrcambridge.visualstudio.com/Infer.NET/_apis/build/status/Nightly%20macOS%20Release)](https://msrcambridge.visualstudio.com/Infer.NET/_build/latest?definitionId=138)|

## Installing pre-built binaries

Binaries for Infer.NET are located on [nuget.org](https://www.nuget.org/packages?q=Microsoft.ML.Probabilistic).  These binaries are cross-platform and work anywhere that .NET is supported, so there is no need to select your platform.  The core packages target .NET Standard 2.0, making them useable from any project that targets .NET framework version 4.6.1 or .NET Core 3.1, as explained at [.NET implementation support](https://docs.microsoft.com/en-us/dotnet/standard/net-standard).  You do not need to clone the GitHub repository to use the pre-built binaries.

There currently are [four maintained Infer.NET nuget packages](https://www.nuget.org/packages?q=Microsoft.ML.Probabilistic):

1. `Microsoft.ML.Probabilistic` contains classes and methods needed to execute the inference code.
1. `Microsoft.ML.Probabilistic.Compiler` contains the Infer&#46;NET Compiler, which takes model descriptions written using the Infer&#46;NET API and converts them into inference code. It also contains utilities for the visualization of the generated code.
1. `Microsoft.ML.Probabilistic.Learners` contains complete machine learning applications including a classifier and a recommender system.
1. `Microsoft.ML.Probabilistic.Visualizers.Windows` contains an alternative .NET Framework and Windows specific set of visualization tools for exploring and analyzing models.

NuGet packages do not need to be manually downloaded.  Instead, you add the package name to your project file, and the binaries are downloaded automatically when the project is compiled.  Most code editors have an option to add a NuGet package reference to an existing project file.  For example, in [Visual Studio 2017 for Windows](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio), you select `Project -> Manage NuGet packages`.

[.NET Core 3.1](https://www.microsoft.com/net/download/) provides command-line tools for creating and editing project files.
Using the command line, you can add a NuGet package reference to an existing project file with:
```
dotnet add package Microsoft.ML.Probabilistic
dotnet add package Microsoft.ML.Probabilistic.Compiler
dotnet add package Microsoft.ML.Probabilistic.Learners
```

## Tutorials and Examples

There is a getting started guide on [docs.microsoft.com](https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/matchup-app-infer-net).

More tutorials and examples can be found on the [Infer&#46;NET website](https://dotnet.github.io/infer/userguide/Infer.NET%20tutorials%20and%20examples.html).

## Documentation

Documentation can be found on the [Infer&#46;NET website](https://dotnet.github.io/infer/userguide/).

## Structure of Repository

* The Visual Studio solution `Infer.sln` in the root of the repository contains all Infer&#46;NET components, unit tests and sample programs from the folders described below.

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

  * `TestApp` contains C# console application for quick invocation and debugging of various Infer&#46;NET components.

  * `TestFSharp` is an F# console project for smoke testing of Infer&#46;NET F# wrapper.

  * `TestPublic` contains scenario tests for tutorial code. These tests are a part of the PR and nightly builds.

  * `Tests` - main unit test project containing thousands of tests. These tests are a part of the PR and nightly builds. The folder `Tests\Vibes` contains MATLab scripts that compare Infer&#46;NET to the [VIBES](https://vibes.sourceforge.net/) package. Running them requires `Vibes2_0.jar` (can be obtained on the [VIBES](https://vibes.sourceforge.net/) website) to be present in the same folder.

  * `Learners` folder contains the unit tests and the test application for `Learners` (see above).

* `build` folder contains the YAML definitions for the Continuous Integration builds and the specification files for the nuget packages.

* `docs` folder contains the scripts for building API documentation and for updating https://dotnet.github.io/infer. Please refer to [README.md](docs/README.md) for more details.

## Building Infer.NET from its source code

Please, refer to our [building guide](BUILDING.md).

## Contributing

We welcome contributions! Please review our [contribution guide](CONTRIBUTING.md).

When submitting pull request with changed or added factor, please make sure you updated factor documentation as described [here](docs/README.md#Documenting-Factors). 


## License

Infer&#46;NET is licensed under the [MIT license](LICENSE.txt).

## .NET Foundation

Infer&#46;NET is a [.NET Foundation](https://www.dotnetfoundation.org/projects) project.
It's also a part of [ML.NET](https://github.com/dotnet/machinelearning) machine learning framework.

There are many .NET related projects on GitHub.

- [.NET home repo](https://github.com/Microsoft/dotnet) - links to 100s of .NET projects, from Microsoft and the community.

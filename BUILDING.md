# Building Infer.NET from its source code

- [Building with Visual Studio 2017](#building-with-visual-studio-2017)
- [Building from the command line](#building-from-the-command-line)

To build from source, you must first clone the repository.
Next decide whether you want to use a code editor like Visual Studio (recommended) or the command line.
When building, you must select a configuration.
All of the Infer&#46;NET libraries target .NET Standard 2.0. Projects that produce executables (including test projects) mostly target .NET Framework 4.6.1, .NET Core 2.1, or both depending on build configuration:

| Configurations | Targeted Frameworks |
|:---|---:|
| Debug, Release | both .NET Framework 4.6.1 and .NET Core 2.1 |
| DebugFull, ReleaseFull | .NET Framework 4.6.1 only |
| DebugCore, ReleaseCore | .NET Core 2.1 only |


## Building with Visual Studio 2017

1. If you don't have Visual Studio 2017, you can install the free [Visual Studio 2017 Community](https://visualstudio.microsoft.com/vs/community/).
1. Start Visual Studio.
1. Select `File -> Open -> Project/Solution` and open the `Infer.sln` solution file located in your cloned repository.
1. Select a build configuration using `Build -> Configuration Manager...`.  When switching between configurations that change the targeted frameworks, Visual Studio currently requires you to close and re-open the solution file using `File -> Close Solution` and `File -> Open`.
1. Compile using `Build -> Build Solution`.
1. At this point, you can play with the [tutorials and examples](https://dotnet.github.io/infer/userguide/Infer.NET%20tutorials%20and%20examples.html), or run all tests to verify the installation.  Run the tutorials by setting the startup project to `Tutorials`.  If your configuration is `DebugFull` or `ReleaseFull`, you will get the [Examples Browser](https://dotnet.github.io/infer/userguide/The%20examples%20browser.html).  Otherwise, edit `src/Tutorials/RunMe.cs` to see different tutorials.  Run an example by setting the startup project to that example.
1. To run all tests, open the test explorer using `Test -> Windows -> Test Explorer`.
1. In the test explorer search bar, type `-Trait:"BadTest" -Trait:"OpenBug" -Trait:"CompilerOptionsTest" -Trait:"Performance"` to exclude long-running tests and tests that are not supposed to succeed.
1. Click `Run All`.

## Building from the command line

The core components of Infer&#46;NET run on .NET Core 2.1.  Some optional code, such as the [Examples Browser](https://dotnet.github.io/infer/userguide/The%20examples%20browser.html), use [Windows Forms](https://docs.microsoft.com/en-us/dotnet/framework/winforms/) and therefore require .NET framework or Mono. 
Some samples, such as the [Monty Hall problem](https://dotnet.github.io/infer/userguide/Monty%20Hall%20problem.html), use [WPF](https://docs.microsoft.com/en-us/visualstudio/designers/introduction-to-wpf) and therefore require .NET framework.

### Prerequisites

* **[.NET Core 2.1 SDK](https://www.microsoft.com/net/download/)** to build and run .NET Standard and .NET Core projects.

* (Optional) On Windows, the **[.NET framework developer pack](https://www.microsoft.com/net/download)**.  On other platforms, **[Mono](https://www.mono-project.com/download/stable/)** (version 5.0 and higher) and the **[NuGet](https://docs.microsoft.com/en-us/nuget/install-nuget-client-tools)** package manager.

### Build 

To build .NET Standard libraries and .NET Core executables, run in the root of the repository either
```bash
dotnet build -c DebugCore Infer.sln
```
to build debug assemblies, or
```bash
dotnet build -c ReleaseCore Infer.sln
```
to build release assemblies.

The corresponding commands to build .NET Standard libraries and .NET Framework executables with Mono are
```bash
msbuild /c:DebugFull /p:MonoSupport=true /restore Infer.sln
```
and
```bash
msbuild /c:ReleaseFull /p:MonoSupport=true /restore Infer.sln
```
Please, expect build failure messages about examples that use WPF GUI. Libraries and executables that don't reference WPF should build, though.

### Run unit tests

Unit tests are written using the [XUnit](https://xunit.github.io/) framework.
In order to run unit tests, build the test project and execute one of the following commands:
```bash
dotnet ~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.1/xunit.console.dll <path to netcoreapp2.1 assembly with tests> <filter>
```
```bash
mono ~/.nuget/packages/xunit.runner.console/2.3.1/tools/net452/xunit.console.exe <path to net461 assembly with tests> <filter>
```

There are three test assemblies in the solution:

- **Microsoft.ML.Probabilistic.Tests.dll** in the folder `test/Tests`. 
- **TestPublic.dll** in the folder `test/TestPublic`.
- **Microsoft.ML.Probabilistic.Learners.Tests.dll** in the folder `test/Learners/LearnersTests`. 

Depending on the build configuration and targeted framework, the assemblies will be located in the `bin/Debug<Core|Full>/<netcoreapp2.1|net461>` or `bin/Release<Core|Full>/<netcoreapp2.1|net461>` subdirectories
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
dotnet ~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.1/xunit.console.dll bin/DebugCore/netcoreapp2.1/Microsoft.ML.Probabilistic.Tests.dll -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance

dotnet ~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.1/xunit.console.dll bin/DebugCore/netcoreapp2.1/Microsoft.ML.Probabilistic.Tests.dll -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTests -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -parallel none
```
To run the same set of tests on Mono:

```bash
mono ~/.nuget/packages/xunit.runner.console/2.3.1/tools/net452/xunit.console.exe bin/DebugFull/net461/Microsoft.ML.Probabilistic.Tests.dll -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance

mono ~/.nuget/packages/xunit.runner.console/2.3.1/tools/net452/xunit.console.exe bin/DebugFull/net461/Microsoft.ML.Probabilistic.Tests.dll -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTests -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -parallel none
```

Helper scripts `netcoretest.sh` and `monotest.sh` for running unit tests on .NET Core and Mono respectively are located in the `test` folder.

## Using MKL on x64 platform
`Infer.Net` supports optional [Single Dynamic Library](https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-linkage-and-distribution-quick-reference-guide)-linkage to Intel MKL. This model seems to have a known [issue](https://software.intel.com/en-us/articles/a-new-linking-model-single-dynamic-library-mkl_rt-since-intel-mkl-103). Steps below describe how to use Intel MKL with `Infer.Net`. 
1. Download [Intel MKL](https://software.intel.com/en-us/mkl/) which includes redistributables. Typically, this is installed in  "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win". We'll reference this folder as `MKL_DIR`.
1. Navigate to `MKL_DIR\mkl` and rename `mkl_rt.dll` to `mkl.dll`.
1. Add `MKL_DIR\mkl` and `MKL_DIR\compiler` to PATH environment variable.
1. In Runtime project settings > Build > Add a conditional compilation symbol `LAPACK`.
1. In Runtime project settings > Build > Check `Allow unsafe code`.
1. In TestApp project settings (or project of your choice), add all the paths for MKL redistributables as Reference paths.
1. Restart Visual Studio or Command Prompt.
1. Run TestApp (or project of your choice).
1. An alternative to having two paths defined is to copy all dlls into one folder and add that folder to PATH as well as reference it from the project.
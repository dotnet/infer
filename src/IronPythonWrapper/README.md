# Usage

Make sure folders containing `InferNetWrapper.py` and Infer.NET binaries are present among the python's search paths.

# Running Tests and Examples

* Run `SetupCompiler.cmd` script. It runs `dotnet publish` on `Compiler.csproj` and puts the results into `src/IronPythonWrapper/Compiler` folder. 
* Open `TestWrapper.sln` for tests or `InferNetExamples.sln` for examples.
* Run the only project in the opened solution.
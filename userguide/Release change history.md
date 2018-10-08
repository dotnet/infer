---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Release change history

#### 0.3 (5th October 2018)

*   Open source release on GitHub, with MIT license.
*   The Infer.NET Compiler and Runtime libraries target .NET Standard 2.0, making them useable on Linux (without installing Mono) and macOS.  On Windows, Infer.NET requires .NET framework version 4.6.1 or .NET Core 2.1, as explained at [.NET implementation support](https://docs.microsoft.com/en-us/dotnet/standard/net-standard).
*   The namespace prefix is now Microsoft.ML.Probabilistic, and many classes have changed their namespaces.
*   Graph visualization differs between platforms and is controlled by `InferenceEngine.Visualizer`.  By default, graphs are visualized using [GraphViz](http://www.graphviz.org/) in a web browser.  On Windows, users can select the `WindowsVisualizer` which visualizes graphs using [MSAGL](https://github.com/microsoft/automatic-graph-layout).
*   Added `InferenceEngine.Compiler.BrowserMode` for viewing the steps of the compiler.  See [Transform browser](Transform browser.md).
*   Added `InferenceEngine.ModelNamespace` for customizing the namespace of the generated code.  The default namespace of generated code is now `Models`.

#### 2.7 (19th March 2018)

*   Changed `Variable.InitialiseTo` to have a larger effect on the message schedule. This may change the runtime behavior of some models.
*   The runtime memory consumption of some models is greatly reduced.
*   Distributions are DataContract serializable (and therefore JSON.NET serializable).
*   Added `Variable.Split` and `SplitSubarray` for splitting arrays.
*   Added `Variable.InnerProduct` for arrays.
*   Added `Variable.ISparseList`.
*   Added `Variable.Min`.
*   Added `ListDistribution`.
*   Added `Variable.Matrix`.
*   Added `Variable.SetMarginalPrototype`.
*   Added `PositiveDefiniteMatrix.CholeskyInPlace`.
*   `Variable.VectorGaussianFromMeanAndVariance` supports unknown mean.
*   `Variable.MatrixTimesVector` accepts `Variable<double[,]>`.
*   `Variable.Max` supports Gamma distributions.
*   `Variable<Vector>` supports addition operator (+)
*   `Variable<int>` supports multiplication operator (*)
*   A Wishart matrix variable can be multiplied by a Gamma variable.
*   `Variable<double>` supports exponentiation operator (^) and equality (==)
*   Factor.IsGreaterThan can make comparisons between Gamma variables or Beta variables.
*   Added TruncatedGamma distribution and Variable.TruncatedGammaFromShapeAndRate factor.
*   Added factory methods to DiscreteChar; such as `Letter()`, `Digit()`, `Lower()`, `Upper()`.
*   Fixed a bug in `TruncatedGaussian.Sample`.
*   Improvements to Binomial distribution class.
*   Performance improvements to `StringFormat` operators.
*   Performance improvements to `StringDistribution.ToString()`.
*   The posterior distributions of the Matchbox recommender learner are publicly accessible.
*   Added `Util.ReadCsv`
*   Added `Matrix.Kronecker`, `Matrix.Commutation`
*   Added `Matrix.SetToLeastSquares` and `DenseVector.SetToLeastSquares`
*   Added `Matrix.SetToRightSingularVectors`
*   Added `Rand.SampleWithoutReplacement`
*   New engine.Compiler options: TraceAllMessages, InitialisationAffectsSchedule  
*   Added BackwardPass option to Sequential attribute.
*   Added PointEstimate attribute for Variables.
*   Improvements to MatlabReader and MatlabWriter.
*   User-defined factors can have 'out' parameters.
*   Distribution interfaces are now covariant when appropriate.
*   `Vector` and `Matrix` implement `IReadOnlyList`.  
*   `DistributionArrays` implement `IReadOnlyList`.
*   Changed uses of ApplicationException to InferException.

#### 2.6 (25th November 2014)

*   Infer.NET now requires .NET framework version 4.5. If your project uses an earlier version of .NET then you will get errors of the form "The type or namespace name 'MicrosoftResearch' could not be found".
*   Infer.NET now includes two pre-built learners: a classifier and recommender system. These can be invoked directly from the command-line or a .NET language, without having to learn the Infer.NET API.
*   Infer.NET Fun is no longer supported.
*   Chain and grid models defined using offset indexing are now given efficient message-passing schedules.
*   Added the Chess Analysis example.
*   Added support for random Wishart matrices in EP. The Mixture Of Gaussians tutorial now works with EP.
*   The TruncatedGaussian tutorial now works with VMP.
*   Factor graphs can now be viewed in Visual Studio using `InferenceEngine.SaveFactorGraphToFolder`.
*   Variable.If(value==x) is now allowed and is equivalent to Variable.If(x==value).
*   Variable.Copy(VariableArray) now returns a `VariableArray`.
*   Added `Variable.Double`.
*   Added `Variable.Replicate` and `Variable.InitialiseBackwardTo` for more precise control over initialisation.
*   Improved support for `Variable.IList` as an alternative to `Variable.Array`.
*   Improved support for product and ratio of Gamma-distributed variables.
*   Added `ModelCompiler.UseExistingSourceFiles` option for debugging.
*   Added `ModelCompiler.AllowDerivedParents` option.
*   Added `GetMode`, `FromDerivatives` method to various distributions.
*   Added `Discrete.SetToPadded`. Some methods on Discrete now allow mismatched dimensions (with implied padding).
*   Added `MMath.BesselI`, the incomplete Gamma function, and the incomplete Beta function.
*   Added `Vector.FindAll`. 
*   `Vector.Outer` now returns a Matrix instead of PositiveDefiniteMatrix.
*   Improved support for Poisson and Binomial distributions.
*   Improved handling of point-mass priors and initialisers in EP operators.
*   Improved handling of user-provided initialisers.
*   Improved precision of GaussianOp, IsBetween, GammaFromShapeAndRate, and other factors.
*   Added experimental EP support for string random variables and operations on strings such as concatenation, formatting, etc.
*   Added a new series of tutorials on inference with string random variables.

#### 2.5 Beta 2 (18th April 2013)

*   Added `engine.Compiler.TreatWarningsAsErrors`
*   `SetToRatio` now takes an additional argument
*   `Variable.Random(PointMass)` is equivalent to `Variable.Constant`, constants can be inferred, point masses are quoted correctly for various distributions
*   Variable.If(x==value) is equivalent to `Variable.Case(x,value)`. Variable.If(x==i) inside of ForEach(i) is equivalent to `Variable.Switch(x)`.
*   Compiler produces better message-passing schedules
*   null constants are allowed
*   Improved accuracy of GaussianFromMeanAndPrecision with unknown precision for EP
*   `InitialiseTo` can be applied to array elements, making it simpler to define initialisers for arrays, such as `x[item].InitialiseTo(xInitArray[item]);`
*   `SparseVectors` display in a more compact format, and display as dense when they are actually dense
*   `Wishart.Uniform` bugfix
*   Added `Wishart.FromMeanAndMeanLogDeterminant`, `Discrete.GetMedian`
*   Added `Variable.MatrixTimesScalar`, `Variable.WishartFromShapeAndRate`, `Variable.CountTrue`
*   Added `Matrix.Parse`
*   MatlabReader/Writer now support multidimensional arrays
*   Added `QueryTypes.MarginalDividedByPrior` to replace InferOutput. IGeneratedAlgorithm no longer implements GetOutputMessage (use Marginal with QueryType instead)
*   `Infer(x, QueryType)` requires x to have the QueryType as an attribute, for non-default QueryTypes
*   WetGrassSprinklerRain now works with VMP
*   Multi-class Bayes Point Machine has been re-architected to be more robust and efficient for large numbers of classes
*   `Matrix.SetToOuterTranspose` bugfix (affected all VectorGaussians with dimension >= 40)
*   Infer.NET Fun inference: new strongly-typed inference function (infer) directed by user-specified compound distribution type; new Model<A,B> type that avoids model recompilation for changes to input data
*   Infer.NET Fun syntax: additional operations for in-model symmetry breaking and providing range, dimension and sparsity information to Infer.NET when necessary

#### 2.5 Beta 1 (28th September 2012)

*   Infer.NET is now released as a zip folder
*   Infer.NET now requires .NET framework version 4.
*   Added `Variable.Repeat` blocks to the modelling API. The LDA example now uses them.
*   Infer.NET Fun, which allows you to write many models directly in F#, is now included as part of the Infer.NET release.
*   New examples: Recommender System, DifficultyAbility, and StudentSkills.
*   Some examples that were previously stand-alone solutions have been moved into the Examples Browser.
*   `InferenceEngine.ShowFactorGraph` shows constant values, among other improvements.
*   Removed `engine.BrowserMode`.
*   Added `engine.Compiler.UseSerialSchedules` option.
*   Added `engine.Compiler.AddComments` option.
*   Models undergo additional checks for validity. `x.SetTo(y)` now gives an error if y has been used in any other expressions.
*   The generated class is now marked 'partial'.
*   Distributions now implement GetLogAverageOfPower.
*   MatlabWriter supports `bool[]` and `int[]`.
*   Distributions can be serialized into XML via DataContractSerializer.
*   `Matrix.SetToEigenvectorsOfSymmetric` can compute eigenvectors and eigenvalues of a symmetric matrix.
*   GammaFromShapeAndRate supports a Gamma-distributed shape with VMP, and a Gamma-distributed rate with EP.
*   `Variable.Binomial` supports a Poisson-distributed trialCount. '+' supports Poisson-distributed integers.
*   `Variable.Constrain` methods are more efficient and consistent with each other. `Variable.ConstrainTrue(x)` is now equivalent to `Variable.ConstrainEqual(x,true)`, and similarly for `ConstrainFalse`, `ConstrainPositive`, and `ConstrainBetween`.
*   `Variable.Subarray` is more efficient in certain cases.
*   Removed a bug that sometimes caused the compiler to never complete.
*   Removed a bug that prevented use of `Variable.GetItem` on `Vectors`.

#### 2.4 Beta 3 (11th October 2011)

*   `Variable.Softmax()` now accepts sparse vectors and is more robust.
*   Speed and memory improvements to LDA example.
*   Improvements to the generated code, especially the message-update schedules.
*   Improved evidence calculations for TruncatedGaussians.
*   `Variable.Switch` is more efficient when dealing with a large number of cases.
*   `Variable.Logistic` is more robust with ExpectationPropagation.
*   GetItems and Subarray now support jagged arrays.
*   Infer.NET now checks that observed arrays are the correct length, at the point when the observation is set. This may break some existing user code where observed arrays are set before the array length is set.
*   `Variable.ForEach()` supports any number of ranges.
*   `Variable.Array()` has new overloads for making deep jagged VariableArrays.
*   Improved handling of deterministic factors in 'if' blocks (fewer spurious AllZeroExceptions are thrown).
*   `Dirichlet.Uniform(dimension, initialCount)` is now called `Dirichlet.Symmetric`.
*   Infer.NET introduction document.
*   Infer.NET 101 document which comes with a series of examples.
*   DivideMessages option for Shared variables.
*   Bayesian PCA and Discrete Bayesian Network examples.
*   Improved the Rand.Poisson sampler.

#### 2.4 Beta 2 (17th December 2010)

*   EP now uses message division by default, which speeds up many models. To get the old behavior for a particular variable, add the new DivideMessages attribute to the variable.
*   Setting `variable.ObservedValue` to any value will clear cached inference results, even if the variable was already observed to that value.
*   Improved efficiency of Gibbs sampling.
*   Gibbs sampling works better with nested if statements.
*   Improved numerical accuracy of various factors.
*   Improved support for TruncatedGaussian distributions.
*   Added `Rand.NormalBetween` for sampling from truncated Gaussians.
*   Added `WrappedGaussian` distribution and Rotate factor.
*   Inference is no longer allowed for local variables in a ForEach block (this has caused some minor changes to the Multi-class BPM classifiers).

#### 2.4 Beta 1 (30th October 2010)

*   Extensive improvements to the documentation + additional examples.
*   Introduction of quality bands for inference components (algorithms, operators, distributions) to make transparent which components are mature/preview/experimental. Quality auditing functions for giving errors or warnings if certain quality levels are not met by all used components.
*   New method of fine tuning what variables are inferred using the OptimiseForVariables property on InferenceEngine.
*   Preliminary (experimental) support for max product belief propagation in undirected models.
*   Added Silverlight version of the Infer.NET runtime to support using pre-compiled models in Silverlight 3.0 or above.
*   Multicore support now uses .NET framework 4.0 support for parallel tasks (to use set engine.Compiler.UseParallelForLoops = true)
*   Added experimental support for sparse Gaussian Processes  - see [Gaussian process classifier](Gaussian Process classifier.md) for an example.
*   Added the `DoNotInfer` attribute
*   Added `Compiling` and `Compiled` events to monitor model compilation
*   Ability to monitor the progress of inference using the `ProgressChanged` event on InferenceEngine.
*   Added `SumWhere` factor
*   The indexer of a loop can now be accessed using the Index property on a ForEachBlock. Also, loops can now be cloned.
*   Accuracy improvements to several factors, including: BernoulliFromLogOdds, DiscreteFromLogProbs, Logistic, Softmax, and Exp.
*   Improved efficiency of factors that manipulate Vectors (GetItem, Concat, etc.).
*   Many optimisations of the generated code, reducing computational cost and memory consumption. These are controlled by the new `OptimiseInferenceCode` switch on the model compiler.
*   The generated code now follows a different structure, documented in the user guide. Applications which call directly into generated code will need to be changed.
*   Reduced memory consumption in the compiler and the generated code. Added engine.Compiler.FreeMemory option.
*   Repeated inference on the same model with different observed data is more efficient. Added `Update()` method to warm-start the fixed-point iterations.
*   Added ability to control the operator search path, which allows selection between alternate implementations of an operator (e.g. with different speed/accuracy characteristics).
*   Deterministic variables (e.g. as a result of observations) can now be inferred.
*   `SparseVector` and `SparseList` classes.
*   Support for sparse messages to reduce memory consumption in models with integers variables over large domains.
*   Improvements to Gibbs sampling.
*   Several new examples including an LDA wrapper with various scalability options.

#### 2.3 Beta 4 (12th November 2009)

*   Added `SharedVariableArray2D`.
*   Variable.GammaFromShapeAndScale now supports random parameters.
*   Added support for multiplication of a Gaussian-distributed variable with a Gamma-distributed or Beta-distributed variable.
*   Added `Variable.Vector` for converting random arrays to random Vectors.
*   MatlabReader added.
*   Improved Gibbs sampling - more models and speed improvements.
*   Supports VisualStudio 2010 beta 2 and October 2009 CTP releases of F#, and fixes F# import error for MicrosoftResearch.Compiler.dll

#### 2.3 Beta 3 (4th September 2009)

*   Some bug fixes for 2.3 Beta 2.

#### 2.3 Beta 2 (27th August 2009)

*   BernoulliFromLogOdds and Logistic now support Expectation Propagation.
*   Added plus operator and comparison operators for integer variables.
*   Added `Concat`, `Subvector`, and `GetItem` factors for `Vector` variables.
*   DiscreteUniform now allows a random size.
*   Observed variables can now be inferred (the result is a point mass distribution on the observed value).
*   Changed the order of arguments to Binomial and Multinomial.
*   Beta distributions print out differently.

#### 2.3 Beta 1 (3rd August 2009)

*   Added `Variable.Logistic`, `Variable.Softmax`, `Variable.BernoulliFromLogOdds`, `Variable.DiscreteFromLogProbs`, `Variable.Binomial`, `Variable.Multinomial`, `Variable.AllTrue` factors.
*   EP evidence is now computed in a different way, which is more numerically stable. This is relevant when implementing new factors or calling operator methods directly. Specifically, the definition of LogEvidenceRatio (the method for computing EP evidence) has changed. However the overall evidence value is the same as before.
*   Jagged arrays can now be initialized using `InitialiseTo`.
*   Model is now recompiled when trying to infer a variable not included in an earlier InferAll. 
*   Improved handling of nested Switch blocks.
*   Improved handling of If and Case blocks with non-random conditions.
*   Better support for SharedVariableArrays and arrays defined by SetTo.
*   Can now index 2D arrays by observed variables (both indices must be observed).
*   Added `Variable.GammaFromShapeAndRate`.
*   Reduced memory allocation in the generated code. When possible, messages are now allocated once in the `Reset()` method and reused across calls to Infer.
*   Added `InferenceEngine.ReturnCopies` flag.
*   Removed a confusing overload of InferenceEngine.Infer<>
*   Improved accuracy of `PositiveDefiniteMatrix.SetToInverse`. Added `LowerTriangularMatrix.SetToInverse`, `Matrix.SetToOuter(Matrix)`, `Matrix.SetToOuterTranspose(Matrix)`.
*   Experimental multicore support using Parallel Extensions library (to use set engine.Compiler.UseParallelForLoops = true)
*   Support for enum types with `Variable.EnumDiscrete()`
*   Some efficiency improvements for If and Switch blocks
*   Support for returning arrays of distributions from Infer e.g. `Infer<Bernoulli[]>()`
*   Gibbs sampling. Some factors not yet supported such as gates and array factors.
*   F# wrapper - hides some of API complexity, providing distribution and domain-specific Variable and Distribution arrays types.
*   IronPython wrapper - hides some of API complexity for IronPython users
*   Runs on Linux with Mono

#### 2.2 Beta 2 (7th January 2009)

*   Fixed bug in multiplication of a `Gamma` variable
*   Fixed problem with locales which use a comma to represent a decimal point
*   Tutorial example for mixture of Gaussians now matches documentation
*   Reduced memory consumption when transform browser mode is set to 'never'
*   Documented examples using F#, C++/CLI, and IronPython

#### **2.2 Beta 1 (5th December 2008)**

*   Some minor bug fixes

#### **Version 2.2.31202  (2nd December 2008)**

*   `Given`, `Constant` and `RandomVariable` classes are deprecated. Use `Variable` instead, and set the `ObservedValue` and `IsReadOnly` properties as described in the documentation.
*   Support for jagged arrays
*   Variables must now be defined in all branches of an If or Case block unless they are local to that block
*   If, Case, and Switch statements can now take non-random conditions
*   Generated code is now fully commented
*   DLL structure simplified to Microsoft.ML.Probabilistic.Compiler.dll and Microsoft.ML.Probabilistic.dll
*   Namespace changes:
    *   Algorithms moved to `Microsoft.ML.Probabilistic` namespace
    *   Many utility classes moved into `MicrosoftResearch.Core` namespace and sub-namespaces (e.g. `MicrosoftResearch.Core.Math`)
*   Many classes have been marked internal
*   DistributionArray etc are now for internal use only. They implement IDistribution, and the API provides methods to retrieve .NET arrays of distributions
*   Several methods for creating random int/Vector variables now optionally take a range to indicate the cardinality/dimensionality of the variable
*   Easier to use generated code in a standalone fashion
*   Distribution classes are now serializable
*   Shared variable improvements including support for SharedVariableArray
*   Support for extracting multiple elements from an array using indexing
*   Snapshot of the online documentation is now included in the installed product
*   Many bug fixes

#### **Internal version** 2.1.30904 (4th September 2008)

*   Removed dependence on Reflector.
*   Support for 3D random variable arrays.
*   `InitialiseTo()` can take a Given, so that initializers may be changed at runtime.
*   Added `Variable.Max` for taking the maximum of two random doubles.
*   Added `Variable.Copy`, `Variable.PointMass`, `Variable.Uniform` for various distributions.
*   Added operator overloads for +,-,*,/ random variables with constants.
*   Added ^ operator for VectorGaussian.
*   Added option `InferenceEngine.Compiler.WriteSourceFiles=false` to prevent writing source files (they are compiled in memory instead).
*   Improved accuracy of `SetToSum` methods.
*   Improved accuracy of message-passing across if/case/switch blocks.
*   Expectation Propagation now handles Beta/Dirichlet distributions in a more robust way, reducing the occurrence of ImproperMessageExceptions.
*   `SetToSum` now forces a proper distribution by default for Beta/Dirichlet. Setting the static field `AllowImproperSum=true` restores the old behavior.
*   The order that loop variables appear in array indices can now be different from the nesting order of the loops.
*   More fixes to SetTo in if/case/switch blocks.
*   Fixed handling of AreEqual factor in VMP.
*   Fixed evidence computation for various factors.
*   Fixed bugs in scheduling.
*   Fixed bug in `Rand.Perm`.
*   Fixed handling of nested if/case blocks.

#### **Internal version** 2.1.30523 (23rd May 2008)

*   New API for SharedVariables.
*   Fixed handling of SetTo in if/case/switch blocks.
*   Fixed handling of And and Or factors in VMP.
*   Fixed handling of array variables defined inside gates (they were sometimes incorrectly inferred as constant).
*   `VectorGaussian` marginal prototype is now inferred for `Factor.VectorGaussian` with constant arguments.
*   `GivenArray.Value` and `ConstantArray.Value` are now `IList<T>` instead of `T[]`.
*   Reduced the occurrence of improper message exceptions during EP.
*   InitialiseTo only accepts distribution classes (not arrays as previously).
*   Range constructor no longer accepts a name (use the Named method instead).
*   Improved handling of boundary cases in `Factor.IsBetween`.
*   `Variable.ConstrainBetween` is more efficient (has a dedicated operator class).
*   Speed ups to `DistributionArray`
*   Changed default `GeneratedSource` folder to be below the current folder.
*   The name of generated classes can now be specified, using the inference engine `ModelName` property.
*   Removed debug messages, and instead made `InferenceEngine.ShowProgress` default to true (on).

#### **Internal version** 2.1.30320 (20th March 2008)

*   Proper attribute added to message function parameters which must be proper. Currently, this has the same behaviour as the "SkipIfUniform" attribute.
*   Improved handling of corner cases in SetToSum.
*   Shared variables work with VMP
*   Marginal prototypes are propagated automatically for `Factor.GetItem` and `Factor.GetItems`.
*   XML code documentation added to install for improved Intellisense
*   Some optimisations for message operators
*   Improved diagnostic messages during compile

#### **Internal version** 2.1.30310 (10th March 2008)

*   Increased accuracy in some EP messages
*   Improper message exceptions happen less often
*   Improved scheduling of VMP
*   Fixed indexing of constant/given arrays by random integers
*   Fixed click model example

#### **Internal version** 2.1.30305 (5th March 2008)

Highlights:

*   Expectation propagation (EP)
*   Variational Message Passing (VMP)
*   Belief propagation (sum-product algorithm) as a special case of EP.
*   Exponential family distributions: Gaussian, Gamma, Beta, etc.
*   Factors for arithmetic and boolean operations
*   Modelling API
*   Model compilation for efficient execution
*   Consistent operator syntax for message computation
*   Plates
*   Gates
*   Evidence computation
*   Many application samples and tutorial examples  
    

​

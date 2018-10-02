---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Running inference](Running inference.md)

## Inference engine settings

High-level inference settings are all accessed via properties or methods of an **InferenceEngine** object (in the **Microsoft.ML.Probabilistic** namespace). The settings take effect at the point of model compilation; typically this occurs when you request a marginal value from one of your variables, at which point both model compilation and inference occur. However, you can separate out compilation and inference as discussed in [Controlling how inference is performed](Controlling how inference is performed.md).

You can get an instance of the **InferenceEngine** class via the following snippet:

```csharp
InferenceEngine engine = new InferenceEngine();
```

#### Algorithm

This setting specifies the inference engine that will be used. [Working with different inference algorithms](Working with different inference algorithms.md) describes this option in more detail. The options currently are:

```csharp
// Use Expectation propagation  
engine.Algorithm = new ExpectationPropagation();
```

```csharp
// Use Variational Message Passing  
engine.Algorithm = new VariationalMessagePassing();
```

```csharp
// Use Gibbs sampling  
engine.Algorithm = new GibbsSampling();
```

The default setting for the algorithm is Expectation Propagation. You can also optionally specify the algorithm when you create the engine:

```csharp
InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
```

#### Compiler

This property holds the compiler that the inference engine uses to compile the model into efficient inference code. You cannot modify this property but you can modify properties of the returned ModelCompiler object as described in the table below. 

| _Property_ | _Description and example_ | _Default value_ |
|----------------------------------------------------------|
| **GeneratedSourceFolder** | The relative or absolute path of the folder where the C# inference source code should be generated. `engine.Compiler.GeneratedSourceFolder = "MyGeneratedSourceFolder";` | "GeneratedSource" |
| **WriteSourceFiles** | Specifies whether the generated source code should be written to disk. To prevent writing source files to disk, so that instead they are only generated in memory, use: `engine.Compiler.WriteSourceFiles = false;` | true |
| **GenerateInMemory** | Specifies whether the inference assembly should be generated in memory or on disk. Generating the assembly in memory can avoid security exceptions if you are running your inference from a network drive. If you want to debug into your generated inference code, you must set this to `false`: `engine.Compiler.GenerateInMemory = false;` | true |
| **IncludeDebugInformation** | Specifies whether the inference assembly should be compiled with debugging symbols. If you want to debug into your generated inference code, you must set this to `true`: `engine.Compiler.IncludeDebugInformation = true;` | false |
| **ReturnCopies** | If true, code will be generated to return copies of the internal marginal distributions. If this is not done, modifying returned marginals in place may affect the result of future inference calculations. Even when set to true, the copying is only done during inference; when Infer returns a cached result, this will be the same object previously returned. If you want to save memory by avoiding creating these copies, you can set this to `false`: `engine.Compiler.ReturnCopies = false;` | true |
| **FreeMemory** | If true, internal message arrays will be freed and re-allocated every time inference is run. This minimizes the memory overhead of inference. If false, internal message arrays will be kept alive between runs. To speed up inference when running the same model many times with different data, you can set this to false: `engine.Compiler.FreeMemory = false;` | true |
| **AddComments** | If true, the generated code will contain comments describing each statement. To reduce the size of the generated code, you can set this to `false`. | true |
| **OptimiseInferenceCode** | If true, various optimisations will be applied to the generated inference code which normally result in significant speed ups and reduction in memory usage. You should generally never set this to false - this option is provided to help with debugging of these new optimisation features. The only circumstance where you may try setting this to false is if the compiler fails in an optimisation transform. If you have an example which only works without optimisation, please report this to the Infer.NET team. | true |
| **IgnoreEqualObserved ValuesForValueTypes** | Let's you control what happens when you set an observed value which is equal to the old observed value for value types. If this property is true, setting an equal value will be ignored. If false, setting an equal value may cause inference to be re-run. | true |
| **IgnoreEqualObserved ValuesForReferenceTypes** | Let's you control what happens when you set an observed value which is equal to the old observed value for reference types. If this property is true, setting an equal value will be ignored. If false, setting an equal value may cause inference to be re-run. | false |
| **RecommendedQuality** | Recommended [quality band](Quality bands.md). The compiler will issue a warning for any component in your model which has a quality band below this level. | `QualityBand.Preview` |
| **RequiredQuality** | Required [quality band](Quality bands.md). The compiler will raise an error for any component in your model which has a quality band below this level. | `QualityBand.Experimental` |
| **GivePriorityTo, PriorityList** | The compiler uses a [search path](Modifying the operator search path.md) when looking for factor implementations. This method allows you to move certain implementations to the front of the search path, thereby changing properties of the inference algorithm. | The first factor implementation found when searching through loaded assemblies. |
| **UseParallelForLoops** | If true, instructs the compiler to use multiple cores by emitting parallel 'for' loops rather than normal 'for' loops. | false |
| **UseSerialSchedules** _(experimental)_ | If true, instructs the compiler to update variables in a sequential order (determined by Sequential attributes attached to ranges). See the [Difficulty versus ability](Difficulty versus ability.md) example. _This is an experimental feature and subject to change in future releases._ | true |
| **AllowSerialInitializers** _(experimental)_ | Controls the schedule when UseSerialSchedules=true. Set this to false to reduce memory in exchange for more computation. Useful for large datasets that would otherwise run out of memory. _This is an experimental feature and subject to change in future releases._​ | true |​

#### DefaultEngine

All the defaults specified on this page can modified by changing properties of DefaultEngine which is a static property of the **InferenceEngine class**. For example:

```csharp
InferenceEngine.DefaultEngine.Algorithm = new VariationalMessagePassing();  
InferenceEngine.DefaultEngine.ShowFactorGraph = true;
```

These settings will then become the default settings when a new **InferenceEngine** instance is created.

#### Group()

Allows you to specify that a set of variables are in a group. This is currently only used to manually specify the blocks in Block Gibbs Sampling. Note that groupings are automatically determined as necessary, so it is not typically necessary for you to call this method; but this does give you a manual override. A list of variables is passed to this method; the first argument is the root variable in the block which determines the order of message passing within it (first marginalisation within the block occurs towards the root, and then sampling is done away from from the root). Here is an example for a product of two Gaussians:

```csharp
Variable<double> a = Variable.GaussianFromMeanAndVariance(1, 2);  
Variable<double> b = Variable.GaussianFromMeanAndVariance(2, 3);  
Variable<double> c = a * b;  
GibbsSampling gs = new GibbsSampling();  
gs.BurnIn = 100; gs.Thin = 10;  
InferenceEngine engine = new InferenceEngine(gs);  
engine.NumberOfIterations = 10000;  
engine.Group(a, c);  
engine.Group(b, c);  
var marg = engine.Infer<Gaussian>(c);
```

#### ModelName

Allows a name to be specified for the model, which is used in the class name of the generated code. Defaults to "Model".

```csharp
// Change the model name  
engine.ModelName = "MyMixtureModel";
```

#### NumberOfIterations

This specifies how many iterations the algorithm will run for. An 'iteration' is a single update of all the variables in a model. For the EP and VMP algorithms. only a handful of iterations are sometimes required, and the inference algorithm may converge well before the default number of 50 iterations for the two algorithms.

```csharp
// Change the number of iterations  
engine.NumberOfIterations = 10;
```

Setting the number of iterations to -1 tells the inference engine to use the algorithm-specific default number of iterations. Note that Gibbs sampling will typically need many more iterations than EP or VMP.

This property can be used to do **incremental inference**. For example, suppose the NumberOfIterations is set to 10 and Infer() is called. Setting the NumberOfIterations to 20 and calling Infer() again will result in _a further 10 iterations being executed_. NumberOfIterations is effectively controlling the argument to **Execute()** on the underlying generated algorithm (see [Controlling how inference is performed](Controlling how inference is performed.md)).

#### OptimiseForVariables

This property lets you specify a list of variables which the inference engine should optimise inference for. If you set this property, the inference engine will internally generate code for performing inference on exactly this set of variables, avoiding the overhead of computing or caching marginals for any other variables. If you set this property and then call **Infer()** on a variable not in this list, you will get an error.

```csharp
// Optimise the engine to infer variables x, y and z. 
engine.OptimiseForVariables = new[] {x,y,z};
```

Setting this property to null reverts the engine to its normal behaviour where it will automatically choose which variables to perform inference on.

#### **ResetOnObservedValueChanged**

This option is useful for reducing the number of iterations required when observed values change. If true (the default), calling Infer() resets messages to their initial values if an observed value has changed. This ensures that the result of inference is always the same for the same set of observed values, but requires a full set of iterations to be performed each time. If false, the messages will not be reset when an observed value has changed. Thus the result of inference may depend on the previous observed values, but convergence can be faster. This option effectively switches between using **Update** instead of **Execute** on the underlying generated algorithm (see [Controlling how inference is performed](Controlling how inference is performed.md)).

```csharp
// Do not reset the message state  
engine.ResetOnObservedValueChanged = false;  

engine.NumberOfIterations = 10;  
// Perform 10 iterations from the initial state  
engine.Infer(x);  
y.ObservedValue = 4;  
// Perform 10 additional iterations (since y has changed)  
// but from the current state  
engine.Infer(x);  
engine.NumberOfIterations = 12;  
// Perform 2 additional iterations (since 10 already done  
// for the current y)  
engine.Infer(x);  
engine.NumberOfIterations = 5;  
// Perform 5 additional iterations (since 5 < 12)  
// from the current message state  
engine.Infer(x);
```

#### ShowFactorGraph

This specifies whether to display the factor graph corresponding to the model. A factor graph is a graphical representation of a model which shows each variable as a circle and each factor or constraint as a square. If a variable participates in the factor or the constraint, then an edge is shown between the corresponding circle and square. Factor graphs are more than just a picture of the model; they are intimately related to the message passing algorithms used for inference in Infer.NET.

```csharp
// Show the factor graph of the model  
engine.ShowFactorGraph = true;
```

The default setting for this bool property is false.

#### SaveFactorGraphToFolder

If not null, all factor graphs generated by calls to Infer will be saved to the given folder (which will be created if it doesn't already exist). The graph is saved in DGML format, which can be viewed and manipulated in Visual Studio. The file name is the model name, with extension ".dgml". If the folder name is an empty string (as opposed to null), the graphs will be saved to the current folder.

```csharp
// Save all factor graphs to the 'graphs' sub-folder  
engine.SaveFactorGraphToFolder = "graphs";
```

The default setting for this `string` property is `null`.

#### ShowMsl

If this flag is set to true, the [Model Specification Language (MSL)](The Model Specification Language.md) version of your model will be written to the Console

```csharp
// Show the model in the Model Specification Language  
engine.ShowMsl = true;
```

The default setting for this `bool` property is `false`.

#### ShowProgress

If this flag is set to true, progress information is written to the console, updating at each iteration.

```csharp
// Print out progress information  
engine.ShowProgress = true;
```

The default setting for this `bool` property is `true`.

#### ShowTimings

If this flag is set to true, timing information is written to the console at the conclusion of the inference, showing the time taken for compilation (if compilation is done) and for inference.

```csharp
// Print out timing information  
engine.ShowTimings = true;
```

The default setting for this `bool` property is `false`.

#### ShowWarnings

If this flag is set to true, any warnings encountered during inference will be printed to the console. The default setting for this `bool` property is `true`, so that you are aware of any problems encountered. To suppress warnings, set the property to `false`.

```csharp
// Do not print out warnings  
engine.ShowWarnings = false;
```

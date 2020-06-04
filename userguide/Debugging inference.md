---
layout: default 
--- 

[Infer.NET user guide](index.md)

## Debugging inference

When you run inference by asking for a marginal there are two steps that take place. First your model is 'compiled' into some efficient engine code. Then, this code is run to provide your updated values (see [how Infer.NET works](how Infer.NET works.md)). Either of these steps can fail for reasons which will now be discussed.

#### Model Compilation

The compilation step may fail for one of the following reasons:

1.  There are syntax errors in your code. In other words, your model code, whether using the modelling API or the Model Specification language, violates one of the rules of the language.
2.  The syntax is correct, but the model includes factors, constraints, and/or operators which are not supported, or not supportable.

The compilation step will throw an exception in either of these cases. Here is an example of a program which will not compile for the second reason. This is because the `GaussianFromMeanAndVariance` factor needs a positive variance parameter, so it does not make sense for the variance to itself have a Gaussian distribution. The error message will list a set of details such as "Gaussian is not of type double" and "Gaussian is not of type Gamma", indicating that the variance parameter needs to be either a raw double (i.e. not a Variable) or have a Gamma distribution.

```csharp
InferenceEngine engine = new InferenceEngine();  
Variable<double> V = Variable.GaussianFromMeanAndVariance(0, 1);  
Variable<double> X = Variable.GaussianFromMeanAndVariance(0, V);  
Gaussian marginalX = engine.Infer<Gaussian>(X); // fails  
Console.WriteLine("Posterior X = " + marginalX);
```

#### Debugging and profiling the inference algorithm

The simplest way to debug inference is to print out the messages being passed, using the [TraceMessages](Adding attributes to your model.md#tracemessages) attribute or [TraceAllMessages](inference engine settings.md#traceallmessages) compiler option.

Otherwise, the code produced by the model compiler is available for viewing as source code, and as such can be debugged and profiled like any other piece of source code. This code is saved as a C# source file (.cs) in a `GeneratedSource` folder. The name of the file is `[ModelName]_EP.cs` or `[ModelName]_VMP.cs` where, by default, _ModelName_ is set to `Model`.
You can set the **ModelName** property on the engine, prior to calling inference:

```csharp
InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());  
engine.ModelName = "GaussianExample";
```

In this example, the name of the generated source code would be `GaussianExample_VMP.cs`.
If more than one model is compiled in your program, then an integer is appended to _ModelName_ in order to make the name unique (starting at zero).

Before you can debug into the generated source code, you must set the following options on the model compiler:

```
engine.Compiler.GenerateInMemory = false;  
engine.Compiler.WriteSourceFiles = true;  
engine.Compiler.IncludeDebugInformation = true;
```

If you are getting an exception during inference, the above steps are sufficient to make the debugger halt at the offending line.  
If you want to step through the messages being passed during inference, open the generated source file and set a breakpoint at the start of **Execute** method. Then use the debugger as normal to step through the code.

To read the generated code, it will help to [attach names](Creating variables.md) to the variables in the model and see the page describing the [structure of generated inference code](Structure of generated inference code.md).  It may also help to [show the factor graph](inference engine settings.md#showfactorgraph) or the [MSL](inference engine settings.md#showmsl).  

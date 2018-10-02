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

The compilation step will throw an exception in either of these cases. Here is an example of a program which will not compile for the second reason. This is because the 'GaussianFromMeanAndVariance' factor needs a positive variance parameter, so it does not make sense for the variance to itself have a Gaussian distribution. The error message will list a set of details such as "Gaussian is not of type double" and "Gaussian is not of type Gamma", indicating that the variance parameter needs to be either a raw double (i.e. not a Variable) or have a Gamma distribution.

```csharp
InferenceEngine engine = new InferenceEngine();  
Variable<double> V = Variable.GaussianFromMeanAndVariance(0, 1);  
Variable<double> X = Variable.GaussianFromMeanAndVariance(0, V);  
Gaussian marginalX = engine.Infer<Gaussian>(X); // fails  
Console.WriteLine("Posterior X = " + marginalX);
```

#### Debugging and profiling the inference algorithm

If the model compilation succeeds, the code will be run when you request a marginal from the model. The code that is run is available for viewing as source code, and as such can be debugged and profiled like any other piece of source code. This code is saved as a C# source file (.cs) in a 'GeneratedSource' folder. The name of the file is _ModelName__EP.cs or _ModelName__VMP.cs where, by default, _ModelName_ is set to 'Model'. There are two exceptions for this. If more than one model is compiled in your program, then an integer is appended to 'Model' in order to make the name unique. Alternatively, you can set the **ModelName** property on the engine, prior to calling inference:

```csharp
InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());  
engine.ModelName = "GaussianExample";
```

In this example, the name of the generated source code would be GaussianExample_VMP.cs.

One common run-time error occurs when an algorithm diverges. For example, the example Gaussian model will often diverge for the EP algorithm unless additional initialisation is given.

Before you can debug into the generated source code, you must set the following options on the model compiler:

```
engine.Compiler.GenerateInMemory = false;  
engine.Compiler.WriteSourceFiles = true;  
engine.Compiler.IncludeDebugInformation = true;
```

**Hint:** to step through the messages being passed during inference, open the generated source file, set a breakpoint at the start of **Execute** method. Then use the debugger as normal to step through the code.

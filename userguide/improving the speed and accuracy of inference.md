---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Improving the speed and accuracy of inference

There are often multiple ways of using Infer.NET to solve a particular problem. The list below suggests ways in which you can change your Infer.NET code to improve its speed and/or the accuracy of the solution given.

*   _Use observed random variables rather than constants_  
    Avoid using constants for data and/or priors, and use observed variables instead - this means that your model can be run multiple times with different data and priors without recompiling the model. See the [truncated Gaussian tutorial](Truncated Gaussian tutorial.md) for an example of this. 

*   _Use variable arrays and ranges rather than .NET arrays._  
    Where possible, use variable arrays made with `Variable.Array()` instead of arrays of Variable instances (see [Learning a Gaussian tutorial](Learning a Gaussian tutorial.md)). 

*   _Use variable arrays rather than vectors._
    When you don't need to represent correlations in the posterior distribution, use variable arrays instead of vectors.

*   _Use Subarray instead of array indexing or GetItems._
    The model compiler can optimise certain uses of Subarray better than it can for GetItems.

*   _Use sparse messages when using large Discrete or Dirichlet distributions._  
    The messages needed to perform inference in models with large Discrete or Dirichlet distributions often have sparse structure. This can be exploited by [using sparse messages](using sparse messages.md) to significantly reduce memory consumption and increase speed. 

*   _Specify the exact set of variables to be inferred_

    Use the **OptimiseForVariables** property on the inference engine to specify exactly which variables you want to infer. This is more efficient than the **Infer()** method which always computes all variable marginals.

*   _Check for warnings from the model compiler_
    If the model compiler warns about excessive memory consumption, change the offending parts of the model.

*   _Reuse memory for the marginals_  
    Set **Compiler.ReturnCopies** to false on the inference engine. This causes the engine to modify the previous marginal distributions in place. However, be warned that you need to copy the marginal distributions yourself if you want to save them. 

*   _Reuse memory for internal messages_  
    Set **Compiler.FreeMemory** to false on the inference engine. This causes the engine to keep internal messages allocated between inference runs. However, be warned that this can significantly increase the memory consumption of your program.

*   _Reduce the number of factors_  
    In general, the fewer factors in your factor graph, the faster and more accurate inference can be. Try to rewrite your model to reduce the number of factors if possible. The **ShowFactorGraph** option on the inference engine should help.

*   _Profile the inference code_  
    You can profile the generated inference code to find bottlenecks, as described in [debugging inference](Debugging inference.md).

*   _Reduce the number of iterations of inference_  
    The default number of iterations is 50 - your model may need fewer iterations to converge. Modify **NumberOfIterations** in the [inference engine settings](inference engine settings.md) or use a custom convergence criterion (see [Controlling how inference is performed](Controlling how inference is performed.md)).

*   _Ensure that you are using optimised inference code_  
    Make sure that the compiler property OptimiseInferenceCode is set to true (it should normally be true and is so by default).

*   _Avoid the 64-bit JIT compiler_  
    If you compile your executable in Visual Studio with platform target = x64 or "Any CPU", then on a 64-bit OS you will be using the 64-bit JIT compiler at runtime, which is very inefficient on large methods such as the ones that Infer.NET generates. This causes a significant memory and computation spike on the first iteration of inference. To avoid this, change the platform target to x86.

*   _Avoid server-mode and concurrent garbage collection on a 64-bit machine_

    If you are running on a 64-bit machine, configure the garbage collector so that it does not run concurrently and so that it does not run in server mode. You can do this by adding the following lines to your application's app.config file:  
    
    ```xml
    <configuration>  
      <runtime>  
        <gcServer  enabled="false" />  
        <gcConcurrent enabled="false" />  
      </runtime>  
    </configuration>
    ```

*   _Use parallel for loops (experimental feature)_  
    You can get the model compiler to emit parallel for loops, allowing multiple cores to be used when running inference. Set the compiler **UseParallelForLoops** property to **true**.

​

---
layout: default 
--- 
 
[Infer.NET user guide](index.md) : [Running inference](Running inference.md)

## Monitoring the progress of inference

When you call **Infer()** on an inference engine and the answer has not already been computed, the engine will compile the model and then execute the inference algorithm. Depending on the model and the algorithm, this can take some time. It is often useful to be able to monitor the progress of the algorithm, for example, to show a progress bar to the user indicating how long the inference will take.

To allow you to monitor progress, the inference engine has a **ProgressChanged** event. This event is triggered whenever the progress of inference changes - normally once per iteration.

You can listen to this event using the following line of code:

```csharp
// Add a listener to the inference engine  
engine.ProgressChanged += new  InferenceProgressEventHandler(engine_ProgressChanged);
```

In this case, the **engine_ProgressChanged** method will be called each time the progress changes. It has the following signature:

```csharp
// Method to respond to inference progress events  
void engine_ProgressChanged(InferenceEngine engine, InferenceProgressEventArgs progress)
```

The **InferenceEngine** object is the engine that **Infer()** was called on. The **InferenceProgressEventArgs** object contains information about the current state of the inference algorithm. It has these properties:

*   **Iteration** \- the number of the iteration that just completed (starting at 0).
*   **Algorithm** \- the IGeneratedAlgorithm object which is being used to execute the inference algorithm. This can be queried for the current state of the algorithm e.g. to find the current value of marginals etc.

Here is a simple example of a method which prints the progress of inference to the console.

```csharp
// Method to respond to inference progress events  
void engine_ProgressChanged(InferenceEngine engine, InferenceProgressEventArgs progress)  
{  
  Console.WriteLine("Iteration {0} of {1}", progress.Iteration+1,  
                     engine.NumberOfIterations);  
}
```

Note that you can get simple feedback to the console without using this event by setting **engine.ShowProgress** to **true**. However, by using this event you can provide feedback however you like. If you choose to provide customised feedback to the console (like in this example), then you will want to switch off the default feedback by setting **engine.ShowProgress** to **false**.

_Note: In a future release, we plan to extend this event so that is can be used to create custom convergence criteria. In this case, the signature of the event handler may change slightly._

The **[ModelCompiler](../apiguide/api/Microsoft.ML.Probabilistic.ModelCompiler.html)** class provides a pair of events for monitoring compilation. They are called **Compiling** (for when compilation starts) and **Compiled** (for when it finishes). See the code documentation for more details.

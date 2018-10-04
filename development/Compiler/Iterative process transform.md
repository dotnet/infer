---
layout: default
---
[Infer.NET development](../index.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Iterative process transform

This transform takes the output of the scheduling process and converts it into a class that implements the `IExecutableProcess` interface. Roughly, each field in this class corresponds to a message, and there are methods to update messages and to return marginals. The input to the transform is a single method, whose parameters represent observed values and whose statements perform inference. If the input has multiple methods, the entire transform is repeated for each one. Fixed-point iterations are represented by `while(true)` loops. The goal of the output class is to interface to the user and manage the process of executing the schedule, with special attention to the case where observed values change and inference needs to be re-executed. The transform performs the following actions:

*  Create methods that execute pieces of the schedule.
*  Create a master method `Execute` that runs the entire schedule by calling the method for each piece.
*  Convert local variables into fields of the class.
*  Create properties for observed values. When an observed value is changed, these set the appropriate change flags so that parts of the schedule will be re-executed.
*  Create a `Reset` method that marks everything to be re-executed.
*  Create a `ResumeLastRun` property that controls how loops are initialized when observed values change.
*  Create methods to return marginals and output messages. These simply return the contents of the corresponding field. Specifically,
*  `SomeVariableMarginal()` returns the marginal distribution of `SomeVariable`. It is created for every variable used as an argument to `InferNet.Infer`.
*  `SomeMessageOutput()` returns the arbitrary message `SomeMessage`. Normally this is the likelihood message for an output variable (i.e. the marginal divided by the prior). It is created for every message whose `DependencyInformation` has IsOutput=true.

The division of the schedule into methods is dictated by how each statement depends on the observed values, i.e. the parameters of the method provided as input. The transform works out the subset of observed values that each statement depends on. A method is created for each distinct subset, and all statements whose dependencies equal exactly that subset are placed into the method. The idea is that when an observed value is changed, only the pieces of schedule that depend on that observed value are re-executed. To manage this, each method has a flag which indicates if it needs to re-execute. When an observed value changes, it sets the flags for all methods that depend on it.
 
When `Execute` is called, it blindly invokes all of the schedule methods. Each method checks if it needs to re-execute, and if not returns immediately.
 
The number of iterations is an argument of `Execute`. This number is passed down into any piece of schedule that has a `while(true)` loop. The `while(true)` is converted into a `for` loop from 1 to the number of iterations. If the schedule has multiple `while` loops, they will each execute for the given number of iterations. Note that if the number of iterations is changed (whether increased or decreased) then these loops will be re-initialized and re-executed. The meaning of Execute is "run inference for exactly this many iterations." It does _not_ mean "run this many _additional_ iterations."
 
Sometimes when observed values change, we don't want to re-initialized loops from scratch, but rather continue from the existing message state. For example, if the observed value has changed only slightly, we may hope to save time by starting from the previous state. The `ResumeLastRun` property enables this. To implement this, we need to know exactly which statements provide the initialization of a given loop, so that we can enable/disable them as appropriate. These are found in the `InitializerSet` attribute of the `while(true)` loop, attached by the [Dead code transform](Dead code transform.md). These statements are tagged with the parameter dependencies of the `while` loop, to indicate that even though these statements do not depend on those parameters directly, they may need to be re-executed when those parameters change (in order to initialize the `while` loop).
---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Controlling how inference is performed](Controlling how inference is performed.md) 

## Structure of generated inference code

Infer.NET generates a C# class for performing inference on your model. By default, the source of this class is placed at: . There are circumstances where it is useful to use this generated code directly, for example, to use a [precompiled inference algorithm](Using a precompiled inference algorithm.md) or to manually modify the message-passing schedule or some other aspect of the code. To help with this, we will now describe the structure of the generated class. Generated source code files also contain full documentation for the class, fields and methods so you can also refer to these directly.

**(Debug/Release)\\\bin\\\GeneratedSource\\\\[ModelName\].cs**. There are circumstances where it is useful to use this generated code directly, for example, to use a [precompiled inference algorithm](Using a precompiled inference algorithm.md) or to manually modify the message-passing schedule or some other aspect of the code. To help with this, we will now describe the structure of the generated class. Generated source code files also contain full documentation for the class, fields and methods so you can also refer to these directly.

#### The Generated Class

The class is an ordinary C# class of name **\[ModelName\]_\[Algorithm\]**, for example **Model0_EP**. It typically contains a number of public fields, properties, and methods. A property is created to hold each observed variable in the model. Fields are created for:

*   all requested marginals
*   all forward and backwards messages needed to compute requested marginals
*   invalidation flags to track which parts of the schedule need to be recomputed when observed values change

Methods are created for:

*   Setting observed values and initialising messages

*   Performing inference by updating messages

*   Retrieving posterior marginals and output messages


Each field and method has a documentation comment describing its purpose.

#### Generated Methods for Performing Inference

The methods in the table below are used when performing inference. The **Reset**, **Execute** and **Update** methods can all be called through the **IGeneratedAlgorithm** interface.

| _Method_ | _Purpose_ |
|----------------------|
| **Reset()** | **Resets all messages to their initial values.** |
| **Execute()** | **Execute the inference algorithm for the specified number of iterations, starting from the initial state.** |
| **Update()** | **Performs additional iterations of inference from the current state.** |
| **XXXMarginal()** | **Returns the current marginal for the variable 'XXX'.** This method can be called at any time to return the current estimate of the marginal distribution, as given by the current state of the inference algorithm. |
| **XXXMarginalDividedByPrior()** | **Returns the current output message for the variable 'XXX'.** This method is similar to XXXMarginal() methods but returns the 'output message' which is the current marginal for the variable divided by the prior. This message is very useful when models are being combined in a modular fashion and is the mechanism used when [sharing variables between models](Sharing variables between models.md). | 

These methods are only generated for variables which are marked with a [QueryType attribute](Adding attributes to your model.md).

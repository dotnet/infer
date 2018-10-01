---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Modifying the operator search path

The operator search path allows you to switch between different implementations of the message operators for a factor. In some cases, Infer.NET includes different implementations of a factor, allowing you to control speed versus accuracy. The search path also allows you to override a built-in message operator with your own implementation.

For example, consider the `SumOp` class from the page on [How to add a new factor and message operators](How to add a new factor and message operators.md). On that page,  `SumOp` was registered against a custom factor called `MyFactors.Sum`. Suppose you instead wanted to register `SumOp` against the built-in factor `Factor.Sum`. To start, you would change the `FactorMethod` attribute to read:

```csharp
[FactorMethod(typeof(Factor), "Sum", typeof(double[]))]  
public  static  class  SumOp  
{  
  ... 
}
```

At this point, you will get `AmbiguousMatchExceptions` when you try to compile a model, because Infer.NET will find two implementations of the same message. To tell Infer.NET to use your `SumOp` instead of the built-in class, you add it to the operator search path. This is done by calling `engine.Compiler.GivePriorityTo()` on an `InferenceEngine`, like so:

```csharp
InferenceEngine engine = new InferenceEngine();  
engine.Compiler.GivePriorityTo(typeof(SumOp));
```

The next time you perform inference with that engine, the model will be recompiled to use your `SumOp`. Besides a type object, you can also pass in a namespace string, an Assembly, or a Module. `GivePriorityTo` adds the given type/namespace/Assembly/Module to the front of the search path. To remove `SumOp` from the search path and go back to the default operator, call `RemovePriority`:

```csharp
engine.Compiler.RemovePriority(typeof(SumOp));
```

You can inspect the search path at any time via `engine.Compiler.PriorityList`.

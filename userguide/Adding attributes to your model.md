---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Adding attributes to your model

Infer.NET allows models to be annotated with attributes that provide information to the inference engine. Such attributes can be used to affect how inference is performed in particular parts of the model.

 Attributes can be added to a variable using the following syntax:

```csharp
x.AddAttribute(new MyAttribute());
```

where **x** is of type Variable<T>. It is often convenient to add attributes to a variable when it is created, using the following inline syntax:

```csharp
Variable<bool> b = Variable.Bernoulli(0.5).Attrib(new MyAttribute());
```

Any object can be attached as an attribute to your model. The following attribute types are currently recognised by the inference engine:

| **Attribute class** | **Description** | **Example** |
|-----------------------------------------------------|
| _MarginalPrototype_ | In certain circumstances, the inference engine is unable to determine what type of distribution to use as the marginal for a particular variable. In this case, you will receive an error asking you to specify the 'marginal prototype' for that variable. A marginal prototype is an instance of a distribution class e.g. new Gamma() which indicates the form of the desired marginal. The parameters of the supplied distribution are not used, just the class and dimensionality. To specify a marginal prototype for a variable, add a _MarginalPrototype_ attribute. | `x.AddAttribute(new MarginalPrototype(new Gamma()));` |
| _TraceMessages_ | Causes all messages related to this variable to be passed to `System.Diagnostics.Trace.WriteLine` during inference.  | `x.AddAttribute(new TraceMessages());` <br/> `Trace.Listeners.Add(new TextWriterTraceListener(Console.Out));` <br/> `Trace.AutoFlush = true;` |
| _DoNotInfer_ | Tells the inference engine that this variable will not be inferred, so that more efficient code can be generated. Alternatively, you can use the OptimiseForVariables option to achieve the same effect. |  |
| _DivideMessages_ | Says whether outgoing messages from a variable should be calculated by division. This is especially relevant to the Expectation Propagation algorithm where division is much more efficient. However division can be less accurate numerically, and this attribute gives the option to use multiplication instead. | `x.AddAttribute(new DivideMessages(false));` |
| _QueryType_ | This attribute tells the model compiler in advance what query types you will perform on this variable. The `QueryTypes` class contains the standard query types. By default, the only QueryType allowed is Marginal. If you do attach this attribute, then Marginal queries will no longer be available on this variable unless it is one of the attached QueryTypes. | `x.AddAttribute(QueryTypes.Samples);` <br /> `x.AddAttribute(QueryTypes.Conditionals);` |

Some attributes are set by a method on the variable, and these are shown in the following table. Although these attributes can be added in the standard way as in the previously table, there is some additional processing that needs to be done when setting these attributes, and this is handled in the method:

| **Method** | **Description** | **Example** |
|--------------------------------------------|
| `void SetSparsity(Sparsity sparsity)` | Sets the [sparsity](Variable types and their distributions.md) for Vector random variables. Typically this is only meaningful for models involving Dirichlet factors, and you do not normally have to call this method as the sparsity is inferred from the sparsity of the vector parameter. | `x.SetSparsity(Sparsity.Sparse);` |
| `void SetValueRange(Range valueRange)` | Specifies the range of values taken by an integer variable, or the dimension of a Vector variable. This method can be used to explicitly specify the value range for a variable in cases where it cannot be deduced by the model compiler. Many of the factory methods that create integer or Vector variables can take a range argument and will apply this attribute automatically. **You should only use this method if none of these factory methods is suitable, and you get an error message about a missing ValueRange.** | `Range k = new Range(mixtureSize);` <br /> `Variable<Vector> weights = Variable.New<Vector>();` <br /> `weights.SetValueRange(new ValueRange(k)));` |

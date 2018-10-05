---
layout: default 
--- 
[Infer.NET development](index.md)

## Polymorphic random variables

If we have a polymorphic method call e.g.

```csharp
object obj = SomeFunc();
var s = obj.ToString();
```

The actual ToString() method executed will depend on the runtime type of 'obj'.  This is achieved in C# using a [virtual method table](http://en.wikipedia.org/wiki/Virtual_method_table).  

In Infer.NET, we might have factors corresponding to ToString() for different types.  However, we have no mechanism at inference time that corresponds to the use of a virtual method table for dynamic dispatch.

It would be useful to develop such a mechanism.  It might be possible to use the C# mechanism or we may have to build our own.  

### Wrapper approach

One way to approach this problem is with wrappers.
The model compiler would generate new distribution classes that group an existing distribution with its operator implementations.  Each factor that wanted to use runtime dispatch would have a corresponding interface.  The generated distribution classes would implement these interfaces.  For example, the factor 'ToString' would have an interface 'HasToStringOperator' that provides delegates (as properties) for the ToString messages.  When the model compiler sees ToString in the model, it would generate a distribution class that implements HasToStringOperator.  The implementation of ToString has several parts.  There is one generic operator implementation that does runtime dispatch and multiple specific implementations for specific distribution types such as DateDistribution.  The generic operator for ToString would take any distribution that implements HasToStringOperator, fetch the delegate from the interface, and call the delegate.  Note that the operator implementation would not be aware of the generated distribution class, so the compiler would have to insert appropriate wrapping/unwrapping code into the delegates, essentially providing glue between the generic operator (which only knows the type HasToStringOperator) and the specific operator (which only knows the type DateDistribution).


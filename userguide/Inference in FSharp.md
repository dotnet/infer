---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md)

## Inference in F\#

### The Variable.Infer method

The syntax for the Variable.Infer method can be quite confusing, especially for array variables, as the Infer method has both a generic and a non-generic form. Which one you use depends on what you want to do with the result.

Suppose we have a `Variable<float> x` which we want to infer. Let's look at two alternative calls into the Infer method

```fsharp
let xPost1 = ie.Infer<Gaussian>(x)  
let xPost2 = ie.Infer< >(x)
```

The first line returns a Gaussian instance which is good if you want to call methods on the posterior distribution, but you do have to know what the distribution of the  `Variable<float>` is. The second line does not require you to know what the distribution is, as it returns the distribution as an object; however, this limits what you can do with the distribution without casting it further.

Now let's look at arrays. By default, the Infer function will return a Distribution array of some sort; in general you will not know the concrete class, but you can type it in a number of ways. Suppose we have a `VariableArray<float> x` which we want to infer. The following lines both return the same object:

```
let xPost1 = ie.Infer<IDistribution<float[]>>(x)  
let xPost2 = ie.Infer< >(x)
```

Internally, Infer.NET defines various types of Distribution array, but these all implement the generic interface `IDistribution<T>` where T is the type of the array domain (`float[]` in this case). However, it can be useful to have the concrete type returned so you have access to all the methods on the type. These types can be quite complex to specify, especially when dealing with jagged arrays of distributions, and are therefore defined in the `FSharpWrapper` under the respective modules: `FloatDistribution`, `IntDistribution`, `BoolDistribution`. As before, the second line returns an object, and the distribution types can also be used to cast this object if the distribution type is known. The equivalent lines are:

```
let xPost1 = ie.Infer<FloatDistribution.GaussianArray>>(x)  
let xPost2 = ie.Infer< >(x) :?> FloatDistribution.GaussianArray
```

### Inferring .NET arrays of distributions

Often you may want a .NET array of distributions returned rather than a `DistributionArray` instance. You can do this by providing the correct distribution array type as a type parameter to the Infer method. Alternatively you can call the appropriate `FSharpWrapper` method in the Inference module . For example `Inference.InferGaussianArray` has type `InferenceEngine*VariableArray<float> -> Gaussian[]`. So the following two lines are equivalent:

```
let xPost3 = ie.Infer<Gaussian[]>(x)  
let xPost4 = Inference.InferGaussianArray(ie, x)
```

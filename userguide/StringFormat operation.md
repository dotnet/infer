---
layout: default 
--- 

[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Strings Tutorial 2: StringFormat Operation

This page describes an experimental feature that is likely to change in future releases.

In this tutorial we'll take a look at a powerful string operation supported in Infer.NET, **StringFormat**, and the sorts of models one can define with it.

You can run the code in this tutorial either using the [Examples Browser](The examples browser.md) or by opening the Tutorials solution in Visual Studio and executing **StringFormat.cs**.

### Inferring an argument

The **StringFormat** operation supported in Infer.NET is very similar to the **String.Format** method in .NET in both signature and semantics. There are a few subtle differences, which we'll discuss in the next section. One way of using **StringFormat** in a probabilistic model is to define a piece of text to be the result of a **StringFormat** call, and then try to work out what the arguments were given the text. We will demonstrate that by inferring the name of a person from a greeting text. Let's for now assume that we're going to fix the form of the greeting. It is also natural to assume that the name of a person starts with a capital letter, followed by a number of lowercase letters. That allows us to define the following (rather simple) model of some greeting text:

```csharp
Variable<string> name = Variable.StringCapitalized().Named("name");  
Variable<string> text = Variable.StringFormat("My name is {0}.", name).Named("text");
```

The **Variable.StringCapitalized** method creates a string random variable from a uniform distribution over all strings that start with an uppercase letter, followed by one or more lowercase letters. As it was the case with **Variable.StringUniform**, this is an improper distribution. Now, we can run inference on a piece of text and infer the name:

```csharp
text.ObservedValue = "My name is John.";  

Console.WriteLine("name is '{0}'", engine.Infer(name));
```

This code will output

```
name is 'John'
```

### Inferring the template

So far we've fixed the form of the greeting. We can instead try to learn it by simultaneously working out what the template and the name are from the text. That will require us to specify a prior distribution over the template. It would be reasonable to say that the template should have a name placeholder somewhere, surrounded by non-word characters like a space, a full stop, or a comma. That leads us to the following model:

```csharp
Variable<string> name = Variable.StringCapitalized().Named("name");  
Variable<string> template =  
    (Variable.StringUniform() + Variable.CharNonWord() + "{0}" + Variable.CharNonWord() + Variable.StringUniform()).Named("template");  
Variable<string> text = Variable.StringFormat(template, name).Named("text");
```

**Variable.CharNonWord** creates a character random variable from a uniform distribution over all characters that cannot be a part of a word. As in C#, character variables can be concatenated with strings to produce other strings. If we now observe the text to be "Hello, mate! I'm Dave." and run inference, Infer.NET will work out that:

```
name is 'Dave'  
template is 'Hello, mate! I'm {0}.'
```

"Hello" is not considered a possible name because "H" is the first character in the text and thus there is no non-word character before it. "I'm" also cannot be a name under our model because **Variable.StringCapitalized** only allows for letters. But in order to further understand the inference results, it's now worth stating how the Infer.NET **StringFormat** is different from the .NET **String.Format**. In the Infer.NET **StringFormat**

*   Placeholders ({0}, {1} etc.) for each of the arguments must be present in the format string exactly once.
*   No braces are allowed in the format string except of those used to specify placeholders. So, it's not possible to use '{' to specify a single opening brace.
*   Only string arguments are supported.
*   Placeholder specifications can contain an argument index only. Specifying the minimum length of the argument string representation or a per-argument format string is not supported.

The first bullet is quite relevant to our example. If **StringFormat** allowed for the placeholder not to be present in the format string at all, it would have led to the possibility that the template is "Hello, mate! I'm Dave." and the text provides no information about the name at all, which is undesirable. Requiring each placeholder to be present in the format string exactly once helps to reduce the ambiguity of backward reasoning, and in our case allows us to unambiguously determine the template. Nevertheless, versions of **StringFormat** that relax this limitation may be added to Infer.NET in future releases.

Even with the placeholder presence restriction in place, the prior we've used for the template wouldn't always lead to unambiguous inference. For instance, if the observed text is "Hi! My name is John.", the results of the inference will be

```
name is 'My|John'  
template is 'Hi!( {0} name is John.| My name is {0}.)'
```

That is because "My" starts with a capital letter and is surrounded by non-word characters, just as for "John". In order to handle this ambiguity, we can either improve the prior over the template, or provide more data:

```csharp
Variable<string> name2 = Variable.StringCapitalized().Named("name2");  
Variable<string> text2 = Variable.StringFormat(template, name2).Named("text2");  

text2.ObservedValue = "Hi! My name is Tom.";  

Console.WriteLine("name is '{0}'", engine.Infer(name));  
Console.WriteLine("name2 is '{0}'", engine.Infer(name2));  
Console.WriteLine("template is '{0}'", engine.Infer(template));
```

Now Infer.NET can unambiguously work everything out:

```
name is 'John'  
name2 is 'Tom'  
template is 'Hi! My name is {0}.'
```

### Using the learned template

It is possible to combine forward and backward reasoning within the same model. For instance, we may want to generate a greeting text for a yet another name without explicitly providing the template, but inferring it from the data instead. To achieve this goal, all we have to do is define another text variable:

```csharp
Variable<string> text3 = Variable.StringFormat(template, "Boris").Named("text3");  

Console.WriteLine("text3 is '{0}'", engine.Infer(text3));
```

which gives

```
text3 is 'Hi! My name is Boris.'
```

In this tutorial we saw how to use the **StringFormat** operation to both extract values from a piece of text and learn templates from one or more pieces of text. These are both very useful techniques for processing natural text strings. We can now move to the [next tutorial](Motif Finder.md), where we'll see how to combine integer, boolean, character array and string random variables to define a complex probabilistic model from bioinformatics.

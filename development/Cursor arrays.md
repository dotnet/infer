---
layout: default 
--- 
[Infer.NET development](index.md)

## Cursor arrays

A cursor array is a data structure for storing a large number of structured objects in a small amount of space. It exploits the fact that distribution objects typically share the same structure, i.e. the same dimensionality and the same object graph. Rather than store this information redundantly in each element of an array, the array only stores the raw data which differs from message to message. A cursor object points into the array and makes the array element appear to be a fully structured message object. The user can call methods on the cursor just as if it were a message. Methods which modify the message, such as `SetTo`, write directly into the array.

Cursor arrays are implemented using indexers so that they can be used like a normal array. For example, you can say `a[i].SetTo(msg)` or even `a[i] = msg`. There are some subtleties, however. Reading `a[i]` does not create a clone of the cursor, it only mutates it, so the result is volatile. An expression like `f(a[i],a[j])` will not work; you need to explicitly clone `a[i]` before accessing `a[j]` (but only clone the cursor, not the data). It is possible to write cryptic code in which an array is accessed only for the side-effect on the cursor, as in this example: 

```csharp
Vector cursor = a[0];
foreach(Vector x in a) {
  cursor.SetTo(y);
}
```

This code sets every element of the array to y, because the foreach loop implicitly modifies the cursor. 

#### Value types as an alternative

An alternative way to save storage is with constructed value types. Using the code generation capabilities of .NET, the model compiler can create a value type at runtime which embeds all of the common structure of the messages. The value type only stores the data which differs from message to message. These can then be stored in an ordinary array. Each array element supports all methods of the original message type. 
Compared to cursor arrays, accessing a value array is more intuitive. Furthermore, the memory savings are potentially larger than with cursor arrays, because the same constructed type can be recycled throughout the network, whereas each cursor array must have a separate cursor.

One potential downside is that, because all messages are now value types instead of reference types, special coding techniques are required to deal with large messages (e.g. a 100-dimensional Gaussian message). A value type is copied whenever it is boxed, so boxing must be avoided. Casting to an interface causes boxing, so interfaces are less usable. Furthermore, any method that modifies a message must take it as a reference parameter. Unfortunately, reference parameters are quite limited in C#. For example, properties cannot be reference parameters. Except for array elements, collection elements cannot be reference parameters. You cannot capture a reference parameter, e.g. in a delegate or data structure, for the purposes of updating it later. 

Another potential downside is that it limits your choices of storage format. Each message must occupy a contiguous block of memory. With cursors, you could split the data for a message across multiple arrays. 

The essential problem here is that you lose the storage abstraction which you get from reference types, and consequently code with value types will inevitably be more complicated than under the cursor approach. To simplify the coding we could provide a cursor class that acts as a pointer to the value type. The cursor class can be used as a temporary wrapper to avoid boxing and allow mutation of the value within a method. It could also be used in collections, so that the collection elements are reference types. With this approach, the main difference from cursor arrays is that the underlying message arrays are ordinary system arrays, and you must explicitly create a cursor whenever you want to use one, instead of it being the default.
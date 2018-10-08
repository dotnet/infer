---
layout: default
---
[Infer.NET development](index.md)

## Performance lessons learned

Pretty much all automata library code is about performing very lightweight operations on very large collections of objects (states and transitions). To maximize the performance of such code, it is extremely important to make sure that a) the most efficient ways of iterating collections are being used and b) everything that can be inlined is inlined. In particular:

*   Avoid using collections through interfaces (like using `List<T>` through `IList<T>`). Even when it is in principle possible to infer the actual type of the object, JIT will not do that, emitting code for a virtual call instead.
    *   As a consequence: don't use `ReadOnlyCollection<T>` as it stores the underlying collection as `IList<T>`. Use `ReadOnlyList<T>` instead.
		
*   Don't use delegates for lightweight operations that will be performed on large numbers of objects. JIT does not inline delegate calls.
	
*   Indexing is usually more efficient than using iterators, and arrays usually more efficient than lists. Here is the duration of the same computation performed on different collection types, also using different ways to iterate them.

	```csharp
	struct Value {
		public int Unused { get; set; }
		public int V { get; set; }
		public int Unused2 { get; set; }
	}
	for (int i = 0; i < collection.Count; ++i) {
		result += result + collection[i].V + 1;
	}
	```

	```
	List<T> Indexing: 00:00:08.0507083
	List<T> Enumeration: 00:00:08.5108027	
	List<T> through IList<T> Indexing: 00:00:08.3555647
	List<T> through IList<T> Enumeration: 00:00:09.1270957
	T[] Indexing: 00:00:03.2190594
	T[] Enumeration: 00:00:06.2656707
	T[] through IList<T> Indexing: 00:00:08.3882060
	T[] through IList<T> Enumeration: 00:00:08.2368278
	```

	Some observations:
	*   Arrays are the fastest (presumably because lists create an intermediate copy of `Value` whenever it's being accessed)
	*   It is better to work with array and lists using indexing, not foreach.
	*   Working through `IList<T>` is very slow no matter what's the underlying collection type.


*   If you need to access the value stored at a particular list location more than once, it will make sense to store a copy of the value in a local variable, as every call to the list indexer will create a copy of the value. This is not true for arrays though, as accessing a value type in an array doesn't create copies. Here is an example to illustrate:

	```csharp
	struct Value {
		public int Unused { get; set; }
		public int V { get; set; }
		public int Unused2 { get; set; }
	}
	for (int i = 0; i < collection.Count; ++i) {
		result += result + collection[i].V * collection[i].V + 1;
	}
	for (int i = 0; i < collection.Count; ++i) {
		var v = collection[i];
		result += result + v.V * v.V + 1;
	}
	```

	```
	List<T> Indexing: 00:00:14.2150929
	List<T> Indexing + caching: 00:00:08.5155475
	T[] Indexing: 00:00:05.8134142
	T[] Indexing + caching: 00:00:06.9417585
	```	

*   String inference library tends to create tons of small short-lived objects (states, element distributions etc) that put a lot of burden on garbage collector. The workstation GC is particularly inefficient in handling it as it was optimized for low application latency (important for GUI apps, but not for us). Using server GC (which has been optimized for high throughput, that is, batch processing) with concurrency disabled seems to work much better with string inference.
	
*   Watch [this video](https://channel9.msdn.com/Events/TechEd/NorthAmerica/2013/DEV-B333#fbid=) for more info on writing high-performance managed code.

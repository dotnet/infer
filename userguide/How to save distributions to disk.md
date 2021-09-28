---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## How to save distributions to disk

The Infer.NET distribution classes such as Bernoulli and Gaussian are all marked Serializable and can be serialized by the standard .NET serialization mechanisms. These mechanisms are reviewed below.

### Binary format

The simplest and most efficient way to serialize data to disk is in binary format, using the class [System.Runtime.Serialization.Formatters.Binary.BinaryFormatter](https://docs.microsoft.com/en-us/dotnet/api/system.runtime.serialization.formatters.binary.binaryformatter?view=netframework-4.7.2). Here is an example of saving and loading a Dirichlet distribution:

```csharp
Dirichlet d = new Dirichlet(3.0, 1.0, 2.0);  
BinaryFormatter serializer = new BinaryFormatter();  
// write to disk  
using (FileStream stream = new FileStream("temp.bin", FileMode.Create)) {     
  serializer.Serialize(stream, d);  
}  
// read from disk  
using (FileStream stream = new FileStream("temp.bin", FileMode.Open)) {
  Dirichlet d2 = (Dirichlet)serializer.Deserialize(stream);
  Console.WriteLine(d2);
}
```
### XML format

Another option is to save in XML format, using the class [System.Runtime.Serialization.DataContractSerializer](https://docs.microsoft.com/ru-ru/dotnet/api/system.runtime.serialization.datacontractserializer?view=netframework-4.7.2). Because some distributions require custom XML serialization, you need to pass in an Infer.NET DataContractSurrogate when creating the serializer. Here is an example:  

```csharp
Dirichlet d = new Dirichlet(3.0, 1.0, 2.0);
string fileName = "temp.xml";
DataContractSerializer serializer = new DataContractSerializer(typeof (Dirichlet), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });  
// write to disk  
using (XmlDictionaryWriter writer = XmlDictionaryWriter.CreateTextWriter(new FileStream(fileName, FileMode.Create)))
{
    serializer.WriteObject(writer, d);
}
// read from disk
using (XmlDictionaryReader reader = XmlDictionaryReader.CreateTextReader(new FileStream(fileName, FileMode.Open), new XmlDictionaryReaderQuotas()))
{
    Dirichlet d2 = (Dirichlet)serializer.ReadObject(reader);
    Console.WriteLine(d2);
}
```

Note that DataContractSerializer is preferable to the older XmlSerializer class.

### JSON format

Infer.NET distributions are DataContract serializable, which largely makes them [JSON.NET](https://www.newtonsoft.com/json/help/html/SerializingJSON.htm) serializable. In order to use this library, you need to reference [Newtonsoft.Json.dll](https://www.newtonsoft.com/json) and add using statements to Newtonsoft.Json and Newtonsoft.Json.Serialization in your code.

```csharp
Dirichlet d = new Dirichlet(3.0, 1.0, 2.0);  
var serializerSettings = new JsonSerializerSettings {  
    TypeNameHandling = TypeNameHandling.Auto,  
    ContractResolver = new CollectionAsObjectResolver(),  
    PreserveReferencesHandling = PreserveReferencesHandling.Objects  
};  
var serializer = JsonSerializer.Create(serializerSettings);  
// write to disk  
using (FileStream stream = new FileStream("temp.json", FileMode.Create))  
{
  var streamWriter = new StreamWriter(stream);
  var jsonWriter = new JsonTextWriter(streamWriter);  
  serializer.Serialize(jsonWriter, d);  
  jsonWriter.Flush();  
}  
// read from disk  
using (FileStream stream = new FileStream("temp.json", FileMode.Open))  
{
  var streamReader = new StreamReader(stream);
  var jsonReader = new JsonTextReader(streamReader);
  Dirichlet d2 = serializer.Deserialize<Dirichlet>(jsonReader);  
  Console.WriteLine(d2);  
}
```

By default, JSON.NET serializes classes which implement `IList<T>` as arrays (a list of comma-separated values). However, such classes typically contain more information than this, and therefore need to be serialized as objects. This is true for many of the members of the Infer.NET distributions, and therefore the following contract resolver must be used. It has to be set in the JSON serializer settings, as shown above.

```csharp
///  <summary>  
/// Treats as objects distribution member types which implement <see cref="IList{T}"/>. 
///  </summary>  
class  CollectionAsObjectResolver : DefaultContractResolver  
{  
  private static readonly HashSet<Type> SerializeAsObjectTypes = new HashSet<Type>
  {  
    typeof(Vector),
    typeof(Matrix),
    typeof(IArray<>),
    typeof(ISparseList<>)  
  };
  private static readonly ConcurrentDictionary<Type, JsonContract> ResolvedContracts = new ConcurrentDictionary<Type, JsonContract>();  
  public override JsonContract ResolveContract(Type type) => ResolvedContracts.GetOrAdd(type, this.ResolveContractInternal);  
  private JsonContract ResolveContractInternal(Type type) => IsExcludedType(type)? this.CreateObjectContract(type): this.CreateContract(type);  
  private static bool IsExcludedType(Type type)
  {
    if (type == null) return false;
    if (SerializeAsObjectTypes.Contains(type)) return true;
    if (type.IsGenericType && SerializeAsObjectTypes.Contains(type.GetGenericTypeDefinition())) return true;
    return IsExcludedType(type.BaseType) || type.GetInterfaces().Any(IsExcludedType);  
   }
}
```

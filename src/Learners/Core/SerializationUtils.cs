using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    /// <summary>
    /// Utils
    /// </summary>
    public static class SerializationUtils
    {
        /// <summary>
        /// Gets serializer.
        /// </summary>
        /// <returns>A serializer.</returns>
        public static JsonSerializer GetJsonSerializer()
        {
            var serializerSettings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                ContractResolver = new CollectionAsObjectResolver(),
                PreserveReferencesHandling = PreserveReferencesHandling.Objects,

            };
            var serializer = JsonSerializer.Create(serializerSettings);
            return serializer;
        }

        /// <summary>
        /// Gets a formatter.
        /// </summary>
        /// <returns>A formatter.</returns>
        public static IFormatter GetJsonFormatter() =>
            new JsonFormatter();

        private sealed class JsonFormatter : IFormatter
        {
            //private readonly Dictionary<Stream, object[]> items = new Dictionary<Stream, object[]>();
            //private readonly Dictionary<Stream, int> itemCounts = new Dictionary<Stream, int>();


            public SerializationBinder Binder { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public StreamingContext Context { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public ISurrogateSelector SurrogateSelector { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

            public object Deserialize(Stream serializationStream)
            {
                //////if (!items.ContainsKey(serializationStream))
                //////{
                //////    var streamReader = new StreamReader(serializationStream);
                //////    var jsonReader = new JsonTextReader(streamReader);
                //////    var obj = jsonSerializer.Deserialize(jsonReader);
                //////    items.Add(serializationStream, (object[])obj);
                //////}

                ////var digits = int.MaxValue.ToString().Length;
                ////char[] lengthBuffer = new char[digits];
                ////using (var sr = new StreamReader(serializationStream, Encoding.Default, detectEncodingFromByteOrderMarks: true, bufferSize: 1024, leaveOpen: true))
                ////{
                ////    var bytesRead = sr.Read(lengthBuffer, 0, lengthBuffer.Length);
                ////    if (bytesRead == 0)
                ////    {
                ////        Debugger.Break();
                ////    }

                ////    if (bytesRead != lengthBuffer.Length)
                ////    {
                ////        Debugger.Break();
                ////    }

                ////    var lengthString = new string(lengthBuffer);
                ////    var length = int.Parse(lengthString);

                ////    char[] messageBuffer = new char[length];
                ////    var messageBytesRead = sr.Read(messageBuffer, 0, length);
                ////    if (messageBytesRead == 0)
                ////    {
                ////        Debugger.Break();
                ////    }

                ////    if (messageBytesRead != messageBuffer.Length)
                ////    {
                ////        Debugger.Break();
                ////    }

                ////    var messageString = new string(messageBuffer);
                ////    var sr5 = new StringReader(messageString);

                ////    JsonSerializer serializer = new JsonSerializer();
                ////    var jsonReader123 = new JsonTextReader(sr5);
                ////    //var role = ((Wrapper)serializer.Deserialize(reader)).stored;
                ////    var role = ((JObject)serializer.Deserialize(jsonReader123)).ToObject<DummyWrapper>().obj;

                ////    return role;
                ////}

                //////var streamReader = new StreamReader/(/serializationStream);
                //////var jsonReader = new JsonTextReader(/streamReader) /{ SupportMultipleContent = t/rue };
                ////////var obj = jsonSerializer.Deserialize/(/jsonReader);
                ////////return obj;
                ////////var obj = (JObject)jsonSerializer.Deserialize/(/jsonReader);
                ////////var wrapper = obj.ToObject<DummyWrapper>();
                ////////return wrapper.obj;
                //////var wrapper = (DummyWrapper)/j/sonSerializer.Deserialize(jsonReader);
                //////return wrapper.obj;

                var lengthBuffer = new byte[4]; // The length is an int.
                var bytesRead = serializationStream.Read(lengthBuffer, 0, lengthBuffer.Length);
                if (bytesRead != lengthBuffer.Length)
                {
                    throw new Exception("Could not read object. Missing length header.");
                }

                int length = checked(lengthBuffer[0] + (256 * lengthBuffer[1]) + (256 * 256 * lengthBuffer[2]) + (256 * 256 * 256 * lengthBuffer[3]));

                var messageBuffer = new byte[length];
                var messageBytesRead = serializationStream.Read(messageBuffer, 0, length);
                if (messageBytesRead != messageBuffer.Length)
                {
                    throw new Exception("Could not read object. Missing content.");
                }

                var messageBufferStream = new MemoryStream();
                messageBufferStream.Write(messageBuffer, 0, messageBuffer.Length);
                messageBufferStream.Position = 0;

                using (var streamReader = new StreamReader(messageBufferStream))
                {
                    var jsonSerializer = GetJsonSerializer();
                    var jsonReader123 = new JsonTextReader(streamReader);
                    var role = ((DummyWrapper)jsonSerializer.Deserialize(jsonReader123)).obj;

                    return role;
                }
            }

            public void Serialize(Stream serializationStream, object graph)
            {
                ////var sb = new StringBuilder();
                ////using (var writer = new StringWriter(sb))
                ////{
                ////    using (var jw = new JsonTextWriter(writer))
                ////    {
                ////        jsonSerializer.Serialize(writer, new DummyWrapper { obj = graph });
                ////    }
                ////}

                ////var digits = int.MaxValue.ToString().Length;
                ////var digitSpecifier = $"D{digits}";

                ////using (var streamWriter = new StreamWriter(serializationStream, Encoding.Default, 1024, leaveOpen: true))
                ////{
                ////    streamWriter.Write(sb.Length.ToString(digitSpecifier));
                ////    streamWriter.Write(sb.ToString());
                ////}

                //////using (var streamWriter = new StreamWriter(serializationStream, Encoding.UTF8, 1024, /leaveOpen: /true))
                //////{
                //////    //var wrapper = new DummyWrapper { obj = graph };
                //////    //jsonSerializer.Serialize(streamWriter, wrapper);
                //////    //jsonSerializer.Serialize(streamWriter, graph);
                //////    jsonSerializer.Serialize(streamWriter, new DummyWrapper { obj = graph });
                //////}

                var jsonSerializer = GetJsonSerializer();
                using (var serializedGraphBuffer = new MemoryStream())
                {
                    using (var serializedGraphBufferWriter = new StreamWriter(serializedGraphBuffer))
                    {
                        using (var serializedGraphBufferJsonWriter = new JsonTextWriter(serializedGraphBufferWriter))
                        {
                            jsonSerializer.Serialize(serializedGraphBufferJsonWriter, new DummyWrapper { obj = graph });

                            serializedGraphBufferJsonWriter.Flush();
                            serializedGraphBufferWriter.Flush();

                            checked
                            {
                                var messageLength = (int)serializedGraphBuffer.Length;
                                var messageLengthBuffer = new byte[4];
                                messageLengthBuffer[0] = (byte)(messageLength % 256);
                                messageLength /= 256;
                                messageLengthBuffer[1] = (byte)(messageLength % 256);
                                messageLength /= 256;
                                messageLengthBuffer[2] = (byte)(messageLength % 256);
                                messageLength /= 256;
                                if (messageLength >= 256)
                                {
                                    throw new Exception($"Message of length {serializedGraphBuffer.Length} is not supported.");
                                }
                                messageLengthBuffer[3] = (byte)messageLength;

                                serializationStream.Write(messageLengthBuffer, 0, messageLengthBuffer.Length);

                                serializedGraphBuffer.Position = 0;
                                serializedGraphBuffer.WriteTo(serializationStream);
                            }
                        }
                    }
                }
            }

            private class DummyWrapper
            {
                public object obj { get; set; }
            }
        }

        /// <summary>
        /// Treats as objects distribution member types which implement <see cref="IList{T}"/>.
        /// </summary>
        private class CollectionAsObjectResolver : DefaultContractResolver
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

            private JsonContract ResolveContractInternal(Type type) => IsExcludedType(type)
                ? this.CreateObjectContract(type)
                : this.CreateContract(type);

            private static bool IsExcludedType(Type type)
            {
                if (type == null) return false;
                if (SerializeAsObjectTypes.Contains(type)) return true;
                if (type.IsGenericType && SerializeAsObjectTypes.Contains(type.GetGenericTypeDefinition())) return true;
                return IsExcludedType(type.BaseType) || type.GetInterfaces().Any(IsExcludedType);
            }
        }
    }
}

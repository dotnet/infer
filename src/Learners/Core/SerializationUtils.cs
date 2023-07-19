// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Runtime.Serialization;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Newtonsoft.Json;
    using Newtonsoft.Json.Serialization;

    /// <summary>
    /// Utilities for serialization.
    /// </summary>
    public static class SerializationUtils
    {
        /// <summary>
        /// Gets a JSON serializer.
        /// </summary>
        /// <returns>A JSON serializer.</returns>
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
        /// Gets a JSON formatter.
        /// </summary>
        /// <returns>A JSON formatter.</returns>
        public static IFormatter GetJsonFormatter() =>
            new JsonFormatter();

        private sealed class JsonFormatter : IFormatter
        {
            public SerializationBinder Binder { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public StreamingContext Context { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public ISurrogateSelector SurrogateSelector { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

            public object Deserialize(Stream serializationStream)
            {
                // Newtonsoft.JSON does have an out-of-box SupportMultipleContent setting but it only works for
                // a series of JSON objects, and when there are also values embedded then it fails.
                // Therefore we write for each content section in the stream an integer length. Int32 is fine because
                // we can't have strings larger than this anyway.
                // Note that a TextReader/StreamReader cannot be used here and we read the Stream directly because
                // the readers buffer ahead and the IFormatter.Deserialize interface is that when we return the Steam
                // should be up to (and not beyond) the next value in the stream.
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

                            // Newtonsoft.JSON does have an out-of-box SupportMultipleContent setting but it only works for
                            // a series of JSON objects, and when there are also values embedded then it fails.
                            // Therefore we write for each content section in the stream an integer length. Int32 is fine because
                            // we can't have strings larger than this anyway
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

            /// <summary>
            /// A wrapper object used to force Newtonsoft.JSON to write out the type
            /// information for the wrapped object so that it is deserialized into that object.
            /// </summary>
            private class DummyWrapper
            {
                /// <summary>
                /// Gets or sets the wrapped object.
                /// </summary>
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

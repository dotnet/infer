﻿using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    public static class SerializationUtils
    {
        public static JsonSerializer GetJsonSerializer()
        {
            var serializerSettings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.Auto,
                ContractResolver = new CollectionAsObjectResolver(),
                PreserveReferencesHandling = PreserveReferencesHandling.Objects
            };
            var serializer = JsonSerializer.Create(serializerSettings);
            return serializer;
        }

        public static IFormatter GetJsonFormatter() =>
            new JsonFormatter();

        private sealed class JsonFormatter : IFormatter
        {
            private readonly JsonSerializer jsonSerializer = GetJsonSerializer();

            public SerializationBinder Binder { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public StreamingContext Context { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public ISurrogateSelector SurrogateSelector { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

            public object Deserialize(Stream serializationStream)
            {
                var streamReader = new StreamReader(serializationStream);
                var jsonReader = new JsonTextReader(streamReader);
                return jsonSerializer.Deserialize(jsonReader);
            }

            public void Serialize(Stream serializationStream, object graph)
            {
                using (var streamWriter = new StreamWriter(serializationStream, Encoding.UTF8, 1024, leaveOpen: true))
                {
                    jsonSerializer.Serialize(streamWriter, graph);
                }
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

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    public sealed class InferDataContractResolverTests
    {
        class X<T>
        {
            public class Y<U>
            {
                public class Z<V>
                {
                }
            }
        }

        [Fact]
        public void InferDataContractResolverTests_RecoversTypes()
        {
            TestCase(typeof(Tuple<Tuple<List<string>, Tuple<int>>>));
            TestCase(typeof(string));
            TestCase(typeof(string[]));
            TestCase(typeof(List<string>));
            TestCase(typeof(X<string>.Y<int[]>.Z<char>));
            TestCase(typeof(X<Tuple<List<string>, Tuple<int[]>>>.Y<Tuple<List<string>, Tuple<int>>>.Z<Tuple<List<string>, Tuple<int>[]>>));

            TestAssembly(typeof(string).Assembly);
            TestAssembly(typeof(Gaussian).Assembly);

            void TestAssembly(Assembly assembly)
            {
                var types = assembly.GetTypes();
                foreach (var type in types)
                {
                    TestCase(type);
                }
            }

            void TestCase(Type type)
            {
                var mentionedTypes = InferDataContractResolver.GetMentionedTypes(type.FullName);

                var typeTree = AddMentionedTypes(type);

                void AssertAreEqual(InferDataContractResolver.TypeId x, InferDataContractResolver.TypeId y)
                {
                    if (x.Name != y.Name)
                    {
                        throw new InvalidOperationException("Incorrect name");
                    }

                    if (x.IsArray != y.IsArray)
                    {
                        throw new InvalidOperationException($"Incorrect isArray");
                    }

                    if (x.Arguments.Length != y.Arguments.Length)
                    {
                        throw new InvalidOperationException($"Incorrect length");
                    }

                    for (int i = 0; i < x.Arguments.Length; i++)
                    {
                        AssertAreEqual(x.Arguments[i], y.Arguments[i]);
                    }
                }

                InferDataContractResolver.TypeId AddMentionedTypes(Type mentionedType)
                {
                    if (mentionedType.IsConstructedGenericType)
                    {
                        var genericType = mentionedType.GetGenericTypeDefinition();
                        var typeArguments = mentionedType
                            .GenericTypeArguments
                            .Select(AddMentionedTypes)
                            .ToArray();

                        return new InferDataContractResolver.TypeId(genericType.FullName, typeArguments, isArray: false);
                    }

                    if (mentionedType.IsArray)
                    {
                        var elementType = AddMentionedTypes(mentionedType.GetElementType());
                        return new InferDataContractResolver.TypeId(elementType.Name, elementType.Arguments, isArray: true);
                    }

                    return new InferDataContractResolver.TypeId(mentionedType.FullName, new InferDataContractResolver.TypeId[0], isArray: false);
                }
            }
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
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

        private static void VisitTypesRecursively(Type[] types, Action<Type> visitType)
        {
            var alreadyReviewed = new HashSet<Type>();

            foreach (var type in types)
            {
                VisitTypesRecursively(type, visitType, alreadyReviewed);
            }
        }

        private static void VisitTypesRecursively(Type type, Action<Type> visitType, HashSet<Type> alreadyReviewed)
        {
            // We aren't not interested in exceptions.
            if (type.IsSubclassOf(typeof(Exception)))
            {
                return;
            }

            // We aren't interested in delegates.
            if (type.IsSubclassOf(typeof(Delegate)))
            {
                return;
            }

            // There's nothing for us to test regarding open type parameters.
            if (type.ContainsGenericParameters)
            {
                return;
            }

            // Avoid repeating review of type.
            if (!alreadyReviewed.Add(type))
            {
                return;
            }

            visitType(type);

            if (type.BaseType != null)
            {
                VisitTypesRecursively(type.BaseType, visitType, alreadyReviewed);
            }

            foreach (var property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                // We remove "by ref" because we don't care about this for serialization.
                var typeToTest = property.PropertyType;

                if (typeToTest.IsByRef)
                {
                    typeToTest = typeToTest.GetElementType();
                }

                VisitTypesRecursively(typeToTest, visitType, alreadyReviewed);
            }

            foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                VisitTypesRecursively(field.FieldType, visitType, alreadyReviewed);
            }
        }

        [Fact]
        public void InferDataContractResolverTests_RecoversTypes()
        {
            var alreadyTested = new HashSet<Type>();

            TestCase(typeof(double[,]));

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
                VisitTypesRecursively(types, TestCase);
            }

            void TestCase(Type type)
            {
                // Avoid repeating tests.
                if (!alreadyTested.Add(type))
                {
                    return;
                }

                var mentionedTypes = InferDataContractResolver.ParseTypeString(type.FullName);

                var typeTree = GetParsedTypeString(type);

                AssertAreEqual(mentionedTypes, typeTree);

                void AssertAreEqual(InferDataContractResolver.ParsedTypeString x, InferDataContractResolver.ParsedTypeString y)
                {
                    if (x.Name != y.Name)
                    {
                        throw new InvalidOperationException("Incorrect name");
                    }

                    if (x.ArrayLayout.Length != y.ArrayLayout.Length)
                    {
                        throw new InvalidOperationException($"Incorrect isArray");
                    }

                    for (int i = 0; i < x.ArrayLayout.Length; i++)
                    {
                        if (x.ArrayLayout[i] != y.ArrayLayout[i])
                        {
                            throw new InvalidOperationException($"Incorrect array layout entry");
                        }
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

                InferDataContractResolver.ParsedTypeString GetParsedTypeString(Type mentionedType)
                {
                    if (mentionedType.IsConstructedGenericType)
                    {
                        var genericType = mentionedType.GetGenericTypeDefinition();
                        var typeArguments = mentionedType
                            .GenericTypeArguments
                            .Select(GetParsedTypeString)
                            .ToArray();

                        return new InferDataContractResolver.ParsedTypeString(genericType.FullName, typeArguments, arrayLayout: new int[0]);
                    }

                    if (mentionedType.IsArray)
                    {
                        var arrayLayout = new List<int>();
                        while (mentionedType.IsArray)
                        {
                            arrayLayout.Add(mentionedType.GetArrayRank());
                            mentionedType = mentionedType.GetElementType();
                        }
                        var elementType = GetParsedTypeString(mentionedType);
                        return new InferDataContractResolver.ParsedTypeString(elementType.Name, elementType.Arguments, arrayLayout: arrayLayout.ToArray());
                    }

                    return new InferDataContractResolver.ParsedTypeString(mentionedType.FullName, new InferDataContractResolver.ParsedTypeString[0], arrayLayout: new int[0]);
                }
            }
        }

        [Fact]
        public void InferDataContractResolverTests_RoundTripTypes()
        {
            var alreadyReviewed = new HashSet<Type>();

            TestCase(typeof(Tuple<Tuple<List<string>, Tuple<int>>>));
            TestCase(typeof(string));
            TestCase(typeof(string[]));
            TestCase(typeof(List<string>));

            var runtimeTypes = typeof(Gaussian)
                .Assembly
                .GetTypes();

            foreach (var runtimeType in runtimeTypes)
            {
                // We only test data contracts.
                if (runtimeType.GetCustomAttribute<DataContractAttribute>() == null)
                {
                    continue;
                }

                // First do a shallow test on the data contract itself.
                TestCase(runtimeType);

                // Then do a deep test on its public properties.
                foreach (var property in runtimeType.GetProperties())
                {
                    VisitTypesRecursively(property.PropertyType, TestCase, alreadyReviewed);
                }

                // Then do a deep test on its public fields.
                foreach (var field in runtimeType.GetFields())
                {
                    VisitTypesRecursively(field.FieldType, TestCase, alreadyReviewed);
                }
            }

            void TestCase(Type type)
            {
                var reconstructedType = InferDataContractResolver.ConstructTypeFromString(type.FullName);
                if (type.FullName != reconstructedType.FullName)
                {
                    throw new InvalidOperationException("Round trip failed");
                }
            }
        }
    }
}

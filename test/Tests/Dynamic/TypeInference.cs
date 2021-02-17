// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler.Reflection;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Xunit;
using Assert = Xunit.Assert;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class TypeInferenceTests
    {
        [Fact]
        public void NullableTest()
        {
            Assert.True(Conversion.IsNullable(typeof (string)));
            Assert.False(Conversion.IsNullable(typeof (double)));
            Assert.True(Conversion.IsNullable(typeof (double?)));
            Assert.True(Conversion.IsNullable(typeof (int?)));
            Assert.True(Conversion.IsNullable(typeof (KeyValuePair<int, int>?)));
        }

        [Fact]
        public void ConversionRankingTest()
        {
            Conversion conv, conv2;
            Conversion.TryGetConversion(typeof (Single), typeof (double), out conv);
            Conversion.TryGetConversion(typeof (double), typeof (Single), out conv2);
            Assert.True(conv < conv2);
            Conversion.TryGetConversion(typeof (Single), typeof (Single), out conv2);
            Assert.True(conv > conv2);
            Conversion.TryGetConversion(typeof (Single), typeof (object), out conv);
            Assert.True(conv > conv2);
            Conversion.TryGetConversion(typeof (int[]), typeof (int[]), out conv);
            Conversion.TryGetConversion(typeof (int[]), typeof (IList<int>), out conv2);
            Assert.True(conv < conv2);
            Conversion.TryGetConversion(typeof (int[]), typeof (IList<object>), out conv);
            Assert.True(conv > conv2);
            Conversion.TryGetConversion(typeof (TestEnum), typeof (TestEnum), out conv);
            Conversion.TryGetConversion(typeof (TestEnum), typeof (int), out conv2);
            Assert.True(conv < conv2);
        }

        [Fact]
        public void ConversionTest()
        {
            Single s = 2.5F;
            double d;
            Conversion conv;
            // primitives
            Assert.True(Conversion.TryGetConversion(typeof (Single), typeof (double), out conv));
            Assert.False(conv.IsExplicit);
            object o = conv.Converter(s);
            d = (double) o;
            Console.WriteLine("{0}: {1}", o.GetType(), d);
            Assert.True(Conversion.TryGetConversion(typeof (double), typeof (Single), out conv));
            Assert.True(conv.IsExplicit);
            o = conv.Converter(d);
            s = (Single) o;
            Console.WriteLine("{0}: {1}", o.GetType(), s);
            Assert.True(Conversion.TryGetConversion(typeof (object), typeof (double), out conv));
            Assert.True(conv.IsExplicit);
            o = conv.Converter(s);
            d = (double) o;
            Console.WriteLine("{0}: {1}", o.GetType(), d);
            Assert.True(Conversion.TryGetConversion(typeof (double), typeof (int), out conv));
            Assert.True(conv.IsExplicit);
            Assert.Throws<ArgumentException>(() =>
            {
                o = conv.Converter(d);
            });

            // enums
            Assert.True(Conversion.TryGetConversion(typeof (string), typeof (BindingFlags), out conv));
            Assert.True(conv.IsExplicit);
            BindingFlags flags = (BindingFlags) conv.Converter("Public");

            // array up-conversion
            Assert.True(Conversion.TryGetConversion(typeof (double), typeof (double[]), out conv));
            o = conv.Converter(d);
            double[] array = (double[]) o;
            Console.WriteLine("{0}: {1}", o.GetType(), array[0]);
            Assert.True(Conversion.TryGetConversion(typeof (double[]), typeof (double[,]), out conv));
            o = conv.Converter(array);
            double[,] array2 = (double[,]) o;
            Console.WriteLine("{0}: {1}", o.GetType(), array2[0, 0]);
            Assert.True(Conversion.TryGetConversion(typeof (System.Reflection.Missing), typeof (double[]), out conv));
            o = conv.Converter(d);
            array = (double[]) o;
            Assert.Empty(array);
            Console.WriteLine("{0}: Length={1}", o.GetType(), array.Length);

            // class up-conversion
            Assert.True(Conversion.TryGetConversion(typeof (PositiveDefiniteMatrix), typeof (Matrix), out conv));
            Assert.True(conv.Converter == null);

            // interface up-conversion
            Assert.True(Conversion.TryGetConversion(typeof (double[]), typeof (IList), out conv));
            Assert.True(conv.Converter == null);

            // array covariance (C# 2.0 specification, sec 20.5.9)
            Assert.True(Conversion.TryGetConversion(typeof (string[]), typeof (IList<string>), out conv));
            Assert.True(conv.Converter == null);
            Assert.True(Conversion.TryGetConversion(typeof (string[]), typeof (IList<object>), out conv));
            Assert.True(conv.Converter == null);
            Assert.False(Conversion.TryGetConversion(typeof (string[,]), typeof (IList<string>), out conv));
            Assert.False(Conversion.TryGetConversion(typeof (string[,]), typeof (IList<object>), out conv));

            // array element conversion
            Assert.True(Conversion.TryGetConversion(typeof (object[]), typeof (double[]), out conv));
            Assert.True(conv.IsExplicit);
            object[] oa = new object[2] {1.1, 2.2};
            o = conv.Converter(oa);
            array = (double[]) o;
            Assert.Equal(oa[0], array[0]);
            Assert.Equal(oa[1], array[1]);
            Console.WriteLine("{0}: {1} {2}", o.GetType(), array[0], array[1]);

            // array down-conversion
            Assert.True(Conversion.TryGetConversion(typeof (double[,]), typeof (double[]), out conv));
            array2 = new double[,] {{1.1, 2.2}};
            o = conv.Converter(array2);
            array = (double[]) o;
            Console.WriteLine("{0}: {1} {2}", o.GetType(), array[0], array[1]);

            // custom conversion
            Assert.True(Conversion.TryGetConversion(typeof (Tester), typeof (int), out conv));
            Tester t = new Tester();
            o = conv.Converter(t);
            Console.WriteLine("{0}: {1}", o.GetType(), o);
            Assert.True(Conversion.TryGetConversion(typeof (Tester), typeof (int[]), out conv));
            o = conv.Converter(t);
            Console.WriteLine("{0}: {1}", o.GetType(), o);
            Assert.True(Conversion.TryGetConversion(typeof(int), typeof(Tester), out conv));
            Assert.True(conv.IsExplicit);
            Assert.True(Conversion.TryGetConversion(typeof(int[]), typeof(Tester), out conv));
            Assert.True(conv.IsExplicit);
            Assert.True(Conversion.TryGetConversion(typeof(ImplicitlyConvertibleToTesterDefinesCast), typeof(Tester), out conv));
            Assert.False(conv.IsExplicit);
            Assert.True(Conversion.TryGetConversion(typeof(ImplicitlyConvertibleToTesterCastDefinedOnTester), typeof(Tester), out conv));
            Assert.False(conv.IsExplicit);

            // conversion from null
            Assert.False(Conversion.TryGetConversion(typeof (Nullable), typeof (int), out conv));
            Assert.True(Conversion.TryGetConversion(typeof (Nullable), typeof (int?), out conv));
            Assert.True(Conversion.TryGetConversion(typeof (Nullable), typeof (int[]), out conv));
        }

        [Fact]
        public void GenericParameterConversionTest()
        {
            Type t = typeof (Tester<>).GetGenericArguments()[0];
            Type[] typeParams = typeof (TypeInferenceTests).GetMethod("CanSample").GetGenericArguments();
            Type distributionType = typeParams[0];
            Type domainType = typeParams[1];
            Conversion conv;
            Assert.True(Conversion.TryGetConversion(t, t, out conv));
            Assert.True(conv.Converter == null);
            Assert.False(Conversion.TryGetConversion(distributionType, domainType, out conv));
            Assert.True(Conversion.TryGetConversion(distributionType, typeof (Sampleable<>).MakeGenericType(domainType), out conv));
            Assert.True(conv.Converter == null);
        }

        [Fact]
        public void OverloadingTest()
        {
            double d = 2.0;
            Single s = 1.1F;
            Assert.True((int) Invoker.InvokeStatic(typeof (TypeInferenceTests), "Overloaded", 4) == 1);
            Assert.True((int) Invoker.InvokeStatic(typeof (TypeInferenceTests), "Overloaded", d) == 2);
            Assert.True((int) Invoker.InvokeStatic(typeof (TypeInferenceTests), "Overloaded", s) == 2);
            Assert.True((int) Invoker.InvokeStatic(typeof (TypeInferenceTests), "Overloaded", d, d) == 2);
            // C# does not allow this call:
            // cannot convert from 'double' to 'int'
            // Console.WriteLine(Overloaded(d,d));
        }

        [Fact]
        public void ArrayCovarianceTest()
        {
            string[] strings = new string[1];
            Invoker.InvokeStatic(typeof (ListMethods), "StringListMethod", strings);
            Invoker.InvokeStatic(typeof (ListMethods), "ObjectListMethod", strings);
            Invoker.InvokeStatic(typeof (ListMethods), "GenericListMethod", strings, "hello");
            Invoker.InvokeStatic(typeof (ListMethods), "GenericListMethod", strings, new object());
        }

        public static int Overloaded(int x)
        {
            return 1;
        }

        public static int Overloaded(double x)
        {
            return 2;
        }

        public static int Overloaded(int x, double y)
        {
            return 1;
        }

        public static int Overloaded(double x, int y)
        {
            return 2;
        }

        [Fact]
        public void TypeInferenceTest()
        {
            Console.WriteLine("PositiveDefiniteMatrix is assignable to:");
            TestUtils.PrintCollection<Type>(Binding.TypesAssignableFrom(typeof (PositiveDefiniteMatrix)));
            Console.WriteLine();
            MethodInfo method;
            method = typeof (TypeInferenceTests).GetMethod("DelayedSetTo");
            Type formal = method.GetParameters()[0].ParameterType;
            Type actual = typeof (SettableTo<double>);
            TestInferGenericParameters(formal, actual, method, 1);

            actual = typeof (Matrix);
            TestInferGenericParameters(formal, actual, method, 1);

            method = typeof (TypeInferenceTests).GetMethod("Method1", BindingFlags.NonPublic | BindingFlags.Static);
            Type[] args1 = {typeof (Matrix)};
            TestInferGenericParameters(method, args1, 1);
            Console.WriteLine("Inferring {0} from {1}: ", method, typeof (double));
            Assert.Throws<ArgumentException>(() =>
            {
                Invoker.Invoke(method, null, 4.4);
            });

            method = typeof (TypeInferenceTests).GetMethod("DelayedSetTo");
            Type[] args2 = {typeof (Matrix), typeof (Matrix)};
            TestInferGenericParameters(method, args2, 1);

            method = typeof (TypeInferenceTests).GetMethod("DelayedSampleInto");
            Type[] args3 = {typeof (VectorGaussian), typeof (VectorGaussian)};
            TestInferGenericParameters(method, args3, 1);
            VectorGaussian g = new VectorGaussian(1);
            Invoker.Invoke(method, null, g, g);
            args3[0] = typeof (VectorGaussian[]);
            args3[1] = args3[0];
            method = typeof (TypeInferenceTests).GetMethod("SameSampleableArray");
            TestInferGenericParameters(method, args3, 1);
            VectorGaussian[] ga = {g};
            Assert.Equal(typeof (Vector), (Type) Invoker.Invoke(method, null, ga, ga));

            method = typeof (TypeInferenceTests).GetMethod("SameType");
            Type t = typeof (Matrix);
            int numMatrixParents = 11;
            args2[0] = t;
            args2[1] = t;
            TestInferGenericParameters(method, args2, numMatrixParents);
            Matrix x = new Matrix(1, 1);
            Assert.Equal(t, (Type) Invoker.Invoke(method, null, x, x));
            Assert.Equal(t, (Type) Invoker.Invoke(method.MakeGenericMethod(t), null, x, x));
            args2[0] = t;
            args2[1] = typeof (Nullable);
            TestInferGenericParameters(method, args2, numMatrixParents);
            Assert.Equal(t, (Type) Invoker.Invoke(method, null, x, null));
            args2[0] = typeof (Nullable);
            args2[1] = t;
            TestInferGenericParameters(method, args2, numMatrixParents);
            Assert.Equal(t, (Type) Invoker.Invoke(method, null, null, x));
            args2[0] = typeof (Nullable);
            args2[1] = typeof (Nullable);
            TestInferGenericParameters(method, args2, 1);
            Assert.Equal(typeof (object), (Type) Invoker.Invoke(method, null, null, null));
            args2[0] = typeof (double);
            args2[1] = typeof (int);
            TestInferGenericParameters(method, args2, 6);
            Assert.Equal(typeof (double), (Type) Invoker.Invoke(method, null, 2.5, 7));


            method = typeof (TypeInferenceTests).GetMethod("SameArray");
            args2[0] = typeof (Matrix[]);
            args2[1] = typeof (Matrix[]);
            TestInferGenericParameters(method, args2, numMatrixParents);
            Matrix[] a = {x};
            Assert.Equal(t, (Type) Invoker.Invoke(method, null, a, a));

            method = typeof (System.Math).GetMethod("Sqrt");
            Assert.Equal(System.Math.Sqrt(4.0), (double)Invoker.Invoke(method, null, 4.0));
        }

        [Fact]
        public void OpenTypeInferenceTest()
        {
            MethodInfo method;
            Type[] args2 = new Type[2];

            method = typeof (Tester<>).GetMethod("SameType");
            args2[0] = typeof (double);
            args2[1] = typeof (double);
            TestInferGenericParameters(method, args2, 8);
            //Assert.Equal(typeof(double), (Type)Invoker.Invoke(method, null, 4.0, 5.0));
            method = typeof (Tester<>).GetMethod("Method2");
            args2[0] = typeof (double);
            args2[1] = typeof (int);
            TestInferGenericParameters(method, args2, 8*8);
            //Assert.Equal(typeof(double), (Type)Invoker.Invoke(method, null, 4.0, 5));
        }

        // Fails for versions of .NET below .NET 4
        [Fact]
        public void ClosedTypeInferenceTest()
        {
            MethodInfo method;
            Type[] args = new Type[1];

            method = typeof (Tester<bool>).GetMethod("Method1");
            args[0] = typeof (bool[]);
            TestInferGenericParameters(method, args, 14);
        }

        [Fact]
        public void BindToGenericParameterTest()
        {
            MethodInfo method;
            Type[] args = new Type[2];

            Type t = typeof (Tester<>).GetGenericArguments()[0];
            Type distributionType = typeof (TypeInferenceTests).GetMethod("CanSample").GetGenericArguments()[0];
            Type t3 = typeof (Sampleable<>);

            method = typeof (Tester<>).GetMethod("SameType");
            args[0] = t;
            args[1] = t;
            TestInferGenericParameters(method, args, 2);
            args[0] = distributionType;
            args[1] = distributionType;
            TestInferGenericParameters(method, args, 3);
            args[0] = t3;
            args[1] = t3;
            TestInferGenericParameters(method, args, 1);

            method = typeof (Tester<>).GetMethod("Method2");
            args[0] = t;
            args[1] = t;
            TestInferGenericParameters(method, args, 2*2);
            args[0] = distributionType;
            args[1] = distributionType;
            TestInferGenericParameters(method, args, 3*3);

            method = typeof (TypeInferenceTests).GetMethod("SameSampleableArray");
            Type sampleableArray = distributionType.MakeArrayType();
            args[0] = sampleableArray;
            args[1] = sampleableArray;
            TestInferGenericParameters(method, args, 1);
        }

        [Fact]
        public void SpecialConstraintTest()
        {
            MethodInfo method;
            Type[] args = new Type[1];

            method = typeof (TypeInferenceTests).GetMethod("ConstrainClass");
            args[0] = typeof (object);
            TestInferGenericParameters(method, args, 1);
            args[0] = typeof (int);
            TestInferGenericParameters(method, args, 7);
            args[0] = typeof (ICloneable);
            TestInferGenericParameters(method, args, 1);

            method = typeof (TypeInferenceTests).GetMethod("ConstrainStruct");
            args[0] = typeof (object);
            TestInferGenericParameters(method, args, 0);
            args[0] = typeof (int);
            TestInferGenericParameters(method, args, 1);
            args[0] = typeof (ICloneable);
            TestInferGenericParameters(method, args, 0);
            args[0] = typeof (int?);
            TestInferGenericParameters(method, args, 0);
            // to test that this is invalid:
            // ConstrainStruct<int?>(null);

            method = typeof (TypeInferenceTests).GetMethod("ConstrainConstructor");
            args[0] = typeof (object);
            TestInferGenericParameters(method, args, 1);
            args[0] = typeof (int);
            TestInferGenericParameters(method, args, 2);
            args[0] = typeof (ICloneable);
            TestInferGenericParameters(method, args, 0);
        }

        internal static void Method1<T>(SettableTo<T> x)
        {
        }

        public static Action DelayedSetTo<T>(SettableTo<T> target, T value)
        {
            return delegate () { target.SetTo(value); };
        }

        public static Action DelayedSampleInto<T>(Sampleable<T> sourceDist, HasPoint<T> targetDist)
        {
            return delegate() { targetDist.Point = sourceDist.Sample(); };
        }

        public static Type SameType<T>(T a, T b)
        {
            Console.WriteLine("Selected T = " + typeof (T));
            return typeof (T);
        }

        public static Type SameArray<T>(T[] a, T[] b)
        {
            Console.WriteLine("Selected T = " + typeof (T));
            return typeof (T);
        }

        public static Type SameSampleableArray<T>(Sampleable<T>[] a, Sampleable<T>[] b)
        {
            Console.WriteLine("Selected T = " + typeof (T));
            return typeof (T);
        }

        public static Type ConstrainClass<T>(T value)
            where T : class
        {
            return typeof (T);
        }

        public static Type ConstrainStruct<T>(T value)
            where T : struct
        {
            return typeof (T);
        }

        public static Type ConstrainConstructor<T>(T value)
            where T : new()
        {
            return typeof (T);
        }

        public class Tester<T>
        {
            public static Type SameType(T a, T b)
            {
                Console.WriteLine("Selected T = " + typeof (T));
                return typeof (T);
            }

            public static Type Method2<U>(T a, U b)
            {
                Console.WriteLine("Selected T = " + typeof (T));
                Console.WriteLine("Selected U = " + typeof (U));
                return typeof (U);
            }

            public static Type Method1<U>(U a)
                where U : IList<T>
            {
                Console.WriteLine("Selected T = " + typeof (T));
                Console.WriteLine("Selected U = " + typeof (U));
                return typeof (U);
            }
        }

        public class Tester2<T> : Tester<T>
        {
        }

        public enum TestEnum
        {
            A,
            B
        };

        [Fact]
        public void ConstraintTest()
        {
            MethodInfo method;
            method = typeof (TypeInferenceTests).GetMethod("CanSample");
            Type[] args1 = new Type[] {typeof (VectorGaussian)};
            TestInferGenericParameters(method, args1, 2);
            VectorGaussian g = new VectorGaussian(1);
            VectorGaussian[] ga = {g};
            Assert.Equal(typeof (Vector), (Type) Invoker.Invoke(method, null, ga));

            method = typeof (TypeInferenceTests).GetMethod("SampleInto");
            Type[] args2 = new Type[] {typeof (VectorGaussian), typeof (VectorGaussian)};
            TestInferGenericParameters(method, args2, 2);

            method = typeof (TypeInferenceTests).GetMethod("CanSample");
            Type[] args3 = new Type[] {typeof (double)};
            TestInferGenericParameters(method, args3, 0);
        }

        public static Type CanSample<DistributionType, DomainType>(DistributionType sourceDist)
            where DistributionType : Sampleable<DomainType>
        {
            Console.WriteLine("Selected T = " + typeof (DomainType));
            return typeof (DomainType);
        }

        public static Type SampleInto<DistributionType, DomainType>(DistributionType sourceDist, HasPoint<DomainType> targetDist)
            where DistributionType : Sampleable<DomainType>
        {
            Console.WriteLine("Selected T = " + typeof (DomainType));
            return typeof (DomainType);
        }

        private static void TestInferGenericParameters(MethodInfo method, Type[] args, int minTrueCount)
        {
            Console.WriteLine("Inferring {0} from {1}:", method, Invoker.PrintTypes(args));
            IList<Exception> lastError = new List<Exception>();
            IEnumerator<Binding> iter = Binding.InferGenericParameters(method, args, ConversionOptions.AllConversions, lastError);
            int count = 0;
            while (iter.MoveNext())
            {
                Console.WriteLine(iter.Current);
                count++;
            }
            if (count < minTrueCount)
            {
                if (lastError.Count > 0) throw lastError[0];
                else throw new Exception(String.Format("Only got {0} matches instead of {1}", count, minTrueCount));
            }
            if (minTrueCount >= 0) Assert.True(count >= minTrueCount); // count may be greater on newer runtimes
            if (minTrueCount == 0)
            {
                Console.WriteLine("Correctly threw exceptions: ");
                Console.WriteLine(StringUtil.ToString(lastError));
            }
        }

        private static void TestInferGenericParameters(Type formal, Type actual, MethodInfo method, int minTrueCount)
        {
            Console.WriteLine("Inferring {0} from {1}:", formal, actual);
            Binding binding = new Binding(method);
            IList<Exception> lastError = new List<Exception>();
            IEnumerator<Binding> iter = Binding.InferGenericParameters(formal, actual, binding, lastError, 0, true, ConversionOptions.AllConversions);
            int count = 0;
            while (iter.MoveNext())
            {
                Console.WriteLine(iter.Current);
                count++;
            }
            if (minTrueCount >= 0) Assert.True(count >= minTrueCount); // count may be greater on newer runtimes
            if (minTrueCount == 0)
            {
                Console.WriteLine("Correctly threw exceptions: ");
                Console.WriteLine(StringUtil.ToString(lastError));
            }
        }

        [Fact]
        public void InvokeStaticTest()
        {
            Type t = (Type) Invoker.InvokeStatic(typeof (Tester2<bool>), "Method1", new bool[1]);
            Assert.True(t.Equals(typeof (bool[])));
        }

        [Fact]
        public void InvokeMemberTest()
        {
            Tester t = new Tester();
            // field
            Invoker.InvokeMember(typeof (Tester), "intField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetField, t, 3);
            Assert.Equal(3, (int) Invoker.InvokeMember(typeof (Tester), "intField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetField, t));
            int[] array = new int[] {1, 2, 3};
            Invoker.InvokeMember(typeof (Tester), "arrayField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetField, t, array);
            Assert.Equal(3, (int) Invoker.InvokeMember(typeof (Tester), "arrayField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetField, t, 2));
            Invoker.InvokeMember(typeof (Tester), "arrayField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetField, t, 2, 4);
            Assert.Equal(4, (int) Invoker.InvokeMember(typeof (Tester), "arrayField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetField, t, 2));

            // property
            Invoker.InvokeMember(typeof (Tester), "intProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, t, 13);
            Assert.Equal(13, (int) Invoker.InvokeMember(typeof (Tester), "intProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, t));
            array = new int[] {1, 2, 3};
            Invoker.InvokeMember(typeof (Tester), "arrayProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, t, array);
            Assert.Equal(3, (int) Invoker.InvokeMember(typeof (Tester), "arrayProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, t, 2));
            Invoker.InvokeMember(typeof (Tester), "arrayProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, t, 2, 4);
            Assert.Equal(4, (int) Invoker.InvokeMember(typeof (Tester), "arrayProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, t, 2));

            // overloaded indexer property
            Assert.Equal(1, (int) Invoker.InvokeMember(typeof (Tester), "One", BindingFlags.Public | BindingFlags.Instance | BindingFlags.InvokeMethod, t));
            Invoker.InvokeMember(typeof (Tester), "Item", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, t, 2, 5);
            Assert.Equal(5, (int) Invoker.InvokeMember(typeof (Tester), "Item", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, t, 2));
            Invoker.InvokeMember(typeof (Tester), "Item", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, t, 2.0, 5);
            Assert.Equal(20, (int) Invoker.InvokeMember(typeof (Tester), "Item", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, t, 2.0));

            // field which is an indexed object
            Tester t2 = new Tester();
            Invoker.InvokeMember(typeof (Tester), "objectField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetField, t2, t);
            Assert.Equal(2, (int) Invoker.InvokeMember(typeof (Tester), "objectField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetField, t2, 1));
            Invoker.InvokeMember(typeof (Tester), "objectField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetField, t2, 2, 7);
            Assert.Equal(7, (int) Invoker.InvokeMember(typeof (Tester), "objectField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetField, t2, 2));

            // property which is an indexed object
            t2 = new Tester();
            Invoker.InvokeMember(typeof (Tester), "objectProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, t2, t);
            Assert.Equal(2, (int) Invoker.InvokeMember(typeof (Tester), "objectProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, t2, 1));
            Invoker.InvokeMember(typeof (Tester), "objectProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, t2, 2, 8);
            Assert.Equal(8, (int) Invoker.InvokeMember(typeof (Tester), "objectProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, t2, 2));

            // operator
            Assert.Equal(5, (int) Invoker.InvokeMember(typeof(int), "op_Addition", BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod, 2, 3));
            Assert.True((bool)Invoker.InvokeMember(typeof(int), "op_Equality", BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod, 2, 2));

            // delegate field/property
            Tester.TestDelegate d = t.One;
            Invoker.InvokeMember(typeof (Tester), "delegateField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetField, t, d);
            Assert.Equal(1,
                            (int) Invoker.InvokeMember(typeof (Tester), "delegateField", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetField, t, new object[0]));
            d = t.Two;
            Invoker.InvokeMember(typeof (Tester), "delegateProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, t, d);
            Assert.Equal(2,
                            (int)
                            Invoker.InvokeMember(typeof (Tester), "delegateProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, t, new object[0]));
        }

        [Fact]
        public void DelegateGroupTest()
        {
            Tester t = new Tester();
            // overloaded instance method with a target
            DelegateGroup dg = (DelegateGroup) Invoker.InvokeMember(typeof (Tester), "Two", BindingFlags.Public | BindingFlags.Instance | BindingFlags.InvokeMethod, t, null);
            Assert.Equal(2, (int) dg.DynamicInvoke());
            // assign the group to a property (this will pick one method out of the group)
            Invoker.InvokeMember(typeof (Tester), "delegateProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, t, dg);
            Assert.Equal(2,
                            (int)
                            Invoker.InvokeMember(typeof (Tester), "delegateProperty", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, t, new object[0]));
            // instance method with no target
            dg = (DelegateGroup) Invoker.InvokeMember(typeof (Tester), "One", BindingFlags.Public | BindingFlags.Instance | BindingFlags.InvokeMethod, null, null);
            Assert.Equal(1, (int) dg.DynamicInvoke(t));
            Converter<Tester, int> cti = (Converter<Tester, int>) dg.GetDelegate(typeof (Converter<Tester, int>));
            Assert.Equal(1, cti(t));
            // overloaded instance method with no target
            dg = (DelegateGroup) Invoker.InvokeMember(typeof (Tester), "Two", BindingFlags.Public | BindingFlags.Instance | BindingFlags.InvokeMethod, null, null);
            Assert.Equal(2, (int) dg.DynamicInvoke(t));
            cti = (Converter<Tester, int>) dg.GetDelegate(typeof (Converter<Tester, int>));
            Assert.Equal(2, cti(t));
            // static method with no target
            dg = (DelegateGroup) Invoker.InvokeMember(typeof (Tester), "Three", BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod, null, null);
            Assert.Equal(3, (int) dg.DynamicInvoke());
        }

        public delegate object WeaklyTypedFunction(object[] args);

        public delegate void WeaklyTypedAction(object[] args);

        [Fact]
        public void ConvertDelegateTest()
        {
            Action action = (Action) Conversion.ConvertDelegate(typeof (Action), new WeaklyTypedAction(Tester.WeaklyTypedAction));
            action();
            Action<double> doubleAction = (Action<double>) Conversion.ConvertDelegate(typeof (Action<double>),
                                                                                      new WeaklyTypedAction(Tester.WeaklyTypedAction));
            doubleAction(1.1);
            WeaklyTypedAction wtaction = (WeaklyTypedAction) Delegate.CreateDelegate(typeof (WeaklyTypedAction),
                                                                                     "foo", typeof (Tester).GetMethod("WeaklyTypedActionWithTarget"));
            wtaction(new object[0]);
            doubleAction = (Action<double>) Conversion.ConvertDelegate(typeof (Action<double>), wtaction);
            doubleAction(2.2);
            // changing return type to void
            doubleAction = (Action<double>) Conversion.ConvertDelegate(typeof (Action<double>),
                                                                       new WeaklyTypedFunction(Tester.WeaklyTypedMethod));
            doubleAction(3.3);
            Converter<double, double> df = (Converter<double, double>) Conversion.ConvertDelegate(typeof (Converter<double, double>),
                                                                                                  new WeaklyTypedFunction(Tester.WeaklyTypedMethod));
            Assert.Equal(7.7, df(7.7));
            WeaklyTypedFunction wtfun = (WeaklyTypedFunction) Delegate.CreateDelegate(typeof (WeaklyTypedFunction),
                                                                                      4.4, typeof (Tester).GetMethod("WeaklyTypedMethodWithTarget"));
            df = (Converter<double, double>) Conversion.ConvertDelegate(typeof (Converter<double, double>), wtfun);
            Assert.Equal(4.4, df(7.7));

            Converter<double[], double[]> daf = (Converter<double[], double[]>) Conversion.ConvertDelegate(typeof (Converter<double[], double[]>),
                                                                                                           new WeaklyTypedFunction(Tester.WeaklyTypedMethod));
            double[] da = daf(new double[] {1.1, 2.2});
            TestUtils.PrintCollection(da);
        }

        public class Tester
        {
          public int intField;
          public int[] arrayField;

          public int intProperty
          {
            get
            {
              return intField;
            }
            set
            {
              intField = value;
            }
          }

          public int[] arrayProperty
          {
            get
            {
              return arrayField;
            }
            set
            {
              arrayField = value;
            }
          }

          public int this[int index]
          {
            get
            {
              return arrayField[index];
            }
            set
            {
              arrayField[index] = value;
            }
          }

          public int this[double index]
          {
            get
            {
              return arrayField[(int)index] * 2;
            }
            set
            {
              arrayField[(int)index] = value * 2;
            }
          }

          public Tester objectField;

          public Tester objectProperty
          {
            get
            {
              return objectField;
            }
            set
            {
              objectField = value;
            }
          }

          public static int[] staticArrayField;

          public int One()
          {
            return 1;
          }

          public int Two(int x)
          {
            return x;
          }

          public int Two()
          {
            return 2;
          }

          public static int Three()
          {
            return 3;
          }

          public delegate int TestDelegate();

          public TestDelegate delegateField;

          public TestDelegate delegateProperty
          {
            get
            {
              return delegateField;
            }
            set
            {
              delegateField = value;
            }
          }

          public Tester()
          {
            delegateField = One;
          }

          public static object WeaklyTypedMethodWithTarget(object target, object[] args)
          {
            WeaklyTypedActionWithTarget(target, args);
            return target;
          }

          public static object WeaklyTypedMethod(object[] args)
          {
            WeaklyTypedAction(args);
            if (args.Length == 0)
              return 1.1;
            return args[0];
          }

          public static void WeaklyTypedActionWithTarget(object target, object[] args)
          {
            Console.WriteLine("target = {0}", target);
            WeaklyTypedAction(args);
          }

          public static void WeaklyTypedAction(object[] args)
          {
            if (args.Length == 0)
              Console.WriteLine("no args");
            for (int i = 0; i < args.Length; i++)
            {
              Console.WriteLine("args[{0}] = {1}", i, args[i]);
            }
          }

          public object Invoke(object target, object[] args)
          {
            return WeaklyTypedMethodWithTarget(target, args);
          }

          public void Action(object target, object[] args)
          {
            WeaklyTypedActionWithTarget(target, args);
          }

          public static explicit operator int(Tester t)
          {
            return t.intField;
          }

          public static explicit operator int[](Tester t)
          {
            return t.arrayField;
          }

            public static explicit operator Tester(int x)
            {
                var tester = new Tester
                {
                    intField = x
                };
                return tester;
            }

            public static explicit operator Tester(int[] x)
            {
                var tester = new Tester
                {
                    arrayField = x
                };
                return tester;
            }

            public static implicit operator Tester(ImplicitlyConvertibleToTesterCastDefinedOnTester _) => new Tester();
        }

        public class ImplicitlyConvertibleToTesterDefinesCast
        {
            public static implicit operator Tester(ImplicitlyConvertibleToTesterDefinesCast _) => new Tester();
        }

        public class ImplicitlyConvertibleToTesterCastDefinedOnTester { }

        /// <summary>
        /// Help class of dynamically invoking methods.
        /// </summary>
        public class ListMethods
        {
            public static void StringListMethod(IList<string> list)
            {
            }

            public static void ObjectListMethod(IList<object> list)
            {
            }

            public static void GenericListMethod<T>(IList<T> list, T value)
            {
                Console.WriteLine("GenericListMethod T = {0}", typeof(T));
            }
        }
    }
}
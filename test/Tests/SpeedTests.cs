// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using System.Diagnostics;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Assert = Xunit.Assert;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using BernoulliArray = DistributionStructArray<Bernoulli, bool>;

    
    public class SpeedTests
    {
        /// <summary>
        /// Test the GMM from the Swift paper:
        /// http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-12.pdf
        /// </summary>
        public void SwiftGmmTest()
        {
            Range cluster = new Range(4).Named("cluster");
            var center = Variable.Array<double>(cluster).Named("center");
            center[cluster] = Variable.GaussianFromMeanAndVariance(0, 50).ForEach(cluster);
            var precision = Variable.Array<double>(cluster).Named("precision");
            precision[cluster] = Variable.GammaFromShapeAndRate(1, 1).ForEach(cluster);
            Range item = new Range(100).Named("item");
            var assign = Variable.Array<int>(item).Named("assign");
            assign[item] = Variable.DiscreteUniform(cluster).ForEach(item);
            var x = Variable.Array<double>(item).Named("x");
            using (Variable.ForEach(item))
            {
                var c = assign[item];
                using (Variable.Switch(c))
                {
                    x[item] = Variable.GaussianFromMeanAndPrecision(center[c], precision[c]);
                }
            }

            // generate data
            var centerTrue = Util.ArrayInit(cluster.SizeAsInt, i => Gaussian.Sample(0, 1.0 / 50));
            var precisionTrue = Util.ArrayInit(cluster.SizeAsInt, i => Gamma.Sample(1, 1));
            var assignTrue = Util.ArrayInit(item.SizeAsInt, i => Rand.Int(cluster.SizeAsInt));
            var data = Util.ArrayInit(item.SizeAsInt, i => Gaussian.Sample(centerTrue[assignTrue[i]], precisionTrue[assignTrue[i]]));
            x.ObservedValue = data;

            var alg = new GibbsSampling();
            alg.BurnIn = 0;
            InferenceEngine engine = new InferenceEngine(alg);
            engine.NumberOfIterations = 10000;
            engine.ShowTimings = true;
            engine.ShowProgress = false;
            //engine.Compiler.UseExistingSourceFiles = true;
            engine.Compiler.IncludeDebugInformation = false;
            //engine.Compiler.UseParallelForLoops = true;
            Console.WriteLine(engine.Infer(center));
        }

        public static void CompilationTest()
        {
            for (int i = 0; i < 1000; i++)
            {
                Variable<bool> x = Variable.Bernoulli(0.1).Named("x");
                InferenceEngine engine = new InferenceEngine();
                engine.Compiler.ShowProgress = true;
                Bernoulli xActual = engine.Infer<Bernoulli>(x);
                Console.WriteLine("Number of loaded assemblies: {0}", AppDomain.CurrentDomain.GetAssemblies().Length);
            }
        }

        public static void CompilationTestMsl()
        {
            IDeclarationProvider declProvider;
            var modelMethods = FindModelMethods(out declProvider);

            Console.WriteLine("Compiling {0} model methods...", modelMethods.Length);
            var sw = new Stopwatch();

            sw.Restart();
            RunTests(modelMethods, declProvider, true);
            sw.Stop();
            Console.WriteLine("Compiled {0} models in {1}ms (average time = {2}ms)", modelMethods.Length, sw.ElapsedMilliseconds, sw.ElapsedMilliseconds / modelMethods.Length);

            sw.Restart();
            RunTests(modelMethods, declProvider, false);
            sw.Stop();
            Console.WriteLine("Compiled {0} models (without compiling generated code) in {1}ms (average time = {2}ms)", modelMethods.Length, sw.ElapsedMilliseconds, sw.ElapsedMilliseconds / modelMethods.Length);
        }

        private static MethodInfo[] FindModelMethods(out IDeclarationProvider declProvider)
        {
            var mslTests = TestUtils.FindTestMethods(Assembly.GetExecutingAssembly(), "CsoftModel");
            var types = mslTests.Select(mi => mi.DeclaringType).Distinct().ToArray();
            Console.WriteLine("Getting type decls (takes a few seconds)...");
            var typeLookup = types.ToDictionary(t => t, t => RoslynDeclarationProvider.Instance.GetTypeDeclaration(t, true));
            var methodLookup = typeLookup.Values.SelectMany(td => td.Methods).ToDictionary(md => md.MethodInfo, md => md);
            declProvider = new CachedDeclarationProvider() {Cache = typeLookup};
            return mslTests.SelectMany(mi => FindModels(methodLookup[mi])).ToArray();
        }

        private static void RunTests(MethodInfo[] modelMethods, IDeclarationProvider declProvider, bool compileGeneratedCode)
        {
            foreach (var m in modelMethods)
            {
                InferenceEngine engine = new InferenceEngine();
                engine.Compiler.DeclarationProvider = declProvider;
                engine.ShowTimings = false;
                engine.ShowProgress = false;
                //engine.Compiler.ShowProgress = true;
                try
                {
                    if (compileGeneratedCode)
                    {
                        engine.Compiler.CompileWithoutParams(m);
                    }
                    else
                    {
                        engine.Compiler.GetTransformedDeclaration(declProvider.GetTypeDeclaration(m.DeclaringType, true), m, new AttributeRegistry<object, ICompilerAttribute>(true));
                    }
                }
                catch { }
            }
        }

        private class CachedDeclarationProvider : IDeclarationProvider
        {
            public Dictionary<Type, ITypeDeclaration> Cache; 
            public ITypeDeclaration GetTypeDeclaration(Type t, bool translate)
            {
                return Cache[t];
            }
        }

        private static IEnumerable<MethodInfo> FindModels(IMethodDeclaration methodDeclaration)
        {
            foreach (var stmt in methodDeclaration.Body.Statements)
            {
                var exprStmt = stmt as IExpressionStatement;
                if (exprStmt != null)
                {
                    IExpression expr = exprStmt.Expression;
                    if (expr is IAssignExpression)
                    {
                        expr = ((IAssignExpression) expr).Expression;
                    }
                    if (expr is IMethodInvokeExpression)
                    {
                        var invoke = (IMethodInvokeExpression) expr;
                        if (invoke.Arguments.Count > 1)
                        {
                            var arg = invoke.Arguments[0] as IDelegateCreateExpression;
                            if (arg != null)
                            {
                                yield return (MethodInfo)arg.Method.MethodInfo;
                            }
                        }
                    }
                }
            }
        } 

        public static void SetToFunction<T>(T[] result, Func<T, T, T> func, T[] a, T[] b)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = func(a[i], b[i]);
            }
        }

        public void SetToFunctionInstance<T>(T[] result, Func<T, T, T> func, T[] a, T[] b)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = func(a[i], b[i]);
            }
        }

        public static void SetToSum(double[] result, double[] a, double[] b)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        public static void SetToSum2(double[] result, double[] a, double[] b)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = a[i] + b[i] + a[i] + b[i] + a[i] + b[i];
            }
        }

        public void SparseDirichletTest()
        {
            int size = 2000;
            Dirichlet probs = new Dirichlet(Vector.Constant(size, 1.0));
            Discrete result = new Discrete(Vector.Constant(size, 1.0));

            Stopwatch watch = new Stopwatch();
            int count = 10000;
            watch.Start();
            for (int i = 0; i < count; i++)
                DiscreteFromDirichletOp.SampleAverageConditional(probs, result);
            Console.WriteLine("Sparse Dirichlet test: " + watch.ElapsedMilliseconds);
        }

        public void SetToSumTest()
        {
            int size = 10000;
            double[] a = Util.ArrayInit(size, i => 1.0 + i);
            double[] b = Util.ArrayInit(size, i => 2.0 + i);
            double[] c = Util.ArrayInit(size, i => 3.0 + i);
            Stopwatch watch = new Stopwatch();
            int count = 1000;
            for (int iter = 0; iter < 3; iter++)
            {
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    SetToSum(c, a, b);
                }
                Console.WriteLine("SetToSum: " + watch.ElapsedMilliseconds);
                watch.Reset();
                if (false)
                {
                    watch.Start();
                    for (int i = 0; i < count; i++)
                    {
                        SetToSum2(c, a, b);
                    }
                    Console.WriteLine("SetToSum2: " + watch.ElapsedMilliseconds);
                    watch.Reset();
                }
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    SetToFunctionInstance(c, Factor.Plus, a, b);
                }
                Console.WriteLine("SetToFunctionInstance(Plus): " + watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    SetToFunction(c, Factor.Plus, a, b);
                }
                Console.WriteLine("SetToFunction(Plus): " + watch.ElapsedMilliseconds);
                watch.Reset();
            }
        }

        public static T Reduce<T>(T initial, Func<T, T, T> func, T[] a)
        {
            for (int i = 0; i < a.Length; i++)
            {
                initial = func(initial, a[i]);
            }
            return initial;
        }

        // same as Reduce but an instance method.
        public T ReduceInstance<T>(T initial, Func<T, T, T> func, T[] a)
        {
            for (int i = 0; i < a.Length; i++)
            {
                initial = func(initial, a[i]);
            }
            return initial;
        }

        public static double Sum(double[] a)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i];
            }
            return sum;
        }

        public static double Sum2(double[] a)
        {
            double sum = 0.0;
            for (int i = a.Length - 1; i > 0; i--)
            {
                sum += a[i];
            }
            return sum;
        }

        public static double Sum2(Array1d<double> a)
        {
            double sum = 0.0;
            for (int i = a.Count - 1; i > 0; i--)
            {
                sum += a[i];
            }
            return sum;
        }

        public static double Sum(Array1d<double> a)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Count; i++)
            {
                sum += a[i];
            }
            return sum;
        }

        public static double Sum(Vector a)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Count; i++)
            {
                sum += a[i];
            }
            return sum;
        }

        public static double Sum(DenseVector a)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Count; i++)
            {
                sum += a[i];
            }
            return sum;
        }

        public void SumTest()
        {
            int size = 10000;
            double[] a = Util.ArrayInit(size, i => 1.0 + i);
            Array1d<double> a1d = new Array1d<double>(Util.ArrayInit(size, i => 1.0 + i));
            Vector av = Vector.FromArray(a);
            DenseVector adv = DenseVector.FromArray(a);
            Stopwatch watch = new Stopwatch();
            int count = 1000;
            for (int iter = 0; iter < 3; iter++)
            {
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    Sum(a);
                }
                Console.WriteLine("Sum: " + watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    Sum2(a);
                }
                Console.WriteLine("Sum2: " + watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    Sum(a1d);
                }
                Console.WriteLine("Sum(Array1d): " + watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    Sum2(a1d);
                }
                Console.WriteLine("Sum2(Array1d): " + watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    ReduceInstance(0.0, Factor.Plus, a);
                }
                Console.WriteLine("ReduceInstance(Plus): " + watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    Reduce(0.0, Factor.Plus, a);
                }
                Console.WriteLine("Reduce(Plus): " + watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    Sum(av);
                }
                Console.WriteLine("Sum(Vector): " + watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    Sum(adv);
                }
                Console.WriteLine("Sum(DenseVector): " + watch.ElapsedMilliseconds);
                watch.Reset();
            }
        }

        public struct Array1d<T> : IList<T>
        {
            public T[] Array;

            public Array1d(T[] array)
            {
                this.Array = array;
            }

            public int IndexOf(T item)
            {
                throw new Exception("The method or operation is not implemented.");
            }

            public void Insert(int index, T item)
            {
                throw new Exception("The method or operation is not implemented.");
            }

            public void RemoveAt(int index)
            {
                throw new Exception("The method or operation is not implemented.");
            }

            public T this[int index]
            {
                get { return Array[index]; }
                set { Array[index] = value; }
            }

            public void Add(T item)
            {
                throw new Exception("The method or operation is not implemented.");
            }

            public void Clear()
            {
                throw new Exception("The method or operation is not implemented.");
            }

            public bool Contains(T item)
            {
                throw new Exception("The method or operation is not implemented.");
            }

            public void CopyTo(T[] array, int arrayIndex)
            {
                throw new Exception("The method or operation is not implemented.");
            }

            public int Count
            {
                //[System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
                get { return Array.Length; }
            }

            public bool IsReadOnly
            {
                get { throw new Exception("The method or operation is not implemented."); }
            }

            public bool Remove(T item)
            {
                throw new Exception("The method or operation is not implemented.");
            }

            public IEnumerator<T> GetEnumerator()
            {
                throw new Exception("The method or operation is not implemented.");
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                throw new Exception("The method or operation is not implemented.");
            }
        }

        public static void IListTest2()
        {
            int size = 100;
            Bernoulli[] array = new Bernoulli[size];
            DistributionStructArray<Bernoulli, bool> list = new DistributionStructArray<Bernoulli, bool>(size);
            Bernoulli result = new Bernoulli();
            Stopwatch watch = new Stopwatch();
            int count = 100000;
            for (int iter = 0; iter < 3; iter++)
            {
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    Distribution.SetToProductOfAll(result, array);
                }
                Console.WriteLine("MultiplyAll(array): " + watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    Distribution.SetToProductOfAll(result, list);
                }
                Console.WriteLine("MultiplyAll(list): " + watch.ElapsedMilliseconds);
                watch.Reset();
            }
        }

        public static void IListTest()
        {
            int[] array = new int[1];
            array[0] = 3;
            List<int> list = new List<int>();
            list.Add(7);
            Stopwatch watch = new Stopwatch();
            int count = 10000000;
            for (int iter = 0; iter < 3; iter++)
            {
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    OverloadedFunction(array);
                }
                Console.WriteLine("OverloadedFunction(array): {0}ms", watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    OverloadedFunction(list);
                }
                Console.WriteLine("OverloadedFunction(list): {0}ms", watch.ElapsedMilliseconds);
                watch.Reset();
            }
            for (int iter = 0; iter < 3; iter++)
            {
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    IListFunction(array);
                }
                Console.WriteLine("IListFunction(array): {0}ms", watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    IListFunction(list);
                }
                Console.WriteLine("IListFunction(list): {0}ms", watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                IList<int> ilist = new Array1d<int>(array);
                for (int i = 0; i < count; i++)
                {
                    IListFunction(ilist);
                    //IListFunction(new Array1d<int>(array));
                }
                Console.WriteLine("IListFunction(Array1d): {0}ms", watch.ElapsedMilliseconds);
                watch.Reset();
            }
            for (int iter = 0; iter < 3; iter++)
            {
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    GenericIListFunction(array);
                }
                Console.WriteLine("GenericIListFunction(array): {0}ms", watch.ElapsedMilliseconds);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < count; i++)
                {
                    GenericIListFunction(list);
                }
                Console.WriteLine("GenericIListFunction(list): {0}ms", watch.ElapsedMilliseconds);
                watch.Reset();
            }
        }

        public static int IListFunction(IList<int> list)
        {
            return list.Count;
        }

        public static int GenericIListFunction<intList>(intList list)
            where intList : IList<int>
        {
            return list.Count;
        }

        public static int OverloadedFunction(int[] array)
        {
            return array.Length;
        }

        public static int OverloadedFunction(IList<int> list)
        {
            return list.Count;
        }

        public static void ArrayCreateTest()
        {
            Stopwatch watch = new Stopwatch();
            int count = 1000000;
            int[] array = null;
            watch.Start();
            for (int i = 0; i < count; i++)
            {
                array = new int[1];
            }
            Console.WriteLine("new int[]: " + watch.ElapsedMilliseconds);
            watch.Reset();
            watch.Start();
            for (int i = 0; i < count; i++)
            {
                Array.CreateInstance(typeof (int), 1);
            }
            Console.WriteLine("Array.CreateInstance: " + watch.ElapsedMilliseconds);
            watch.Reset();
            watch.Start();
            for (int i = 0; i < count; i++)
            {
                Util.CreateArray<int>(1);
            }
            Console.WriteLine("Util.CreateArray: " + watch.ElapsedMilliseconds);
            watch.Reset();
        }

        public void ReplicateDefTest()
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            BernoulliArray[][] ind_cases_B;
            ind_cases_B = new BernoulliArray[0x1900][];
            for (int index1 = 0; index1 < 0x1900; index1++)
            {
                ind_cases_B[index1] = new BernoulliArray[10];
                for (int index0 = 0; index0 < 10; index0++)
                {
                    ind_cases_B[index1][index0] = new BernoulliArray(1);
                }
            }
            BernoulliArray[][][] ind_cases_uses_B;
            ind_cases_uses_B = new BernoulliArray[0x1900][][];
            for (int index1 = 0; index1 < 0x1900; index1++)
            {
                ind_cases_uses_B[index1] = new BernoulliArray[10][];
                for (int index0 = 0; index0 < 10; index0++)
                {
                    ind_cases_uses_B[index1][index0] = new BernoulliArray[3];
                    for (int _ind = 0; _ind < 3; _ind++)
                    {
                        //Bernoulli[] t = new Bernoulli[1];
                        ind_cases_uses_B[index1][index0][_ind] = new BernoulliArray(new Bernoulli(), 1);
                    }
                }
            }
            watch.Stop();
            Console.WriteLine("Create arrays: " + watch.ElapsedMilliseconds);
            watch.Reset();
            watch.Start();
#if false
            DistributionArray<DistributionArray<Bernoulli>> uses = new DistributionArray<DistributionArray<Bernoulli>>(1);
            for (int i = 0; i < uses.Count; i++) {
                uses[i] = new BernoulliArray(1);
            }
#else
            BernoulliArray[] uses = new BernoulliArray[1];
            for (int i = 0; i < uses.Length; i++)
            {
                uses[i] = new BernoulliArray(1);
            }
#endif
            if (false)
            {
                for (int i = 0; i < 100000; i++)
                {
                    Distribution.SetToProductOfAll(new Bernoulli(), new Bernoulli[1]);
                }
            }
            if (false)
            {
                for (int i = 0; i < 100000; i++)
                {
                    Distribution.SetToProductOfAll(new Bernoulli(), new BernoulliArray(1));
                }
            }
            if (false)
            {
                for (int index1 = 0; index1 < 0x1900; index1++)
                {
                    for (int index0 = 0; index0 < 10; index0++)
                    {
                        //DistributionArray<Bernoulli> result = ind_cases_uses_B[index1][index0][0];
                        //result.SetTo(uses[0]);
                        //ind_cases_uses_B[index1][index0][0] = result;
                        //ind_cases_uses_B[index1][index0][0] = Distribution.MultiplyAll<DistributionArray<Bernoulli>>(uses, result);
                        BernoulliArray result = new BernoulliArray(1);
                        Distribution.SetToProductOfAll(result, uses);
                    }
                }
            }
            if (false)
            {
                for (int index1 = 0; index1 < 0x1900; index1++)
                {
                    for (int index0 = 0; index0 < 10; index0++)
                    {
                        ind_cases_uses_B[index1][index0][0] = ReplicateOp.DefAverageLogarithm(uses, ind_cases_uses_B[index1][index0][0]);
                    }
                }
            }
            if (false)
            {
                for (int index1 = 0; index1 < 0x1900; index1++)
                {
                    for (int index0 = 0; index0 < 10; index0++)
                    {
                        ind_cases_uses_B[index1][index0][1] = ReplicateOp.DefAverageLogarithm(uses, ind_cases_uses_B[index1][index0][1]);
                    }
                }
            }
            watch.Stop();
            Console.WriteLine("Initialize: " + watch.ElapsedMilliseconds);
            watch.Reset();
            watch.Start();
            for (int index1 = 0; index1 < 0x1900; index1++)
            {
                for (int index0 = 0; index0 < 10; index0++)
                {
                    //Distribution.MultiplyAll(ind_cases_uses_B[index1][index0], ind_cases_B[index1][index0]);
                    ind_cases_B[index1][index0] = ReplicateOp.DefAverageLogarithm(ind_cases_uses_B[index1][index0], ind_cases_B[index1][index0]);
                }
            }
            watch.Stop();
            Console.WriteLine("ML.Probabilistic: " + watch.ElapsedMilliseconds);
        }

        // test the impact of subnormal/denormalized numbers on linear algebra speeds
        public void SubnormalTest()
        {
            Stopwatch watch = new Stopwatch();
            PositiveDefiniteMatrix A = new PositiveDefiniteMatrix(256, 256);
            A.SetAllElementsTo(1e-100);
            for (int i = 0; i < A.Rows; i++)
            {
                A[i, i] = 1;
            }
            LowerTriangularMatrix U = new LowerTriangularMatrix(A.Rows, A.Cols);
            watch.Start();
            U.SetToCholesky(A);
            watch.Stop();
            Console.WriteLine("normal: " + watch.ElapsedMilliseconds);
            A.SetAllElementsTo(1e-310);
            for (int i = 0; i < A.Rows; i++)
            {
                A[i, i] = 1;
            }
            watch.Reset();
            watch.Start();
            U.SetToCholesky(A);
            watch.Stop();
            Console.WriteLine("denormal: " + watch.ElapsedMilliseconds);
        }

        /* Times on MSRC-MINKA2, .NET4 exec
         * without LAPACK, x64, N=500, blocked OuterTranspose:
Sampling A: 8
A T: 4
Calculating A A^T (slow): 615
Calculating A A^T (fast): 296
Cholesky: 50
Invert triangular: 84
Outer: 294
Outer transpose: 313 (600 if not blocked)
Inverse: 442
         * without LAPACK, x64, N=2000, blocked OuterTranspose:
Sampling A: 138
A T: 54
Calculating A A^T (slow): 67944
Calculating A A^T (fast): 20971
Cholesky: 3353
Invert triangular: 13388
Outer: 20631
Outer transpose: 20262
Inverse: 36842
         * with LAPACK, x64, N=2000:
Sampling A: 135
A T: 54
Calculating A A^T (slow): 1167
Calculating A A^T (fast): 852
Cholesky: 158
Invert triangular: 550
Outer: 834
Outer transpose: 803
Inverse: 613
         * The speedup is around 20x for performing the same operation with LAPACK.
         */

        public void CholeskySpeedTest()
        {
            Stopwatch watch = new Stopwatch();
            int N = 500*4;
            PositiveDefiniteMatrix A = new PositiveDefiniteMatrix(N, N);
            watch.Start();
            for (int i = 0; i < A.Cols; i++)
            {
                for (int j = 0; j < A.Rows; j++)
                {
                    A[i, j] = Rand.Double();
                }
            }
            watch.Stop();
            Console.WriteLine("Sampling A: " + watch.ElapsedMilliseconds);
            watch.Reset();
            watch.Start();
            var AT = A.Transpose();
            watch.Stop();
            Console.WriteLine("A T: " + watch.ElapsedMilliseconds);
            watch.Reset();
            watch.Start();
            var B = new PositiveDefiniteMatrix(A*AT);
            watch.Stop();
            Console.WriteLine("Calculating A A^T (slow): " + watch.ElapsedMilliseconds);
            watch.Reset();
            watch.Start();
            B.SetToOuter(A);
            watch.Stop();
            Console.WriteLine("Calculating A A^T (fast): " + watch.ElapsedMilliseconds);
            LowerTriangularMatrix L = new LowerTriangularMatrix(A.Rows, A.Cols);
            watch.Reset();
            watch.Start();
            // Cholesky is expected to be 6x faster than Outer due to performing fewer operations
            L.SetToCholesky(B);
            watch.Stop();
            Console.WriteLine("Cholesky: " + watch.ElapsedMilliseconds);
            watch.Reset();
            watch.Start();
            L.SetToInverse(L);
            watch.Stop();
            Console.WriteLine("Invert triangular: " + watch.ElapsedMilliseconds);
            var truth = (new PositiveDefiniteMatrix(A.Rows, A.Rows));
            watch.Reset();
            watch.Start();
            truth.SetToOuter(L);
            watch.Stop();
            Console.WriteLine("Outer: " + watch.ElapsedMilliseconds);
            watch.Reset();
            watch.Start();
            truth.SetToOuterTranspose(L);
            watch.Stop();
            Console.WriteLine("Outer transpose: " + watch.ElapsedMilliseconds);
            var C = new PositiveDefiniteMatrix(B);
            watch.Reset();
            watch.Start();
            C.SetToInverse(B);
            watch.Stop();
            Console.WriteLine("Inverse: " + watch.ElapsedMilliseconds);
        }

        public void LogisticSpecialFunctionsSpeed()
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            // old=3000ms
            // new=173ms
            double x;
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    x = MMath.LogisticGaussian(j, i/100.0);
                }
            }
            watch.Stop();
            Console.WriteLine("{0}ms", watch.ElapsedMilliseconds);
        }

        /// <summary>
        /// Tests a performance anomaly in .NET
        /// </summary>
        public void MemoryAllocationSlowdown()
        {
            int n = 15;
            Range featureRange = new Range(n);

            VariableArray<double> mean = Variable.Array<double>(featureRange).Named("mean");
            VariableArray<double> precision = Variable.Array<double>(featureRange).Named("precision");

            VariableArray<double> contrib = Variable.Array<double>(featureRange).Named("contrib");
            contrib[featureRange] = Variable.GaussianFromMeanAndPrecision(mean[featureRange], precision[featureRange]);

            Variable<double> sum = Variable.Sum(contrib) + Variable<double>.GaussianFromMeanAndVariance(0, 1).Named("tmp");

            Variable.ConstrainPositive(sum);

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            engine.ShowProgress = false;
            engine.ShowTimings = false;
            engine.Compiler.FreeMemory = false;
            //engine.ShowFactorGraph = true;
            engine.NumberOfIterations = 5;
            mean.ObservedValue = new double[n];
            precision.ObservedValue = new double[n];
            engine.OptimiseForVariables = new List<IVariable>() {contrib};
            var ca = engine.GetCompiledInferenceAlgorithm(contrib);

            Random ran = new Random();
            DateTime start = DateTime.Now, last = DateTime.Now;
            Dictionary<string, int> d = new Dictionary<string, int>();
            bool doAllocation = true;
            for (int i = 0; i < 100001; i++)
            {
                if (doAllocation)
                {
                    for (int j = 0; j < 5; j++)
                    {
                        byte[] ss = new byte[10];
                        ran.NextBytes(ss);
                        string cur = "";
                        for (int k = 0; k < 10; k++)
                            cur += (char) (ss[k]);
                        d[cur] = i;
                    }
                }

                if (i%10000 == 0)
                    Console.WriteLine(i);
                if (i%10000 == 0)
                {
                    Console.WriteLine("Seconds since the start {0}; seconds since last output {1}.", new TimeSpan(DateTime.Now.Ticks - start.Ticks).TotalSeconds,
                                      new TimeSpan(DateTime.Now.Ticks - last.Ticks).TotalSeconds);
                    last = DateTime.Now;
                }
                double[] mean1 = new double[n], precision1 = new double[n];
                for (int j = 0; j < mean1.Length; j++)
                {
                    mean1[j] = ran.NextDouble();
                    precision1[j] = ran.NextDouble();
                }
                mean.ObservedValue = mean1;
                precision.ObservedValue = precision1;
                ca.SetObservedValue(mean.Name, mean.ObservedValue);
                ca.SetObservedValue(precision.Name, precision.ObservedValue);

                //object g = engine.Infer(contrib);
                ca.Execute(engine.NumberOfIterations);
                object g = ca.Marginal(contrib.Name);
            }
        }

        /// <summary>
        /// Rolled Markov chain model
        /// </summary>
        private class MarkovChain
        {
            public Variable<int> NumNodes = Variable.New<int>().Named("NumNodes");
            public Variable<int> NumNodesMinus1 = Variable.New<int>().Named("NumNodesMinus1");
            public VariableArray<int> PrevIndices;
            public VariableArray<int> NextIndices;
            public VariableArray<int> Nodes;
            public Variable<Discrete>[] transitionProbs;
            public InferenceEngine engine = new InferenceEngine();
            public bool inferDone = false;

            public MarkovChain()
            {
                Range n = new Range(NumNodes).Named("n");
                Range n1 = new Range(NumNodesMinus1).Named("n1");
                Nodes = Variable.Array<int>(n).Named("node");
                Nodes[n] = Variable.DiscreteUniform(3).ForEach(n);
                PrevIndices = Variable.Array<int>(n1).Named("prevIndices");
                NextIndices = Variable.Array<int>(n1).Named("nextIndices");
                var prev = Variable.Subarray(Nodes, PrevIndices).Named("prev");
                var next = Variable.Subarray(Nodes, NextIndices).Named("next");
                transitionProbs = new Variable<Discrete>[3];
                for (int k = 0; k < transitionProbs.Length; k++)
                {
                    transitionProbs[k] = Variable.New<Discrete>().Named("transitionProbs" + k);
                }
                using (Variable.ForEach(n1))
                {
                    for (int k = 0; k < 3; k++)
                    {
                        using (Variable.Case(prev[n1], k))
                            Variable.ConstrainEqualRandom(next[n1], transitionProbs[k]);
                    }
                }
            }

            public void SetNumNodes(int numNodes)
            {
                NumNodes.ObservedValue = numNodes;
                int numNodes1 = numNodes - 1;
                NumNodesMinus1.ObservedValue = numNodes1;
                int[] prevIndices = new int[numNodes1];
                int[] nextIndices = new int[numNodes1];
                for (int i = 0; i < numNodes1; i++)
                {
                    prevIndices[i] = i;
                    nextIndices[i] = i + 1;
                }
                PrevIndices.ObservedValue = prevIndices;
                NextIndices.ObservedValue = nextIndices;
            }

            public Discrete[] Infer()
            {
                if (!inferDone)
                {
                    engine.OptimiseForVariables = new List<IVariable>() {Nodes};
                    inferDone = true;
                }
                return engine.Infer<Discrete[]>(Nodes);
            }
        }

        /// <summary>
        /// Unrolled Markov chain model
        /// </summary>
        public class MarkovChainUnrolled
        {
            public Variable<int>[] Nodes;
            public InferenceEngine engine = new InferenceEngine();
            private int NumNodes;
            public Variable<Discrete>[] transitionProbs;
            public bool inferDone = false;

            public MarkovChainUnrolled(int numNodes)
            {
                NumNodes = numNodes;
                Nodes = new Variable<int>[NumNodes];
                for (int n = 0; n < NumNodes; n++)
                    Nodes[n] = Variable.DiscreteUniform(3);
                transitionProbs = new Variable<Discrete>[3];
                for (int k = 0; k < transitionProbs.Length; k++)
                {
                    transitionProbs[k] = Variable.New<Discrete>().Named("transitionProbs" + k);
                }
                for (int n = 1; n < NumNodes; n++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        using (Variable.Case(Nodes[n - 1], k))
                            Variable.ConstrainEqualRandom(Nodes[n], transitionProbs[k]);
                    }
                }
            }

            public Discrete[] Infer()
            {
                if (!inferDone)
                {
                    engine.OptimiseForVariables = Nodes;
                    inferDone = true;
                }
                Discrete[] result = new Discrete[NumNodes];
                for (int n = 0; n < NumNodes; n++)
                    result[n] = engine.Infer<Discrete>(Nodes[n]);
                return result;
            }

            public void Compile()
            {
                engine.GetCompiledInferenceAlgorithm(Nodes);
            } 
        }

        // Compiler performance test (must run twice)
        public void MarkovChainSpeedTest()
        {
            int numNodes = 100;
            int numNodesToPrint = 5;

            // Rolled version
            var model = new MarkovChain();
            model.SetNumNodes(numNodes);
            model.engine.ModelName = "MarkovChainRolled";
            model.engine.ShowTimings = false;
            model.engine.ShowProgress = false;
            model.engine.Compiler.ShowProgress = true;

            // Unrolled version
            var modelUnrolled = new MarkovChainUnrolled(numNodes);
            modelUnrolled.engine.ModelName = "MarkovChainUnrolled";
            modelUnrolled.engine.ShowTimings = false;
            modelUnrolled.engine.ShowProgress = false;
            modelUnrolled.engine.Compiler.ShowProgress = true;
            for (int iter = 0; iter < 3; iter++)
            {
                Discrete[] transitionProbs = new Discrete[3];
                transitionProbs[0] = new Discrete(0.1, 0.2, 0.7);
                transitionProbs[1] = new Discrete(0.5, 0.2, 0.3);
                transitionProbs[2] = new Discrete(0.7, 0.1, 0.2);
                for (int k = 0; k < transitionProbs.Length; k++)
                {
                    model.transitionProbs[k].ObservedValue = transitionProbs[k];
                    modelUnrolled.transitionProbs[k].ObservedValue = transitionProbs[k];
                }

                // MSRC-MINKA-E640, Scheduler2, Release, unrolled, SchedulingTransform
                // n=40: 352ms
                // n=100: 2244ms
                // MSRC-MINKA2
                // n=40: 177ms
                Stopwatch watch = new Stopwatch();
                watch.Start();
                var post = model.Infer();
                watch.Stop();
                long timeRolled = watch.ElapsedMilliseconds;
                watch.Reset();
                watch.Start();
                var postUnrolled = modelUnrolled.Infer();
                watch.Stop();
                long timeUnrolled = watch.ElapsedMilliseconds;
                Console.WriteLine("rolled = {0}ms   unrolled = {1}ms", timeRolled, timeUnrolled);
                if (false)
                {
                    for (int i = 0; i < numNodesToPrint; i++)
                        Console.WriteLine("{0} {1}", post[i], postUnrolled[i]);
                }
                for (int i = 0; i < post.Length; i++)
                {
                    Assert.True(post[i].MaxDiff(postUnrolled[i]) < 1e-10);
                }
            }
        }

        // ModelBuilder test
        public void MarkovChainUnrolledTest2()
        {
            int nodeNum = 70;
            int statesN = 2;

            Range obs = new Range(1).Named("N");
            Range N = new Range(statesN);

            VariableArray<Dirichlet> transPrior = Variable.Array<Dirichlet>(N).Named("transPrior");
            VariableArray<Vector> transProbs = Variable.Array<Vector>(N).Named("transProbs");
            transProbs[N] = Variable<Vector>.Random<Dirichlet>(transPrior[N]);
            transProbs.SetValueRange(N);

            Variable<Vector> stateProbs = Variable.New<Vector>().Named("stateProbs");
            stateProbs.SetValueRange(N);

            VariableArray<int>[] nodes = new VariableArray<int>[nodeNum];
            for (int k = 0; k < nodeNum; k++)
            {
                VariableArray<int> node = Variable.Array<int>(obs).Named("node_" + k);
                node.SetValueRange(N);
                if (k == 0)
                {
                    node[obs] = Variable.Discrete(stateProbs).ForEach(obs);
                }
                else
                {
                    using (Variable.ForEach(obs))
                    using (Variable.Switch(nodes[k - 1][obs]))
                        node[obs].SetTo(Variable.Discrete(transProbs[nodes[k - 1][obs]]));
                }
                nodes[k] = node;
            }
            InferenceEngine engine = new InferenceEngine();
            engine.ShowTimings = true;
            engine.ShowWarnings = true;

            Vector[] transpro = new Vector[] {Vector.FromArray(0.2, 0.8), Vector.FromArray(0.1, 0.9)};
            transPrior.ObservedValue = transpro.Select(v => Dirichlet.PointMass(v)).ToArray();
            stateProbs.ObservedValue = Vector.FromArray(1.0, 0);
            var post = engine.Infer(nodes[nodeNum - 1]);
            Console.WriteLine(post);
        }

        public void MarkovChainUnrolledLengthTest()
        {
            Discrete[] transitionProbs = new Discrete[3];
            transitionProbs[0] = new Discrete(0.1, 0.2, 0.7);
            transitionProbs[1] = new Discrete(0.5, 0.2, 0.3);
            transitionProbs[2] = new Discrete(0.7, 0.1, 0.2);
            Console.WriteLine("nodes, time(ms)");
            Func<int, long> work = numNodes =>
            {
                var modelUnrolled = new MarkovChainUnrolled(numNodes);
                modelUnrolled.engine.ModelName = "MarkovChainUnrolled";
                modelUnrolled.engine.ShowTimings = false;
                modelUnrolled.engine.ShowProgress = false;
                modelUnrolled.engine.Compiler.ShowProgress = false;
                for (int k = 0; k < transitionProbs.Length; k++)
                {
                    modelUnrolled.transitionProbs[k].ObservedValue = transitionProbs[k];
                }
                Stopwatch watch = new Stopwatch();
                watch.Start();
                modelUnrolled.Compile();
                watch.Stop();
                return watch.ElapsedMilliseconds;
            };
            work(1);
            work(2);
            work(3);
            for (int nodes = 1; nodes < 500; nodes += System.Math.Max(1, (int)(nodes * 0.1)))
            {
                Console.WriteLine("{0}, {1}", nodes, work(nodes));
            }
        }

        public void MarkovChainUnrolledTest3()
        {
            int timesteps = 20;
            int numPhases = 4;
            Range phase = new Range(numPhases).Named("phase");
            Variable<double>[] observations = new Variable<double>[timesteps];

            var phase1Start = Variable.Observed(0).Named("phase1Start");
            var phase2Start = Variable.Observed(1).Named("phase2Start");
            var phase3Start = Variable.Observed(2).Named("phase3Start");

            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock evBlock = null;
            evBlock = Variable.If(evidence);

            var intercept = Variable.Array<double>(phase).Named("intercept");
            intercept[phase] = Variable.GaussianFromMeanAndPrecision(0, 1e-4).ForEach(phase);
            Variable<int>[] phaseAtTime = new Variable<int>[timesteps];
            var noisePrecision = Variable.GammaFromMeanAndVariance(1, 1).Named("noisePrecision");
            //noisePrecision.ObservedValue = 1;
            var rho = Variable.GaussianFromMeanAndPrecision(0, 1).Named("rho");
            //rho.ObservedValue = 0;

            for (int t = 0; t < timesteps; t++)
            {
                phaseAtTime[t] = Variable.New<int>().Named("phaseAtTime" + t);
                phaseAtTime[t].SetValueRange(phase);
                phaseAtTime[t].AddAttribute(new MarginalPrototype(Discrete.Uniform(numPhases)));
                var phase0 = (t < phase1Start).Named("phase0_" + t);
                using (Variable.If(phase0))
                {
                    phaseAtTime[t].SetTo(0);
                }
                using (Variable.IfNot(phase0))
                {
                    var phase1 = (t < phase2Start).Named("phase1_" + t);
                    using (Variable.If(phase1))
                    {
                        phaseAtTime[t].SetTo(1);
                    }
                    using (Variable.IfNot(phase1))
                    {
                        var phase2 = (t < phase3Start).Named("phase2_" + t);
                        using (Variable.If(phase2))
                        {
                            phaseAtTime[t].SetTo(2);
                        }
                        using (Variable.IfNot(phase2))
                        {
                            phaseAtTime[t].SetTo(3);
                        }
                    }
                }
                if (t == 0)
                {
                    var prevNoise = Variable.GaussianFromMeanAndPrecision(0, noisePrecision).Named("statNoise" + t);
                    prevNoise.InitialiseTo(Gaussian.PointMass(0));
                    using (Variable.Switch(phaseAtTime[t]))
                        observations[t] = Variable.GaussianFromMeanAndPrecision(rho * prevNoise + intercept[phaseAtTime[t]], noisePrecision);
                }
                else
                {
                    observations[t] = Variable.New<double>();
                    var phaseChanged = (phaseAtTime[t] != phaseAtTime[t - 1]).Named("phaseChanged" + t);
                    using (Variable.If(phaseChanged))
                    {
                        var prevNoise = Variable.GaussianFromMeanAndPrecision(0, noisePrecision).Named("statNoise" + t);
                        prevNoise.InitialiseTo(Gaussian.PointMass(0));
                        using (Variable.Switch(phaseAtTime[t]))
                            observations[t].SetTo(Variable.GaussianFromMeanAndPrecision(rho * prevNoise + intercept[phaseAtTime[t]], noisePrecision));
                    }
                    using (Variable.IfNot(phaseChanged))
                    {
                        Variable<double> prevNoise = Variable.New<double>().Named("prevNoise" + t);
                        using (Variable.Switch(phaseAtTime[t - 1]))
                        {
                            prevNoise.SetTo(observations[t - 1] - intercept[phaseAtTime[t - 1]]);
                        }
                        using (Variable.Switch(phaseAtTime[t]))
                            observations[t].SetTo(Variable.GaussianFromMeanAndPrecision(rho * prevNoise + intercept[phaseAtTime[t]], noisePrecision));
                    }
                }
                observations[t].Name = "observation" + t;
                observations[t].ObservedValue = 0;
            }
            if(evBlock != null) evBlock.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            engine.ShowTimings = false;
            engine.ShowProgress = false;
            engine.Compiler.ShowProgress = true;

            Console.WriteLine(engine.Infer(rho));
        }

        public void BayesNetChainTest()
        {
            int NumberOfExamples = 1;
            Range N = new Range(NumberOfExamples).Named("N");
            Range R = new Range(2).Named("R");
            var tablePrior = new Dirichlet(1,1);
            int numNodes = 500;
            var nodes = new VariableArray<int>[numNodes];
            var node = Variable.Array<int>(N).Named("node0");
            node[N] = Variable.Discrete(tablePrior.Sample()).ForEach(N);
            node.SetValueRange(R);
            nodes[0] = node;
            for (int i = 1; i < numNodes; i++)
            {
                var table = Variable.Array<Vector>(R).Named("table"+i);
                table.SetValueRange(R);
                table.ObservedValue = Util.ArrayInit(2, j => tablePrior.Sample());
                nodes[i] = AddChildFromOneParent(nodes[i - 1], table).Named("node"+i);
            }
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.ShowProgress = true;
            int nodeToInfer = numNodes - 1;
            Console.WriteLine("node[{0}] = {1}", nodeToInfer, engine.Infer(nodes[nodeToInfer]));
        }

        /// <summary>
        /// Helper method to add a child from one parent
        /// </summary>
        /// <param name="parent">Parent (a variable array over a range of examples)</param>
        /// <param name="cpt">Conditional probability table</param>
        /// <returns></returns>
        public static VariableArray<int> AddChildFromOneParent(
            VariableArray<int> parent,
            VariableArray<Vector> cpt)
        {
            var n = parent.Range;
            var child = Variable.Array<int>(n);
            using (Variable.ForEach(n))
            using (Variable.Switch(parent[n]))
                child[n] = Variable.Discrete(cpt[parent[n]]);
            return child;
        }

        public void ClippedGaussianWithThresholdParameterSpeedTest()
        {
            Variable<double> threshold = Variable.New<double>().Named("threshold");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            Variable.ConstrainPositive(x - threshold);

            // Set parameter on compiled algorithm
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            threshold.ObservedValue = 0;
            engine.ShowProgress = false;
            engine.ShowTimings = false;
            engine.Compiler.ReturnCopies = false;
            engine.OptimiseForVariables = new List<IVariable>() {x};
            int n = 100000;
            Stopwatch watch = new Stopwatch();
            watch.Start();
            for (int i = 0; i < n; i++)
            {
                double obsThreshold = ((double) i)/n;
                threshold.ObservedValue = obsThreshold;
                object actual = engine.Infer(x);
                //Gaussian actual = engine.Infer<Gaussian>(x);
                //Assert.True(actual.GetMean() > obsThreshold);
            }
            watch.Stop();
            // MSRC-MINKA2
            // original: inference time = 721ms
            // Marginal method: inference time = 573ms
            // SetObservedValue method: inference time = 210ms
            // latest: 120ms
            Console.WriteLine("inference time = {0}ms", watch.ElapsedMilliseconds);

            if (true)
            {
                watch.Reset();
                watch.Start();
                IVariable[] vars = new Variable[] {x};
                for (int i = 0; i < n; i++)
                {
                    double obsThreshold = ((double) i)/n;
                    threshold.ObservedValue = obsThreshold;
                    var ca = engine.GetCompiledInferenceAlgorithm(x);
                    ca.Execute(engine.NumberOfIterations);
                    Gaussian actual = ca.Marginal<Gaussian>(x.Name);
                    Assert.True(actual.GetMean() > obsThreshold);
                }
                watch.Stop();
                // MSRC-MINKA2
                // 110ms
                Console.WriteLine("inference time = {0}ms", watch.ElapsedMilliseconds);
            }

            if (true)
            {
                var ca = engine.GetCompiledInferenceAlgorithm(x);
                watch.Reset();
                watch.Start();
                for (int i = 0; i < n; i++)
                {
                    double obsThreshold = ((double) i)/n;
                    if (false)
                    {
                        threshold.ObservedValue = obsThreshold;
                        ca.SetObservedValue(threshold.Name, threshold.ObservedValue);
                    }
                    else
                    {
                        ca.SetObservedValue("threshold", obsThreshold);
                    }
                    ca.Execute(engine.NumberOfIterations);
                    Gaussian actual = ca.Marginal<Gaussian>(x.Name);
                    Assert.True(actual.GetMean() > obsThreshold);
                }
                watch.Stop();
                // MSRC-MINKA2
                // inference time = 94ms
                Console.WriteLine("inference time = {0}ms", watch.ElapsedMilliseconds);
            }

            ClippedGaussian_EP model = new ClippedGaussian_EP();
            watch.Reset();
            watch.Start();
            for (int i = 0; i < n; i++)
            {
                double obsThreshold = ((double) i)/n;
                model.threshold = obsThreshold;
                model.Execute(50);
                Gaussian actual = model.XMarginal();
                Assert.True(actual.GetMean() > obsThreshold);
            }
            watch.Stop();
            // MSRC-MINKA2
            // inference time = 86ms
            Console.WriteLine("inference time = {0}ms", watch.ElapsedMilliseconds);
        }

        public class ClippedGaussian_EP
        {
            #region Fields

            /// <summary>Field backing the threshold property</summary>
            private double Threshold;

            /// <summary>The number of iterations last computed by Constant. Set this to zero to force re-execution of Constant</summary>
            public int Constant_iterationsDone;

            /// <summary>The number of iterations last computed by Changed_threshold. Set this to zero to force re-execution of Changed_threshold</summary>
            public int Changed_threshold_iterationsDone;

            /// <summary>Field backing the AfterUpdate property</summary>
            private Action<int> afterUpdate;

            /// <summary>Field backing the ResumeLastRun property</summary>
            private bool resumeLastRun;

            /// <summary>The constant 'vGaussian0'</summary>
            public Gaussian vGaussian0;

            /// <summary>Message from definition of 'x'</summary>
            public Gaussian x_F;

            /// <summary>Messages to uses of 'x'</summary>
            public Gaussian[] x_uses_F;

            /// <summary>Messages from uses of 'x'</summary>
            public Gaussian[] x_uses_B;

            /// <summary>Message from definition of 'vdouble2'</summary>
            public Gaussian vdouble2_F;

            /// <summary>Messages to uses of 'vdouble2'</summary>
            public Gaussian[] vdouble2_uses_F;

            /// <summary>Messages from uses of 'vdouble2'</summary>
            public Gaussian[] vdouble2_uses_B;

            /// <summary>Message to definition of 'vdouble2'</summary>
            public Gaussian vdouble2_B;

            /// <summary>Message to marginal of 'x'</summary>
            public Gaussian x_marginal_B;

            #endregion

            #region Properties

            /// <summary>The externally-specified value of 'threshold'</summary>
            public double threshold
            {
                get { return this.Threshold; }
                set
                {
                    if (this.Threshold != value)
                    {
                        this.Threshold = value;
                        this.Changed_threshold_iterationsDone = 0;
                    }
                }
            }

            /// <summary>Called after each iteration</summary>
            public Action<int> AfterUpdate
            {
                get { return this.afterUpdate; }
                set { this.afterUpdate = value; }
            }

            /// <summary>Set to true to re-use previous message state</summary>
            public bool ResumeLastRun
            {
                get { return this.resumeLastRun; }
                set { this.resumeLastRun = value; }
            }

            #endregion

            #region Methods

            /// <summary>Set the observed value of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            /// <param name="value">Observed value</param>
            public void SetObservedValue(string variableName, object value)
            {
                if (variableName == "threshold")
                {
                    this.threshold = (double) value;
                }
            }

            /// <summary>The marginal distribution of the specified variable.</summary>
            /// <param name="variableName">Variable name</param>
            public object Marginal(string variableName)
            {
                if (variableName == "x")
                {
                    return this.XMarginal();
                }
                return null;
            }

            /// <summary>Update all marginals, by iterating message passing the given number of times</summary>
            /// <param name="numberOfIterations">The number of times to iterate each loop</param>
            public void Execute(int numberOfIterations)
            {
                this.Constant();
                this.Changed_threshold();
            }

            /// <summary>Reset all messages to their initial values on the next call to Execute</summary>
            public void Reset()
            {
                this.Constant_iterationsDone = 0;
            }

            /// <summary>Computations that do not depend on observed values</summary>
            public void Constant()
            {
                if (this.Constant_iterationsDone == 1)
                {
                    return;
                }
                this.vGaussian0 = Gaussian.FromNatural(0, 1);
                this.x_F = ArrayHelper.MakeUniform<Gaussian>(vGaussian0);
                // Message to 'x' from Random factor
                this.x_F = UnaryOp<double>.RandomAverageConditional<Gaussian>(this.vGaussian0);
                // Create array for 'x_uses' forwards messages.
                this.x_uses_F = new Gaussian[1];
                for (int _ind0 = 0; _ind0 < 1; _ind0++)
                {
                    this.x_uses_F[_ind0] = ArrayHelper.MakeUniform<Gaussian>(this.vGaussian0);
                }
                // Create array for 'x_uses' backwards messages.
                this.x_uses_B = new Gaussian[1];
                for (int _ind0 = 0; _ind0 < 1; _ind0++)
                {
                    this.x_uses_B[_ind0] = ArrayHelper.MakeUniform<Gaussian>(this.vGaussian0);
                }
                // Message to 'x_uses' from UsesEqualDef factor
                this.x_uses_F[0] = UsesEqualDefOp.UsesAverageConditional<Gaussian>(this.x_uses_B, this.x_F, 0, this.x_uses_F[0]);
                this.vdouble2_F = ArrayHelper.MakeUniform<Gaussian>(vGaussian0);
                // Create array for 'vdouble2_uses' forwards messages.
                this.vdouble2_uses_F = new Gaussian[1];
                for (int _ind0 = 0; _ind0 < 1; _ind0++)
                {
                    this.vdouble2_uses_F[_ind0] = ArrayHelper.MakeUniform<Gaussian>(this.vGaussian0);
                }
                // Create array for 'vdouble2_uses' backwards messages.
                this.vdouble2_uses_B = new Gaussian[1];
                for (int _ind0 = 0; _ind0 < 1; _ind0++)
                {
                    this.vdouble2_uses_B[_ind0] = ArrayHelper.MakeUniform<Gaussian>(this.vGaussian0);
                }
                this.vdouble2_B = ArrayHelper.MakeUniform<Gaussian>(vGaussian0);
                this.Constant_iterationsDone = 1;
                this.Changed_threshold_iterationsDone = 0;
            }

            /// <summary>Computations that depend on the observed value of threshold</summary>
            public void Changed_threshold()
            {
                if (this.Changed_threshold_iterationsDone == 1)
                {
                    return;
                }
                // Message to 'vdouble2' from Difference factor
                this.vdouble2_F = DoublePlusOp.AAverageConditional(this.x_uses_F[0], this.threshold);
                // Message to 'vdouble2_uses' from ReplicateWithMarginal factor
                this.vdouble2_uses_F[0] = ReplicateOp_NoDivide.UsesAverageConditional<Gaussian>(this.vdouble2_uses_B, this.vdouble2_F, 0, this.vdouble2_uses_F[0]);
                // Message to 'vdouble2_uses' from Positive factor
                this.vdouble2_uses_B[0] = IsPositiveOp.XAverageConditional(true, this.vdouble2_uses_F[0]);
                // Message to 'vdouble2' from ReplicateWithMarginal factor
                this.vdouble2_B = ReplicateOp_NoDivide.DefAverageConditional<Gaussian>(this.vdouble2_uses_B, this.vdouble2_B);
                // Message to 'x_uses' from Difference factor
                this.x_uses_B[0] = DoublePlusOp.SumAverageConditional(this.vdouble2_B, this.threshold);
                this.x_marginal_B = ArrayHelper.MakeUniform<Gaussian>(vGaussian0);
                // Message to 'x_marginal' from UsesEqualDef factor
                this.x_marginal_B = UsesEqualDefOp.MarginalAverageConditional<Gaussian>(this.x_uses_B, this.x_F, this.x_marginal_B);
                this.Changed_threshold_iterationsDone = 1;
            }

            /// <summary>
            /// Returns the marginal distribution for 'x' given by the current state of the
            /// message passing algorithm.
            /// </summary>
            /// <returns>The marginal distribution</returns>
            public Gaussian XMarginal()
            {
                return this.x_marginal_B;
            }

            #endregion
        }

        public void SparseVersusDense()
        {
            // Build a large vector and do typical operation on it. See at what
            // sparsity point the sparse version becomes slower than the dense version.
            // This informs the decision for choosing at what point we should
            // convert sparse to dense in some operations.
            int length = 10000;
            Stopwatch watch = new Stopwatch();

            for (int i = 100; i < length/2; i += 100)
            {
                Vector sparse = SparseVector.Zero(length);
                Vector sparse1 = SparseVector.Zero(length);
                Vector sparse2 = SparseVector.Zero(length);
                Vector dense = DenseVector.Zero(length);
                Vector dense1 = DenseVector.Zero(length);
                Vector dense2 = DenseVector.Zero(length);
                int[] perm1 = Rand.Perm(length);
                int[] perm2 = Rand.Perm(length);
                for (int j = 0; j < i; j++)
                {
                    dense1[perm1[j]] = 1;
                    sparse1[perm1[j]] = 1;
                    dense2[perm2[j]] = 1;
                    sparse2[perm2[j]] = 1;
                }

                int numIters = 1000;
                watch.Reset();
                watch.Start();
                for (int k = 0; k < numIters; k++)
                {
                    sparse.SetToSum(sparse1, sparse2);
                }
                watch.Stop();
                double sparseTime = watch.ElapsedMilliseconds;
                watch.Reset();
                watch.Start();
                for (int k = 0; k < numIters; k++)
                {
                    dense.SetToSum(dense1, dense2);
                }
                watch.Stop();
                double denseTime = watch.ElapsedMilliseconds;
                Console.WriteLine("Sparsity fraction: {0}: Sparse time: {1}, dense time: {2}", ((double) i)/length, sparseTime, denseTime);
            }
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
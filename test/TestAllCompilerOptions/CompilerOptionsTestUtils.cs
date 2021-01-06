// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests.TestAllCompilerOptions
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Reflection;
    using System.Threading;
    using System.Threading.Tasks;

    using Xunit.Sdk;

    using Microsoft.ML.Probabilistic.Compiler;
    using Microsoft.ML.Probabilistic.Models;

    public static class CompilerOptionsTestUtils
    {
        public static int TestAllCompilerOptions(bool loadTestAssembly = false)
        {
            var assembly = Assembly.GetCallingAssembly();
            if (loadTestAssembly)
                assembly = Assembly.Load("Microsoft.ML.Probabilistic.Tests");
            var tests = TestUtils.FindTestMethods(assembly);

            Console.WriteLine($"Running {tests.Length} tests with all compiler options");
            TestUtils.WriteEnvironment();

            TestUtils.SetBrowserMode(BrowserMode.Never);
            var failed = TestAllCompilerOptions(tests);

            Console.WriteLine("Executed {0} tests", tests.Length);
            foreach (var g in failed.GroupBy(t => t.Item2))
            {
                Console.WriteLine("{0}.{1} failed:", g.Key.DeclaringType, g.Key.Name);
                foreach (var t in g)
                {
                    Console.WriteLine("\t{0}", t.Item1);
                }
            }
            return failed.Count();
        }

        private static List<Tuple<string, MethodInfo>> TestAllCompilerOptions(
            MethodInfo[] tests)
        {
            var appDomains = new List<AppDomain>();

            var output = new List<Tuple<string, MethodInfo>>();
            try
            {
                var testRunners = new List<TestRunner2>();

                var instance = 0;

                foreach (var loops in new[] { true, false })
                {
                    InferenceEngine.DefaultEngine.Compiler.UseParallelForLoops = loops;
                    foreach (var free in new[] { true, false })
                    {
                        InferenceEngine.DefaultEngine.Compiler.FreeMemory = free;
                        foreach (var copies in new[] { true, false })
                        {
                            InferenceEngine.DefaultEngine.Compiler.ReturnCopies = copies;
                            foreach (var optimise in new[] { true, false })
                            {
                                InferenceEngine.DefaultEngine.Compiler.OptimiseInferenceCode = optimise;

                                var appDomain = AppDomain.CreateDomain(
                                    Guid.NewGuid().ToString(),
                                    new System.Security.Policy.Evidence(),
                                    appBasePath: Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                                    appRelativeSearchPath: null,
                                    shadowCopyFiles: false);

                                appDomains.Add(appDomain);

                                var isolatedTestRunner = (TestRunner2)appDomain.CreateInstanceFrom(
                                    Assembly.GetExecutingAssembly().Location,
                                    typeof(TestRunner2).FullName).Unwrap();

                                isolatedTestRunner.Initialize(instance++, loops, free, copies, optimise, tests);

                                testRunners.Add(isolatedTestRunner);
                            }
                        }
                    }
                }

                Parallel.ForEach(testRunners, item =>
                {
                    item.RunTests();
                });
            }
            finally
            {
                foreach (var item in appDomains)
                {
                    AppDomain.Unload(item);
                }
            }

            return output;
        }

        public class TestRunner2 : MarshalByRefObject
        {
            int instance;
            bool loops;
            bool free;
            bool copies;
            bool optimise;
            MethodInfo[] tests;

            public void Initialize(
                int instance,
                bool loops,
                bool free,
                bool copies,
                bool optimise,
                MethodInfo[] tests)
            {
                this.instance = instance;
                this.loops = loops;
                this.free = free;
                this.copies = copies;
                this.optimise = optimise;
                this.tests = tests;
            }

            public List<Tuple<string, MethodInfo>> RunTests()
            {
                InferenceEngine.DefaultEngine.Compiler.UseParallelForLoops = loops;
                InferenceEngine.DefaultEngine.Compiler.FreeMemory = free;
                InferenceEngine.DefaultEngine.Compiler.ReturnCopies = copies;
                InferenceEngine.DefaultEngine.Compiler.OptimiseInferenceCode = optimise;

                var testsDone = 0;
                var totalTests = tests.Length;

                var startTime = DateTime.UtcNow;
                var lastTime = DateTime.UtcNow;

                var stdout = Console.Out;
                Console.SetOut(StreamWriter.Null);

                void TestFinished(string name, TimeSpan testDuration)
                {
                    Interlocked.Increment(ref testsDone);
                    stdout.WriteLine($"{instance}: {100.0 * testsDone / totalTests}% Elapsed: {DateTime.UtcNow.Subtract(startTime)} Name: {name} Duration: {DateTime.UtcNow.Subtract(lastTime)}");
                    lastTime = DateTime.UtcNow;
                }

                var state = string.Format("UseParallelForLoops={0} FreeMemory={1} ReturnCopies={2} OptimiseInferenceCode={3}", loops, free, copies, optimise);
                var list = RunAllTests(TestFinished, tests).Select(m => new Tuple<string, MethodInfo>(state, m)).ToList();
                return list;
            }

            public static void RunAllTests(Action<string, TimeSpan> testFinished, string path)
            {
                List<string> testNames = TestUtils.ReadTestNames(path);
                var testMethods = testNames.Select(name =>
                {
                    int commaPos = name.IndexOf(',');
                    string methodName = name.Substring(0, commaPos);
                    string className = name.Substring(commaPos + 1);
                    Console.WriteLine(name);
                    Type type = Type.GetType(className, true);
                    MethodInfo method = type.GetMethod(methodName, Type.EmptyTypes);
                    return method;
                }).ToArray();
                RunAllTests(testFinished, testMethods);
            }

            private static void ForEach<T>(IEnumerable<T> tests, Action<T> action)
            {
                foreach (var item in tests)
                {
                    action(item);
                }
            }

            private static IEnumerable<MethodInfo> RunAllTests(Action<string, TimeSpan> testFinished, MethodInfo[] tests)
            {
                var failed = new ConcurrentQueue<MethodInfo>();
                var safeTests = new ConcurrentQueue<MethodInfo>();
                var unsafeTests = new ConcurrentQueue<MethodInfo>();
                ForEach(tests, test =>
                {
                    var testCategories = TraitHelper.GetTraits(test)
                        .Where(trait => trait.Key == "Category")
                        .Select(trait => trait.Value)
                        .ToArray();
                    bool isUnsafe = testCategories.Any(tc => tc == "CsoftModel" || tc == "ModifiesGlobals" || tc == "DistributedTests" || tc == "Performance");
                    if (isUnsafe) unsafeTests.Enqueue(test);
                    else safeTests.Enqueue(test);
                });
                // unsafe tests must run sequentially.
                Trace.WriteLine($"Running {unsafeTests.Count} tests sequentially");
                foreach (var test in unsafeTests)
                {
                    var sw = Stopwatch.StartNew();
                    RunTest(test, failed);
                    testFinished(test.Name, sw.Elapsed);
                }
                var safeTestsArray = safeTests.ToArray();
                Array.Sort(safeTestsArray, (a, b) => a.Name.CompareTo(b.Name));
                Trace.WriteLine($"Running {safeTests.Count} tests in parallel");
                try
                {
                    Parallel.ForEach(safeTestsArray, test =>
                    {
                        var sw = Stopwatch.StartNew();
                        RunTest(test, failed);
                        testFinished(test.Name, sw.Elapsed);
                    });
                }
                catch (AggregateException ex)
                {
                    // To make the Visual Studio debugger stop at the inner exception, check "Enable Just My Code" in Debug->Options.
                    // throw InnerException while preserving stack trace
                    // https://stackoverflow.com/questions/57383/in-c-how-can-i-rethrow-innerexception-without-losing-stack-trace
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                    throw;
                }
                return failed;
            }

            private static void RunTest(MethodInfo test, ConcurrentQueue<MethodInfo> failed)
            {
                object obj = Activator.CreateInstance(test.DeclaringType);
                BindingFlags flags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.InvokeMethod;
                try
                {
                    Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeMember(test.DeclaringType, test.Name, flags, obj);
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(test);
                    Trace.WriteLine(ex);
                    failed.Enqueue(test);
                }
            }
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Collections.Concurrent;
    using System.IO;
    using System.Linq;
    using System.Reflection;
    using System.Xml;
    using System.Diagnostics;
    using System.Threading.Tasks;

    using Xunit;
    using Assert = Xunit.Assert;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Compiler.Transforms;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Compiler;
    using Microsoft.ML.Probabilistic.Math;
    using Xunit.Sdk;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Factors;
    using System.Threading;

    public static class TestUtils
    {
        public static string DataFolderPath { get; } = Path.GetFullPath(Path.Combine(
#if NETCOREAPP
                Path.GetDirectoryName(typeof(ClickTest).Assembly.Location), // work dir is not the one with Microsoft.ML.Probabilistic.Tests.dll on netcore and neither is .Location on netframework
#endif
                "Data"));

        public static void SetDebugOptions()
        {
            //MessageTransform.debug = true;
            //SchedulingTransform.doRepair = false;
            //SchedulingTransform.UseScheduling2 = false;
            //Scheduler.verbose = true;
            //Scheduler.showGraphs = true;
            //Scheduler.NoInitEdgesAreInfinite = true;
            //Scheduler.showOffsetEdges = true;
            //DependencyGraph.debug = true;
            //SchedulingTransform.debug = true;
            //Scheduler.showCapacityBreakdown = true;
            //LoopMerging2Transform.debug = true;
            //LoopReversalTransform.debug = true;
            //LocalAllocationTransform.debug = true;
            //LivenessAnalysisTransform.debug = true;
            //IterativeProcessTransform.debug = true;
            //ForwardBackwardTransform.debug = true;
            //MessageTransform.UseMessageAnalysis = false;
            //PruningTransform.PruneUniformStmts = false;
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.WordWrap = false;
            //IndexingTransform.UseGetItems = false;

            // SHOW STATEMENT AND DECLARATION ATTRIBUTES IN THE TRANSFORM BROWSER
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.ChannelInfo));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.DoNotSendEvidence));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.GroupMember));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.GateBlock));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.MessagePathAttribute));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.ChannelPathAttribute));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.IterationVariableAttribute));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Factors.DependsOnIterationAttribute));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.Initializer));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.IsInferred));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Output));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.MessageArrayInformation));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.OperatorStatement));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.AccumulationInfo));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Factors.MultiplyAllAttribute));
            //Microsoft.ML.Probabilistic.Compiler.View.DeclarationView.AttributeTypesToDisplay.Add(typeof(Microsoft.ML.Probabilistic.Compiler.Transforms.DependencyInformation));
        }

        public static void SetBrowserMode(BrowserMode mode)
        {
            InferenceEngine.DefaultEngine.Compiler.BrowserMode = mode;
        }

        public static void WriteFactorDocumentation()
        {
            //TODO: change path for cross platform using
            FactorDocumentationWriter.WriteFactorDocumentation(@"..\..\..\Runtime\Factors\FactorDocs.xml");
        }

        // check that transform names match their type names
        public static void CheckTransformNames()
        {
            AppDomain app = AppDomain.CurrentDomain;
            Assembly[] assemblies = app.GetAssemblies();
            foreach (Assembly assembly in assemblies)
            {
                // scan all types in the assembly
                Type[] types = assembly.GetTypes();
                foreach (Type type in types)
                {
                    Type face = type.GetInterface(typeof(ICodeTransform).Name);
                    if (face == null) continue;
                    ICodeTransform ict;
                    // find a constructor
                    ConstructorInfo[] ctors = type.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.CreateInstance);
                    if (ctors.Length == 0) ict = (ICodeTransform)Activator.CreateInstance(type);
                    else
                    {
                        ConstructorInfo ctor = ctors[0];
                        ParameterInfo[] parameters = ctor.GetParameters();
                        object[] args = new object[parameters.Length];
                        for (int i = 0; i < parameters.Length; i++)
                        {
                            args[i] = GetDefault(parameters[i].ParameterType);
                        }
                        ict = (ICodeTransform)ctor.Invoke(args);
                    }
                    if (ict.Name != type.Name) throw new Exception(String.Format("{0} has Name=\"{1}\"", type.Name, ict.Name));
                }
            }
        }

        public static void WriteLoadedAssemblies()
        {
            AppDomain app = AppDomain.CurrentDomain;
            Assembly[] assemblies = app.GetAssemblies();
            foreach (Assembly assembly in assemblies)
            {
                Trace.WriteLine(assembly);
            }
        }

        /// <summary>
        /// Write the static fields in all loaded assemblies.  Useful for manually checking thread safety.
        /// </summary>
        public static void WriteStaticFields()
        {
            Assembly assembly = typeof(InferenceEngine).Assembly;
            foreach (var typeInfo in assembly.DefinedTypes)
            {
                foreach (var field in typeInfo.DeclaredFields)
                {
                    if (field.IsStatic && !field.IsLiteral)
                    {
                        Trace.WriteLine($"{typeInfo.FullName}.{field.Name}");
                    }
                }
            }
        }

        public static object GetDefault(Type type)
        {
            if (type.IsValueType) return Activator.CreateInstance(type);
            else return null;
        }

        public static int FindMinimumNumberOfIterations(Action test)
        {
            int oldN = InferenceEngine.DefaultEngine.NumberOfIterations;
            int min = FindMinimum(10, delegate (int n)
                {
                    InferenceEngine.DefaultEngine.NumberOfIterations = n;
                    try
                    {
                        test();
                    }
                    catch
                    {
                        return false;
                    }
                    return true;
                });
            InferenceEngine.DefaultEngine.NumberOfIterations = oldN;
            return min;
        }

        public static int FindMinimum(int start, Func<int, bool> func)
        {
            int maxFalse = 0;
            // find an upper bound
            int x = start;
            while (true)
            {
                if (func(x)) break;
                maxFalse = x;
                try
                {
                    checked
                    {
                        x *= 2;
                    }
                }
                catch (OverflowException)
                {
                    throw new Exception("Test does not pass for any number of iterations");
                }
            }
            int minTrue = x;
            // binary search
            while (minTrue > maxFalse + 1)
            {
                int middle = (minTrue + maxFalse + 1) / 2;
                if (func(middle)) minTrue = middle;
                else maxFalse = middle;
            }
            return minTrue;
        }

        public static void WriteStaticPropertyValues(Type type)
        {
            var properties = type.GetProperties(BindingFlags.Public | BindingFlags.Static);
            foreach (var prop in properties)
                Console.WriteLine("{0} = {1}", prop.Name, prop.GetValue(null));
        }

        public static void WriteEnvironment()
        {
            Console.WriteLine("Environment settings:");
            WriteStaticPropertyValues(typeof(Environment));
            Console.WriteLine("GC settings:");
            WriteStaticPropertyValues(typeof(System.Runtime.GCSettings));
            var coreAssemblyInfo = FileVersionInfo.GetVersionInfo(typeof(object).Assembly.Location);
            Console.WriteLine($"mscorlib version {coreAssemblyInfo.ProductVersion}");
        }

        public static int TestAllCompilerOptions(
            string workingDirectory,
            bool loadTestAssembly,
            bool loops,
            bool free,
            bool copies,
            bool optimise,
            string path = null)
        {
            Directory.SetCurrentDirectory(workingDirectory);

            var assembly = Assembly.GetCallingAssembly();
            if (loadTestAssembly)
                assembly = Assembly.Load("Microsoft.ML.Probabilistic.Tests");
            var tests = TestUtils.FindTestMethods(assembly, path);

            Console.WriteLine($"Running {tests.Length} tests with all compiler options");
            TestUtils.WriteEnvironment();

            TestUtils.SetBrowserMode(BrowserMode.Never);
            var failed = RunTests(tests,
                loops: loops,
                free: free,
                copies: copies,
                optimise: optimise);

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

        public static MethodInfo[] FindTestMethods(
            Assembly assembly,
            string filter = null,
            string path = null)
        {
            var methods = GetTests(assembly, path);

            var excludeCategories = new[] { "BadTest", "OpenBug", "CompilerOptionsTest", "x86" };
            var testMethods = methods.Where(method =>
            {
                if (!method.GetCustomAttributes().Any(attr => attr is FactAttribute)) return false;
                var categories = TraitHelper.GetTraits(method)
                    .Where(trait => trait.Key == "Category" || trait.Key == "Platform")
                    .Select(trait => trait.Value)
                    .ToArray();
                return !categories.Any(excludeCategories.Contains) && (filter == null || categories.Contains(filter));
            }).ToArray();
            return testMethods;
        }

        public static List<Tuple<string, MethodInfo>> RunTests(
            MethodInfo[] tests,
            bool loops,
            bool free,
            bool copies,
            bool optimise)
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
            try
            {
                Console.SetOut(StreamWriter.Null);

                void TestFinished(string name, TimeSpan testDuration)
                {
                    Interlocked.Increment(ref testsDone);
                    stdout.WriteLine($"{100.0 * testsDone / totalTests}% Elapsed: {DateTime.UtcNow.Subtract(startTime)} Name: {name} Duration: {DateTime.UtcNow.Subtract(lastTime)}");
                    lastTime = DateTime.UtcNow;
                }

                return RunAllTests(TestFinished, tests, runInParallel: false).Select(m => new Tuple<string, MethodInfo>($"UseParallelForLoops={loops} FreeMemory={free} ReturnCopies={copies} OptimiseInferenceCode={optimise}: {m.error}", m.test)).ToList();
            }
            finally
            {
                Console.SetOut(stdout);
            }
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
            RunAllTests(testFinished, testMethods, runInParallel: false);
        }

        private static IEnumerable<(MethodInfo test, Exception error)> RunAllTests(Action<string, TimeSpan> testFinished, MethodInfo[] tests, bool runInParallel)
        {
            var failed = new ConcurrentQueue<(MethodInfo, Exception)>();
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
            Trace.WriteLine($"Running {unsafeTests.Count} unsafe tests sequentially");
            foreach (var test in unsafeTests)
            {
                var sw = Stopwatch.StartNew();
                RunTest(test, failed);
                testFinished(test.Name, sw.Elapsed);
            }
            var safeTestsArray = safeTests.ToArray();
            Array.Sort(safeTestsArray, (a, b) => a.Name.CompareTo(b.Name));
            if (runInParallel)
            {
                Trace.WriteLine($"Running {safeTests.Count} safe tests in parallel");
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
            }
            else
            {
                Trace.WriteLine($"Running {unsafeTests.Count} safe tests sequentially");
                foreach (var test in safeTestsArray)
                {
                    var sw = Stopwatch.StartNew();
                    RunTest(test, failed);
                    testFinished(test.Name, sw.Elapsed);
                }
            }

            return failed;
        }

        private static void ForEach<T>(IEnumerable<T> tests, Action<T> action)
        {
            foreach (var item in tests)
            {
                action(item);
            }
        }

        private static void RunTest(MethodInfo test, ConcurrentQueue<(MethodInfo, Exception)> failed)
        {
            object obj = Activator.CreateInstance(test.DeclaringType);
            BindingFlags flags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.InvokeMethod;
            try
            {
                Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeMember(test.DeclaringType, test.Name, flags, obj);
            }
            catch (Exception ex)
            {
                failed.Enqueue((test, ex));
            }
        }

        private static MethodInfo[] GetTests(
            Assembly assembly,
            string path = null)
        {
            if (path == null)
            {
                return assembly.GetTypes().SelectMany(t => t.GetMethods()).ToArray();
            }

            List<string> testNames = ReadTestNames(path);
            return testNames.Select(name =>
            {
                int commaPos = name.IndexOf(',');
                string methodName = name.Substring(0, commaPos);
                string className = name.Substring(commaPos + 1);
                Console.WriteLine(name);
                Type type = Type.GetType(className, true);
                MethodInfo method = type.GetMethod(methodName, Type.EmptyTypes);
                return method;
            }).ToArray();
        }

        public static int TestAllCompilerOptions(string path = null)
        {
            var workingDirectory = Directory.GetCurrentDirectory();

            var failures = 0;
            var loadTestAssembly = true;
            foreach (var loops in new[] { false, true })
            {
                foreach (var free in new[] { false, true })
                {
                    foreach (var copies in new[] { false, true })
                    {
                        foreach (var optimise in new[] { false, true })
                        {
                            failures += TestAllCompilerOptions(
                                workingDirectory,
                                loadTestAssembly: loadTestAssembly,
                                loops: loops,
                                free: free,
                                copies: copies,
                                optimise: optimise,
                                path: path);

                            // Only do this the first time.
                            loadTestAssembly = false;
                        }
                    }
                }
            }

            return failures;
        }

        public static void WriteCodeToRunAllTests(TextWriter writer, string path)
        {
            writer.WriteLine("InferenceEngine.DefaultEngine.BrowserMode = BrowserMode.Never;");
            List<string> testNames = ReadTestNames(path);
            foreach (string name in testNames)
            {
                int commaPos = name.IndexOf(',');
                string methodName = name.Substring(0, commaPos);
                string className = name.Substring(commaPos + 1);
                commaPos = className.IndexOf(',');
                className = className.Substring(0, commaPos);
                int dotPos = className.LastIndexOf('.');
                className = className.Substring(dotPos + 1);
                Console.WriteLine(name);
                writer.WriteLine("(new {0}()).{1}();", className, methodName);
                writer.WriteLine("Variable.CloseAllBlocks();");
            }
        }

        public static List<string> ReadTestNames(string path)
        {
            List<string> testNames = new List<string>();
            XmlTextReader reader = new XmlTextReader(path);
            while (reader.Read())
            {
                reader.MoveToContent();
                if (reader.NodeType == XmlNodeType.Element && reader.Name == "TestMethod")
                {
                    reader.MoveToAttribute("className");
                    reader.ReadAttributeValue();
                    string className = reader.Value;
                    reader.MoveToAttribute("name");
                    string methodName = reader.Value;
                    testNames.Add(methodName + "," + className);
                }
            }
            return testNames;
        }

        /// <summary>
        /// Reads in all of the test result (.trx) files in TestResults and produces a csv file summarizing them.
        /// </summary>
        /// <remarks>
        /// The output file is called duration.csv and placed in the TestResults folder.
        /// It contains the durations in milliseconds, with columns corresponding to the date of the test (leftmost column is oldest).  
        /// The success or failure of the test is ignored.  
        /// You can open this file in Excel and get a quick overview of how the speed of each test is changing over time.  
        /// </remarks>
        public static void WriteAllTestDurations()
        {
            IList<string> paths = TestUtils.GetTestResultPaths();
            if (paths.Count == 0) return;
            Dictionary<string, TestResults> dict = new Dictionary<string, TestResults>();
            Set<string> names = new Set<string>();
            foreach (string path in paths)
            {
                TestResults results = TestUtils.ReadTestResults(path);
                dict[path] = results;
                names.AddRange(results.results.Keys);
            }
            List<string> namesSorted = new List<string>(names);
            namesSorted.Sort();
            //TODO: change path for cross platform using
            using (TextWriter writer = new StreamWriter(@"..\..\..\TestResults\duration.csv"))
            {
                // csv header
                writer.Write("Test Name");
                foreach (string path in paths)
                {
                    writer.Write(',');
                    writer.Write(dict[path].time);
                }
                writer.WriteLine();
                // csv rows
                foreach (string name in namesSorted)
                {
                    writer.Write(name);
                    foreach (string path in paths)
                    {
                        writer.Write(',');
                        TestResults results = dict[path];
                        TestResult result;
                        if (results.results.TryGetValue(name, out result))
                        {
                            double milliseconds = result.duration.Ticks * 0.0001;
                            writer.Write(milliseconds.ToString("f1"));
                        }
                    }
                    writer.WriteLine();
                }
            }
        }

        public static void WriteAllTestComparisons()
        {
            IList<string> paths = TestUtils.GetTestResultPaths();
            if (paths.Count == 0) return;
            string oldPath = paths[0];
            TestResults oldResults = TestUtils.ReadTestResults(oldPath);
            bool notFirst = false;
            foreach (string newPath in paths)
            {
                if (!notFirst)
                {
                    notFirst = true;
                    continue;
                }
                TestResults newResults = TestUtils.ReadTestResults(newPath);
                FileInfo fileInfo = new FileInfo(newPath);
                string diffFileName = fileInfo.FullName.Substring(0, fileInfo.FullName.Length - fileInfo.Extension.Length) + " diff.txt";
                using (TextWriter writer = new StreamWriter(diffFileName))
                {
                    TestUtils.WriteComparison(writer, oldResults, newResults);
                }
                oldPath = newPath;
                oldResults = newResults;
            }
            //TestResults newResults = TestUtils.ReadTestResults(@"..\..\..\TestResults\minka_MSRC-MINKA2 2010-01-08 17_31_34.trx");
            //TestResults oldResults = TestUtils.ReadTestResults(@"..\..\..\TestResults\minka_MSRC-MINKA2 2010-01-08 17_28_32.trx");
            //TestUtils.WriteComparison(oldResults, newResults);
        }

        public static IList<string> GetTestResultPaths()
        {
            //TODO: change path for cross platform using
            DirectoryInfo dir = new DirectoryInfo(@"..\..\..\TestResults");
            FileInfo[] files = dir.GetFiles("*.trx");
            Array.Sort(files, delegate (FileInfo a, FileInfo b) { return a.CreationTimeUtc.CompareTo(b.CreationTimeUtc); });
            return Array.ConvertAll(files, fileInfo => fileInfo.FullName);
        }

        public static TestResults ReadTestResults(string path)
        {
            TestResults results = new TestResults();
            bool foundTime = false;
            bool foundComputerName = false;
            XmlTextReader reader = new XmlTextReader(path);
            while (reader.Read())
            {
                reader.MoveToContent();
                if (reader.NodeType == XmlNodeType.Element && reader.Name == "UnitTestResult")
                {
                    reader.MoveToAttribute("testName");
                    reader.ReadAttributeValue();
                    string testName = reader.Value;
                    while (results.results.ContainsKey(testName)) testName = IncrementName(testName);
                    reader.MoveToAttribute("computerName");
                    if (!foundComputerName)
                    {
                        results.computerName = reader.Value;
                        foundComputerName = true;
                    }
                    else if (results.computerName != reader.Value)
                        Console.WriteLine("Warning: computerName varies between tests (" + results.computerName + "," + reader.Value + ")");
                    bool hasDuration = reader.MoveToAttribute("duration");
                    if (!hasDuration) continue;
                    string durationString = reader.Value;
                    reader.MoveToAttribute("startTime");
                    DateTime time = DateTime.Parse(reader.Value);
                    if (!foundTime) results.time = time;
                    else results.time = Min(results.time, time);
                    reader.MoveToAttribute("outcome");
                    string outcome = reader.Value;
                    TestResult result = new TestResult();
                    result.outcome = outcome;
                    result.duration = TimeSpan.Parse(durationString);
                    results.results[testName] = result;
                }
            }
            return results;
        }

        public static string IncrementName(string name)
        {
            int prefixLength;
            int count = GetNameCount(name, out prefixLength) + 1;
            return name.Substring(0, prefixLength) + " (" + count + ")";
        }

        public static int GetNameCount(string name, out int prefixLength)
        {
            prefixLength = name.Length;
            if (name[name.Length - 1] == ')')
            {
                int start = name.LastIndexOf(' ');
                if (start != -1)
                {
                    start++;
                    if (name[start] == '(')
                    {
                        start++;
                        try
                        {
                            int count = int.Parse(name.Substring(start, name.Length - start - 1));
                            prefixLength = start - 2;
                            return count;
                        }
                        catch
                        {
                        }
                    }
                }
            }
            return 1;
        }

        public static DateTime Min(DateTime a, DateTime b)
        {
            return (a <= b) ? a : b;
        }

        public static IList<TestResultDiff> CompareResults(TestResults oldResults, TestResults newResults)
        {
            List<TestResultDiff> diffs = new List<TestResultDiff>();
            foreach (KeyValuePair<string, TestResult> entry1 in oldResults.results)
            {
                string key = entry1.Key;
                string diff = "";
                double priority = 0;
                TestResult oldResult = entry1.Value;
                TestResult newResult;
                if (!newResults.results.TryGetValue(key, out newResult))
                {
                    diff = "Missing";
                }
                else
                {
                    if (oldResult.outcome != newResult.outcome)
                    {
                        diff = "Different outcomes (" + newResult.outcome + " instead of " + oldResult.outcome + ")";
                        priority = Double.PositiveInfinity;
                    }
                    else
                    {
                        TimeSpan threshold = new TimeSpan(oldResult.duration.Ticks / 100);
                        TimeSpan delta = newResult.duration - oldResult.duration;
                        double percent = 100 * (double)delta.Ticks / oldResult.duration.Ticks;
                        priority = percent;
                        double newMs = newResult.duration.Ticks * 0.0001; // duration in milliseconds
                        double oldMs = oldResult.duration.Ticks * 0.0001;
                        if (delta > threshold)
                        {
                            diff = percent.ToString("0") + "% slower (" + newMs.ToString("f1") + " instead of " + oldMs.ToString("f1") + "ms)";
                        }
                        else if (delta < -threshold)
                        {
                            diff = (-percent).ToString("0") + "% faster (" + newMs.ToString("f1") + " instead of " + oldMs.ToString("f1") + "ms)";
                        }
                    }
                }
                if (diff.Length > 0) diffs.Add(new TestResultDiff(key, diff, priority));
            }
            diffs.Sort(delegate (TestResultDiff a, TestResultDiff b) { return -a.priority.CompareTo(b.priority); });
            return diffs;
        }

        public static void WriteComparison(TextWriter writer, TestResults oldResults, TestResults newResults)
        {
            writer.WriteLine(oldResults.time + " -> " + newResults.time);
            if (oldResults.computerName != newResults.computerName)
                writer.WriteLine("Warning: computerName changes from " + oldResults.computerName + " to " + newResults.computerName);
            IList<TestResultDiff> comp = CompareResults(oldResults, newResults);
            foreach (TestResultDiff diff in comp)
            {
                writer.WriteLine(diff);
            }
        }

        public static void PrintCollection<T>(IEnumerable<T> list)
        {
            foreach (T item in list) Console.WriteLine(item);
        }

        public static void JitLoggingTest()
        {
            if (true)
            {
                // Collect JIT events
                // Requirements for proper operation:
                // - must compile in Release mode
                // - must run outside of Visual Studio debugger
                // - must run in elevated mode
                // - must have run JitLoggingInstall once
                InferenceEngine.DefaultEngine.Compiler.IncludeDebugInformation = false;
                //InferenceEngine.DefaultEngine.Compiler.FreeMemory = false; // this sometimes allows more inlining
                //TestUtils.JitLoggingInstall();  // must be run once
                TestUtils.JitLogging(true);
                //(new EpTests()).ExpTest();
                (new ModelTests()).PoissonRegressionTest();
                //(new IndexingTests()).JaggedSubarrayTest();
                TestUtils.JitLogging(false);
                TestUtils.JitLoggingConvert();
            }
            var list = TestUtils.ReadJitEvents();
            list = list.Where(e => e.InlineeName.StartsWith("get_Item") || e.InlineeName.StartsWith("set_Item")).ToList();
            Console.WriteLine(list.Count);
            if (list.Count < 100) Console.WriteLine(StringUtil.ToString(list));
        }

        public static string[] relevantNamespaces =
            {
                "Microsoft.ML.Probabilistic.Collections",
                "Microsoft.ML.Probabilistic.Math",
                "Microsoft.ML.Probabilistic.Utililies",
                "Microsoft.ML.Probabilistic.Distributions",
                "Microsoft.ML.Probabilistic.Factors",
                ModelBuilder.ModelNamespace
            };

        public static List<JitInliningFailedEvent> ReadJitEvents(string path = "dumpfile.xml")
        {
            List<JitInliningFailedEvent> result = new List<JitInliningFailedEvent>();
            XmlTextReader reader = new XmlTextReader(path);
            while (reader.ReadToFollowing("MethodJitInliningFailed"))
            {
                bool done = false;
                reader.Read();
                JitInliningFailedEvent ev = new JitInliningFailedEvent();
                bool relevant = false;
                while (!done)
                {
                    switch (reader.NodeType)
                    {
                        case XmlNodeType.Element:
                            string field = reader.Name;
                            string value = reader.ReadElementContentAsString();
                            //Console.WriteLine(field+": "+value);
                            if (field == "InlineeNamespace")
                            {
                                ev.InlineeNamespace = value;
                                foreach (string ns in relevantNamespaces)
                                {
                                    if (value.StartsWith(ns)) relevant = true;
                                }
                            }
                            else if (field == "InlineeName")
                            {
                                ev.InlineeName = value;
                            }
                            else if (field == "InlinerNamespace") ev.InlinerNamespace = value;
                            else if (field == "InlinerName") ev.InlinerName = value;
                            else if (field == "FailReason") ev.Reason = value;
                            break;
                        case XmlNodeType.EndElement:
                            done = true;
                            break;
                        default:
                            reader.Read();
                            break;
                    }
                }
                if (relevant) result.Add(ev);
            }
            return result;
        }

        public static void JitLogging(bool enable)
        {
            // See http://blogs.msdn.com/b/clrcodegeneration/archive/2009/05/11/jit-etw-tracing-in-net-framework-4.aspx
            string args;
            if (enable)
                args = @"start clrevents -p {e13c0d23-ccbc-4e12-931b-d9cc2eee27e4} 0x1000 5 -ets";
            else
                args = @"stop clrevents -ets";
            ExecuteCommand("logman", args);
        }

        public static void JitLoggingInstall()
        {
            ExecuteCommand("wevtutil", @"im c:\windows\microsoft.net\framework\v4.0.30319\clr-etw.man");
        }

        public static void JitLoggingConvert()
        {
            // for this to work, you must first have run JitLoggingInstall
            ExecuteCommand("tracerpt", "-y clrevents.etl");
        }

        public static void ExecuteCommand(string fileName, string arguments)
        {
            System.Diagnostics.Process proc = new System.Diagnostics.Process();
            proc.EnableRaisingEvents = false;
            proc.StartInfo.FileName = fileName;
            proc.StartInfo.CreateNoWindow = true;
            proc.StartInfo.UseShellExecute = false;
            proc.StartInfo.RedirectStandardOutput = true;
            proc.StartInfo.RedirectStandardError = true;
            proc.StartInfo.Arguments = arguments;
            proc.Start();
            proc.WaitForExit();
            if (proc.ExitCode < 0)
            {
                string result = proc.StandardOutput.ReadToEnd();
                Console.WriteLine(result);
            }
        }

        /// <summary>
        /// A test that uses this property must have Trait("Category", "ModifiesGlobals")
        /// </summary>
        public static IDisposable TemporarilyAllowGaussianImproperMessages
        {
            get
            {
                bool forceProper = Factors.GaussianOp.ForceProper;
                bool productForceProper = Factors.GaussianProductOp.ForceProper;
                return new TemporaryHelper(
                    () =>
                    {
                        Factors.GaussianOp.ForceProper = false;
                        Factors.GaussianProductOp.ForceProper = false;
                    },
                    () =>
                    {
                        Factors.GaussianOp.ForceProper = forceProper;
                        Factors.GaussianProductOp.ForceProper = productForceProper;
                    });
            }
        }

        /// <summary>
        /// A test that uses this property must have Trait("Category", "ModifiesGlobals")
        /// </summary>
        public static IDisposable TemporarilyAllowGammaImproperProducts
        {
            get { return new TemporaryHelper(() => GammaProductOp_Laplace.ForceProper = false, () => GammaProductOp_Laplace.ForceProper = true); }
        }

        /// <summary>
        /// A test that uses this property must have Trait("Category", "ModifiesGlobals")
        /// </summary>
        public static IDisposable TemporarilyAllowBetaImproperSums
        {
            get { return new TemporaryHelper(() => Beta.AllowImproperSum = true, () => Beta.AllowImproperSum = false); }
        }

        /// <summary>
        /// A test that uses this property must have Trait("Category", "ModifiesGlobals")
        /// </summary>
        public static IDisposable TemporarilyAllowDirichletImproperSums
        {
            get { return new TemporaryHelper(() => Dirichlet.AllowImproperSum = true, () => Dirichlet.AllowImproperSum = false); }
        }

        /// <summary>
        /// A test that uses this property must have Trait("Category", "ModifiesGlobals")
        /// </summary>
        public static IDisposable TemporarilyUseMeanPointGamma
        {
            get { return new TemporaryHelper(() => Factors.VariablePointOp_RpropGamma.UseMean = true, () => Factors.VariablePointOp_RpropGamma.UseMean = false); }
        }

        /// <summary>
        /// A test that uses this property must have Trait("Category", "ModifiesGlobals")
        /// </summary>
        public static IDisposable TemporarilyUnrollLoops
        {
            get { return new TemporaryHelper(() => InferenceEngine.DefaultEngine.Compiler.UnrollLoops = true, () => InferenceEngine.DefaultEngine.Compiler.UnrollLoops = false); }
        }

        /// <summary>
        /// A test that uses this method must have Trait("Category", "ModifiesGlobals")
        /// </summary>
        public static IDisposable TemporarilyChangeQuadratureNodeCount(int quadratureNodeCount)
        {
            int previousValue = ExpOp.QuadratureNodeCount;
            return new TemporaryHelper(() => ExpOp.QuadratureNodeCount = quadratureNodeCount, () => ExpOp.QuadratureNodeCount = previousValue);
        }

        /// <summary>
        /// A test that uses this method must have Trait("Category", "ModifiesGlobals")
        /// </summary>
        public static IDisposable TemporarilyChangeQuadratureShift(bool quadratureShift)
        {
            bool previousValue = ExpOp.QuadratureShift;
            return new TemporaryHelper(() => ExpOp.QuadratureShift = quadratureShift, () => ExpOp.QuadratureShift = previousValue);
        }

        private class TemporaryHelper : IDisposable
        {
            private readonly Action end;

            public TemporaryHelper(Action begin, Action end)
            {
                this.end = end;
                begin();
            }

            public void Dispose()
            {
                end();
            }
        }
    }

    public class JitInliningFailedEvent
    {
        public string InlinerNamespace, InlinerName;
        public string InlineeNamespace, InlineeName;
        public string Reason;

        public override string ToString()
        {
            return String.Format("InliningFailed: {0}.{1} in {2}.{3} ({4})", InlineeNamespace, InlineeName, InlinerNamespace, InlinerName, Reason);
        }
    }

    public class TestResults
    {
        public SortedDictionary<string, TestResult> results = new SortedDictionary<string, TestResult>();
        public string computerName;
        public DateTime time;
    }

    public class TestResult
    {
        public string outcome;
        public TimeSpan duration;
    }

    public class TestResultDiff
    {
        public string name;
        public double priority;
        public string message;

        public TestResultDiff(string name, string message, double priority)
        {
            this.name = name;
            this.message = message;
            this.priority = priority;
        }

        public override string ToString()
        {
            return name + " " + message;
        }
    }



    /// <summary>
    /// Help class for assert options that are not implemented in xUnit.
    /// </summary>
    internal class AssertHelper : Xunit.Assert
    {
        /// <summary>
        /// Compare two doubles with given precision.
        /// </summary>
        /// <param name="expected">Expected value.</param>
        /// <param name="observed">Actual value.</param>
        /// <param name="eps">Precision.</param>
        public static void Equal(double expected, double observed, double eps)
        {
            // Infinity check
            if (expected == observed)
            {
                return;
            }

            Assert.True(Math.Abs(expected - observed) < eps, $"Equality failure.\nExpected: {expected}\nActual:   {observed}");
        }

        /// <summary>
        /// Tests that two given objects are equal by using <see cref="Diffable.MaxDiff"/>.
        /// </summary>
        /// <param name="object1">The first object.</param>
        /// <param name="object2">The second object.</param>
        /// <param name="tolerance">The maximum allowed difference.</param>
        public static void Equal(Diffable object1, Diffable object2, double tolerance)
        {
            Assert.True(object1.MaxDiff(object2) <= tolerance);
        }

        /// <summary>
        /// Check if test runs longer than possible.
        /// </summary>
        /// <param name="act">Test action.</param>
        /// <param name="timeout">Timeout in milliseconds.</param>
        public static void Timeout(Action act, int timeout)
        {
            var task = Task.Run(act);
            Assert.True(task.Wait(timeout), $"Execution was longer than expected ({timeout} milliseconds)");
        }
    }



#if MONO_SUPPORT
    /// <summary>
    /// Empty stream writer provider for Mono.
    /// </summary>
    public class NullWriter : StreamWriter
    {
        private static readonly NullWriter nullWriter = new NullWriter(Stream.Null);

        public static NullWriter Instance
        {
            get { return nullWriter; }
        }

        private NullWriter(Stream stream) : base(stream) { }
        public override void Write(string value) { }
        public override void WriteLine(string value) { }
    }
#endif
}

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
    using Xunit;
    using System.Security.Cryptography;

    public static class CompilerOptionsTestUtils
    {
        public static int TestAllCompilerOptions(
            string workingDirectory,
            bool loadTestAssembly,
            bool loops,
            bool free,
            bool copies,
            bool optimise)
        {
            Directory.SetCurrentDirectory(workingDirectory);

            var assembly = Assembly.GetCallingAssembly();
            if (loadTestAssembly)
                assembly = Assembly.Load("Microsoft.ML.Probabilistic.Tests");
            var tests = TestUtils.FindTestMethods(assembly);

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

        public static async Task<int> RunProcessAsync(
            string fileName,
            string arguments,
            string workingDirectory,
            int instance,
            CancellationToken cancellationToken)
        {
            using (var process = new Process
            {
                EnableRaisingEvents = true,
                StartInfo = new ProcessStartInfo
                {
                    Arguments = arguments,
                    CreateNoWindow = true,
                    FileName = fileName,
                    RedirectStandardError = true,
                    RedirectStandardOutput = true,
                    WorkingDirectory = workingDirectory,
                    UseShellExecute = false,
                },
            })
            {
                process.ErrorDataReceived += (o, e) =>
                {
                    if (!string.IsNullOrWhiteSpace(e.Data))
                    {
                        Console.Error.WriteLine($"{instance} error: {e.Data}");
                        Console.WriteLine($"{instance} error: {e.Data}");
                    }
                };

                process.OutputDataReceived += (o, e) =>
                {
                    if (!string.IsNullOrWhiteSpace(e.Data))
                    {
                        Console.WriteLine($"{instance}: {e.Data}");
                    }
                };

                var processExited = new TaskCompletionSource<int>();
                process.Exited += (o, e) =>
                    processExited.TrySetResult(process.ExitCode);

                cancellationToken.Register(() =>
                {
                    try
                    {
                        process.Kill();
                    }
                    catch (InvalidOperationException)
                    {
                        // The process already exited.
                    }

                    // After the result has been set, the process object
                    // is disposed of -- so it is important to use it
                    // (in the call to Kill above) before setting the result
                    // here.
                    processExited.TrySetCanceled();
                });

                Console.WriteLine($"{instance}: {fileName} {process.StartInfo.Arguments}");

                var startedProcess = process.Start();

                // We set the ID so that we can stop the process if it is cancelled.
                var buildId = Environment.GetEnvironmentVariable("BUILD_BUILDID");
                Console.WriteLine($"##vso[task.setvariable variable=OptionsBuild_{buildId}_{instance}]{process.Id}");
                Assert.True(startedProcess, $"Could not start: {fileName} {arguments}");

                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                var exitCode = await processExited.Task.ConfigureAwait(false);

                return exitCode;
            }
        }

        private async static Task<Task<T>[]> ForEachParallelWithLimit<T>(int limit, Func<Task<T>>[] tasks)
        {
            var outputTaskSources = tasks.Select(_ => new TaskCompletionSource<T>()).ToArray();
            var outputTasks = outputTaskSources.Select(task => task.Task).ToArray();
            var semaphore = new SemaphoreSlim(limit, limit);
            for (int i = 0; i < tasks.Length; i++)
            {
                var index = i;
                new Thread(() =>
                {
                    semaphore.Wait();
                    try
                    {
                        var pending = tasks[index]();
                        try
                        {
                            outputTaskSources[index].SetResult(pending.Result);
                        }
                        catch (Exception e)
                        {
                            outputTaskSources[index].SetException(e);
                        }
                    }
                    finally
                    {
                        semaphore.Release();
                    }
                }).Start();
            }

            await Task.WhenAll(outputTasks);
            return outputTasks;
        }

        private static void CopyDirectory(string source, string destination)
        {
            var directoryInfo = new DirectoryInfo(source);
            Directory.CreateDirectory(Path.Combine(destination, directoryInfo.Name));

            foreach (var file in directoryInfo.GetFiles())
            {
                File.Copy(file.FullName, Path.Combine(destination, directoryInfo.Name, file.Name));
            }

            foreach (var directory in directoryInfo.GetDirectories())
            {
                CopyDirectory(directory.FullName, Path.Combine(destination, directoryInfo.Name));
            }
        }

        public static async Task LaunchTestAllCompilerOptionsProcesses()
        {
            var assembly = Assembly.GetExecutingAssembly().Location;

            var temporaryDirectories = new ConcurrentBag<string>();

            // This cancellation token is used to kill left-over processes.
            var cancellationTokenSource = new CancellationTokenSource();

            var runners = new List<Func<Task<int>>>();

            try
            {
                var processLimit = 1; // Environment.ProcessorCount;
                var semaphore = new Semaphore(processLimit, processLimit);
                var instance = 0;
                foreach (var loops in new[] { true, false })
                {
                    foreach (var free in new[] { true, false })
                    {
                        foreach (var copies in new[] { true, false })
                        {
                            foreach (var optimise in new[] { true, false })
                            {
                                var instanceIndex = instance++;
                                runners.Add(() =>
                                    Task.Run(async () =>
                                    {
                                        var temp = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
                                        temporaryDirectories.Add(temp);
                                        Directory.CreateDirectory(temp);

                                        CopyDirectory("Data", temp);

                                        return await RunProcessAsync(
                                            assembly,
                                            $"\"{temp}\" {loops} {free} {copies} {optimise}",
                                            temp,
                                            instanceIndex,
                                            cancellationTokenSource.Token);
                                    }));
                            }
                        }
                    }
                }

                var results = await ForEachParallelWithLimit(processLimit, runners.ToArray());

                foreach (var result in results)
                {
                    Assert.True(result.IsCompleted, "A process has not completed.");
                    Assert.True(result.Result == 0, "A process has failed.");
                }
            }
            finally
            {
                cancellationTokenSource.Cancel();

                // Wait a few seconds for processes to exit
                // before deleting working directories.
                await Task.Delay(3000);

                foreach (var item in temporaryDirectories)
                {
                    if (Directory.Exists(item))
                    {
                        try
                        {
                            Directory.Delete(item, recursive: true);
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine($"Could not delete temporary directory: {e}");
                        }
                    }
                }
            }
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

                return RunAllTests(TestFinished, tests).Select(m => new Tuple<string, MethodInfo>($"UseParallelForLoops={loops} FreeMemory={free} ReturnCopies={copies} OptimiseInferenceCode={optimise}: {m.error}", m.test)).ToList();
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
            RunAllTests(testFinished, testMethods);
        }

        private static void ForEach<T>(IEnumerable<T> tests, Action<T> action)
        {
            foreach (var item in tests)
            {
                action(item);
            }
        }

        private static IEnumerable<(MethodInfo test, Exception error)> RunAllTests(Action<string, TimeSpan> testFinished, MethodInfo[] tests)
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
    }
}

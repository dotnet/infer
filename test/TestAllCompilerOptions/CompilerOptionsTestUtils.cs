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

    using Xunit;

    public static class CompilerOptionsTestUtils
    {
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
                var processLimit = Environment.ProcessorCount / 2;
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

                                        var dataDirectory = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "Data");
                                        CopyDirectory(dataDirectory, temp);

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
    }
}

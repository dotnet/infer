// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests.TestAllCompilerOptions
{
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Threading.Tasks;
    using Xunit;

    class Program
    {
        static async Task<int> Main(string[] args)
        {
            try
            {
                // This is the process that launches and monitors other processes.
                if (args.Length == 0)
                {
                    // Output the process ID so that the process can be cancelled.
                    Console.WriteLine("Creating processes...");
                    var buildId = Environment.GetEnvironmentVariable("BUILD_BUILDID");
                    var processId = Process.GetCurrentProcess().Id;
                    Console.WriteLine($"##vso[task.setvariable variable=OptionsBuild_{buildId}_Host]{processId}");

                    await CompilerOptionsTestUtils.LaunchTestAllCompilerOptionsProcesses();
                    return 0;
                }

                // This is a worker process.
                var workingDirectory = args[0];
                var loops = bool.Parse(args[1]);
                var free = bool.Parse(args[2]);
                var copies = bool.Parse(args[3]);
                var optimise = bool.Parse(args[4]);
                var result = TestUtils.TestAllCompilerOptions(
                    workingDirectory,
                    loadTestAssembly: true,
                    loops: loops,
                    free: free,
                    copies: copies,
                    optimise: optimise);
                Assert.True(result == 0, "There was a failed test.");
                return 0;
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                return 1;
            }
        }
    }
}

using Loki.Shared;
using System;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.ML.Probabilistic.Tests;
using System.IO;

namespace Loki.Generated
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var logFilePath = Path.Combine(Environment.CurrentDirectory, "log.csv");
            Environment.CurrentDirectory = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Tests"));

            var tests = TestInfo.FromAssembly(typeof(Program).Assembly)
                .Where(ti => !ti.Traits.Any(trait => trait.Key == "Category" && trait.Value == "CompilerOptionsTest"))
                //.Skip(100)
                //.Take(100)
                ;

            var testRun = TestSuiteRunner.RunTestSuite(
                tests,
                new[] { 0ul },
                logFilePath,
                false,
                TimeSpan.FromMinutes(2),
                4,
                new Progress<TestSuiteRunStatus>(s => Console.Title = $"Completed {s.CompletedTests} tests. OpCounts: {string.Join(" | ", s.CurrentTestStatuses.Select(cts => cts.CurrentRunStatus.OperationCount.ToString()))}"));

            var testResultEnumerator = testRun.GetAsyncEnumerator();
            try
            {
                while (await testResultEnumerator.MoveNextAsync())
                {
                    var result = testResultEnumerator.Current;
                    Console.WriteLine($"{result.ContainingTypeFullName}.{result.TestName}, {result.StartingFuel} fuel units: {result.Outcome}\n{result.Message}");
                }
            }
            finally
            {
                await testResultEnumerator.DisposeAsync();
            }
        }
    }
}

using Loki.Shared;
using System;
using System.Threading.Tasks;
using System.Linq;

namespace Loki.Generated
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var tests = TestInfo.FromAssembly(typeof(Program).Assembly)
                //.Where(ti => !ti.Traits.Any(kvp => kvp.Key == "Category" && kvp.Value == "OpenBug"))
                //.Skip(100)
                //.Take(100)
                ;

            var testRun = TestSuiteRunner.RunTestSuite(
                tests,
                new[] { ulong.MaxValue, 0ul },
                "log.csv",
                false,
                null, // TimeSpan.FromMinutes(10),
                1,
                true,
                new Progress<TestSuiteRunStatus>(s => Console.Title = $"Completed {s.CompletedTests} tests; {s.CurrentTestStatuses[0].CompletedRuns}/{s.CurrentTestStatuses[0].TotalRuns} runs; {s.CurrentTestStatuses[0].CurrentRunStatus.OperationCount}/{s.CurrentTestStatuses[0].PerRunOperationCountEstimate} ops."));

            var testResultEnumerator = testRun.GetAsyncEnumerator();
            try
            {
                while (await testResultEnumerator.MoveNextAsync())
                {
                    var result = testResultEnumerator.Current;
                    Console.WriteLine($"{result.ContainingTypeFullName}.{result.TestName}, {result.StartingFuel} fuel units: {result.Outcome}");
                }
            }
            finally
            {
                await testResultEnumerator.DisposeAsync();
            }
        }
    }
}

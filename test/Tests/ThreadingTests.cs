// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using System.Threading;
using Microsoft.ML.Probabilistic.Math;
using System.Diagnostics;
using System.Collections.Concurrent;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Tests using Infer.NET in a multi-threaded context.
    /// </summary>
    public class ThreadingTests
    {
        [Fact]
        public void SimpleParallelThreadsTest()
        {
            int numThreads = 10;
            for (int i = 0; i < numThreads; i++)
            {
                Thread t = new Thread(new ThreadStart(new DocumentationTests().SimpleGaussianExample));
                t.Start();
            }
        }

        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void RandRestartThreadsTest()
        {
            double sum1 = RandRestartThreadsTest_Helper();
            double sum2 = RandRestartThreadsTest_Helper();
            Assert.Equal(sum1, sum2, 1e-10);
        }
        public double RandRestartThreadsTest_Helper()
        {
            // this guarantees that the set of samples is the same, but doesn't guarantee the assignment of samples to threads.
            Rand.Restart(0);
            ConcurrentDictionary<int, double> samples = new ConcurrentDictionary<int, double>();
            Thread[] threads = new Thread[10];
            for (int i = 0; i < threads.Length; i++)
            {
                Thread t = new Thread(new ThreadStart(() => {
                    int index = Thread.CurrentThread.ManagedThreadId;
                    double sample = Rand.Double();
                    samples[index] = sample;
                }));
                t.Start();
                threads[i] = t;
            }
            for (int i = 0; i < threads.Length; i++)
            {
                threads[i].Join();
            }
            double sum = 0;
            foreach(var value in samples.Values) 
            {
                sum += value;
            }
            return sum;
        }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Tutorials", "A simple first example showing the basics of Infer.NET.", Prefix = "1.")]
    public class FirstExample
    {
        public void Run()
        {
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            InferenceEngine engine = new InferenceEngine();
            if (engine.Algorithm is Algorithms.VariationalMessagePassing)
            {
                Console.WriteLine("This example does not run with Variational Message Passing");
                return;
            }
            Console.WriteLine("Probability both coins are heads: " + engine.Infer(bothHeads));
            bothHeads.ObservedValue = false;
            Console.WriteLine("Probability distribution over firstCoin: " + engine.Infer(firstCoin));
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Xunit;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Summary description for ForumTests
    /// </summary>
    
    public class ForumTests
    {
        [Fact]
        public void amin()
        {
            int X_No = 5;
            int Z_No = 3;
            Range X_Range = new Range(X_No).Named("X_Range");
            Range Z_Range = new Range(Z_No).Named("Z_Range");
            int[][] Index = new int[][]{ new int[]{ 0, 1, 3 },
                new int[]{ 1, 2, 4 },
                new int[]{ 2, 4 } };
            int[] sizes = new int[Z_No];
            for (int i = 0; i < Z_No; i++)
                sizes[i] = Index[i].Length;
            VariableArray<int> sizesVar = Variable.Constant(sizes, Z_Range);
            Range Index_Range = new Range(sizesVar[Z_Range]).Named("Index_Range");
            VariableArray<VariableArray<int>, int[][]> indexVar = Variable.Array(Variable.Array<int>(Index_Range), Z_Range).Named("indexVar");
            indexVar.ObservedValue = Index; 
            VariableArray<double> Var_X = Variable.Array<double>(X_Range).Named("Var_X");
            VariableArray<double> Var_Z = Variable.Array<double>(Z_Range);
            Var_X[X_Range] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(X_Range);
            Var_Z[Z_Range] = Variable.Sum(Variable.GetItems(Var_X, indexVar[Z_Range]));
            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(Var_Z)); 
        }

        [Fact]
        public void CoinFlips()
        {
            int threshold = 20;
            int N = 100;
            var n = new Range(N).Named("n");
            var coins = Variable.Array<bool>(n).Named("coins");
            coins[n] = Variable.Bernoulli(.5).ForEach(n);
            var RunLength = new Variable<int>[N];
            var zero = Variable.Random(Discrete.PointMass(0, N));
            RunLength[0] = Variable.Constant(0);
            var thresholdExceeded = Variable.Bernoulli(0);
            for (int i = 1; i < N; i++)
            {
                RunLength[i] = Variable.New<int>();
                using (Variable.If(coins[i - 1]))
                    RunLength[i].SetTo(RunLength[i - 1] + 1);
                using (Variable.IfNot(coins[i - 1]))
                    RunLength[i].SetTo(zero);
                thresholdExceeded |= RunLength[i] > threshold;
            }
            var ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(RunLength[10]));

        }

        [Fact]
        public void AlexJames()
        {
            // Data about tasks: true = correct, false = incorrect
            bool[] TaskA_Math = { false, true, true, true, true, true, true, true, false, false };

            // Range for A 
            int numA = TaskA_Math.Length;
            Range taskARange = new Range(numA);

            VariableArray<bool> data = Variable.Array<bool>(taskARange);

            // The prior
            var taskA_performance = Variable.Beta(1, 1);

            // The likelihood model
            data[taskARange] = Variable.Bernoulli(taskA_performance).ForEach(taskARange);

            // Attach data
            data.ObservedValue = TaskA_Math;

            // Inference engine (EP)
            InferenceEngine engine = new InferenceEngine();

            // Infer probability of Task A Math so can predict Task B_Math etc. 
            Console.WriteLine("isCorrect = {0}", engine.Infer<Beta>(taskA_performance).GetMean());

        }

        [Fact]
        public void Valmir1()
        {
            var t = Variable.Bernoulli(.6);
            Variable<bool> c = Variable.New<bool>();
            using (Variable.If(t))
                c.SetTo(Variable.Bernoulli(.3));
            using (Variable.IfNot(t))
                c.SetTo(Variable.Bernoulli(.8));
            c.ObservedValue = true;
            var ie = new InferenceEngine(new ExpectationPropagation());
            Console.WriteLine("P(T|C)=" + ie.Infer(t));
        }

        [Fact]
        public void Valmir2()
        {
            var SubContractOnTime = Variable.Bernoulli(.95);
            var StaffQualityIsGood = Variable.Bernoulli(.7);
            var Delay = Variable.New<bool>();
            using (Variable.If(SubContractOnTime))
            {
                using (Variable.If(StaffQualityIsGood))
                    Delay.SetTo(Variable.Bernoulli(.05));
                using (Variable.IfNot(StaffQualityIsGood))
                    Delay.SetTo(Variable.Bernoulli(.3));
            }
            using (Variable.IfNot(SubContractOnTime))
            {
                using (Variable.If(StaffQualityIsGood))
                    Delay.SetTo(Variable.Bernoulli(.3));
                using (Variable.IfNot(StaffQualityIsGood))
                    Delay.SetTo(Variable.Bernoulli(.99));
            }
            SubContractOnTime.ObservedValue = false;
            var ie = new InferenceEngine(new ExpectationPropagation());
            Console.WriteLine("P(Delay)=" + ie.Infer(Delay));
        }

    }
}

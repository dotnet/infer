// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Distributions.Automata;
    using System.Collections.Generic;
    using System.IO;

    /// <summary>
    /// Useful routines for testing string inference.
    /// </summary>
    public class StringInferenceTestUtilities
    {
        /// <summary>
        /// The tolerance used when comparing log-probabilities.
        /// </summary>
        private const double LogValueEps = 1e-8;

        #region Utils for testing automata

        /// <summary>
        /// Tests whether the probability of given strings under a given distribution equals to a specified value.
        /// </summary>
        /// <param name="distribution">The distribution.</param>
        /// <param name="trueProbability">The expected probability.</param>
        /// <param name="strings">The strings to test.</param>
        public static void TestProbability(StringDistribution distribution, double trueProbability, params string[] strings)
        {
            TestLogProbability(distribution, Math.Log(trueProbability), strings);
        }

        /// <summary>
        /// Tests whether the logarithm of the probability of given strings under a given distribution
        /// equals to a specified value.
        /// </summary>
        /// <param name="distribution">The distribution.</param>
        /// <param name="trueLogProbability">The expected logarithm of the probability.</param>
        /// <param name="strings">The strings to test.</param>
        public static void TestLogProbability(StringDistribution distribution, double trueLogProbability, params string[] strings)
        {
            TestLogValue(distribution.ToNormalizedAutomaton(), trueLogProbability, strings);
        }

        /// <summary>
        /// Tests whether the value of a given automaton on given strings equals to a specified value.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        /// <param name="trueValue">The expected value.</param>
        /// <param name="strings">The strings to test.</param>
        public static void TestValue(StringAutomaton automaton, double trueValue, params string[] strings)
        {
            TestLogValue(automaton, Math.Log(trueValue), strings);
        }

        /// <summary>
        /// Tests whether the logarithm of the value of a given automaton on given strings equals to a specified value.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        /// <param name="trueLogValue">The expected logarithm of the function value.</param>
        /// <param name="strings">The strings to test.</param>
        public static void TestLogValue(StringAutomaton automaton, double trueLogValue, params string[] strings)
        {
            foreach (string str in strings)
            {
                double logValue = automaton.GetLogValue(str);
                Assert.Equal(trueLogValue, logValue, LogValueEps);
                Assert.Equal(logValue, Clone(automaton).GetLogValue(str));
            }
        }

        /// <summary>
        /// Tests whether the product of given distributions is equal to another distribution on a specified strings.
        /// </summary>
        /// <param name="argument1">The first argument of the product.</param>
        /// <param name="argument2">The second argument of the product.</param>
        /// <param name="trueProduct">The true product.</param>
        /// <param name="stringsToCheckOn">The strings to test.</param>
        public static void TestProduct(
            StringDistribution argument1, StringDistribution argument2, StringDistribution trueProduct, params string[] stringsToCheckOn)
        {
            var product = new StringDistribution();
            double productLogNormalizer = product.SetToProductAndReturnLogNormalizer(argument1, argument2);
            double logAverageOf = argument1.GetLogAverageOf(argument2);
            Assert.Equal(productLogNormalizer, logAverageOf);
            Assert.Equal(logAverageOf, Clone(argument1).GetLogAverageOf(argument2));

            if (trueProduct.IsZero())
            {
                Assert.True(product.IsZero());
                Assert.True(double.IsNegativeInfinity(productLogNormalizer));
            }
            else if (trueProduct.IsPointMass)
            {
                Assert.True(product.IsPointMass);
                Assert.Equal(product.Point, trueProduct.Point);
            }
            else if (trueProduct.IsUniform())
            {
                Assert.True(product.IsUniform());
            }
            else
            {
                Assert.False(product.IsZero());
                Assert.False(product.IsPointMass);
                Assert.False(product.IsUniform());

                Assert.Equal(trueProduct.IsProper(), product.IsProper());

                foreach (var str in stringsToCheckOn)
                {
                    double logProb1 = argument1.GetLogProb(str);
                    double logProb2 = argument2.GetLogProb(str);
                    double logProbProduct = trueProduct.GetLogProb(str);

                    if (double.IsNegativeInfinity(logProb1) || double.IsNegativeInfinity(logProb2))
                    {
                        Assert.True(double.IsNegativeInfinity(logProbProduct));
                    }
                    else if (double.IsNegativeInfinity(logProbProduct))
                    {
                        Assert.True(double.IsNegativeInfinity(logProb1) || double.IsNegativeInfinity(logProb2));
                    }
                    else
                    {
                        Assert.Equal(logProb1 + logProb2, logProbProduct + productLogNormalizer, LogValueEps);
                    }
                }
            }
        }

        /// <summary>
        /// Tests whether the support of a given distribution excludes given values.
        /// </summary>
        /// <param name="distribution">The distribution.</param>
        /// <param name="values">The values to test.</param>
        public static void TestIfExcludes<T>(IDistribution<T> distribution, params T[] values)
        {
            TestInclusionExclusion(distribution, values, false);
        }

        /// <summary>
        /// Tests whether the support of a given distribution includes given values.
        /// </summary>
        /// <param name="distribution">The distribution.</param>
        /// <param name="values">The values to test.</param>
        public static void TestIfIncludes<T>(IDistribution<T> distribution, params T[] values)
        {
            TestInclusionExclusion(distribution, values, true);
        }

        public static void TestAutomatonPropertyPreservation(StringAutomaton automaton, Func<StringAutomaton, StringAutomaton> testedOperation)
        {
            var automatonWithClearProperties = automaton
                .WithLogValueOverride(null)
                .WithPruneStatesWithLogEndWeightLessThan(null);

            var outputForClearProperties = testedOperation(automatonWithClearProperties);

            Assert.Null(outputForClearProperties.LogValueOverride);
            Assert.Null(outputForClearProperties.PruneStatesWithLogEndWeightLessThan);

            var automatonWithSetProperties = automaton
                .WithLogValueOverride(-1)
                .WithPruneStatesWithLogEndWeightLessThan(-128);

            var outputForSetProperties = testedOperation(automatonWithSetProperties);

            Assert.Equal(automatonWithSetProperties.LogValueOverride, outputForSetProperties.LogValueOverride);
            Assert.Equal(automatonWithSetProperties.PruneStatesWithLogEndWeightLessThan, outputForSetProperties.PruneStatesWithLogEndWeightLessThan);
        }

        #endregion

        #region Utils for testing transducers

        /// <summary>
        /// Tests if a projection of a given distribution onto a given transducer has the desired value on a given sequence.
        /// </summary>
        /// <param name="transducer">The transducer.</param>
        /// <param name="input">The distribution to project.</param>
        /// <param name="str">The sequence to test the projection on.</param>
        /// <param name="trueValue">The desired value of the projection on the sequence.</param>
        public static void TestTransducerProjection(
            StringTransducer transducer, StringDistribution input, string str, double trueValue)
        {
            TestTransducerProjection(transducer, input.ToNormalizedAutomaton(), str, trueValue);
        }

        /// <summary>
        /// Tests if a projection of a given automaton onto a given transducer has the desired value on a given sequence.
        /// </summary>
        /// <param name="transducer">The transducer.</param>
        /// <param name="input">The automaton to project.</param>
        /// <param name="str">The sequence to test the projection on.</param>
        /// <param name="trueValue">The desired value of the projection on the sequence.</param>
        public static void TestTransducerProjection(
            StringTransducer transducer, StringAutomaton input, string str, double trueValue)
        {
            // Test ProjectSource(func)
            var result = transducer.ProjectSource(input);
            TestValue(result, trueValue, str);
        }

        /// <summary>
        /// Tests if a transducer has the desired value on a given pair of strings.
        /// </summary>
        /// <param name="transducer">The transducer.</param>
        /// <param name="string1">The first string.</param>
        /// <param name="string2">The second string.</param>
        /// <param name="trueValue">The desired value of the transducer on the strings.</param>
        public static void TestTransducerValue(
            StringTransducer transducer, string string1, string string2, double trueValue)
        {
            // Test ProjectSource(sequence)
            var result = transducer.ProjectSource(string1);
            TestValue(result, trueValue, string2);

            // Test ProjectSource(func)
            TestTransducerProjection(transducer, StringAutomaton.ConstantOn(1.0, string1), string2, trueValue);

            // Test GetValue()
            double value = transducer.GetValue(string1, string2);
            Assert.Equal(trueValue, value, LogValueEps);
        }

        /// <summary>
        /// Tests if a transducer projects each of given strings to a zero function.
        /// </summary>
        /// <param name="transducer">The transducer.</param>
        /// <param name="strings">The strings to test.</param>
        public static void TestIfTransducerRejects(StringTransducer transducer, params string[] strings)
        {
            foreach (string str in strings)
            {
                var res = transducer.ProjectSource(StringAutomaton.ConstantOn(1.0, str));
                Assert.True(res.IsZero());
                var res2 = transducer.ProjectSource(str);
                Assert.True(res2.IsZero());
            }
        }

        #endregion

        #region Utils helpful for computing actual sequence probabilities

        /// <summary>
        /// Computes the probability assigned to a string by a uniform distribution over strings
        /// of length from <paramref name="minLength"/> to <paramref name="maxLength"/>
        /// with alphabet size <paramref name="alphabetSize"/>.
        /// </summary>
        /// <param name="minLength">The minimum string length.</param>
        /// <param name="maxLength">The maximum string length.</param>
        /// <param name="alphabetSize">The alphabet size.</param>
        /// <returns>The probability of a string under the distribution.</returns>
        public static double StringUniformProbability(int minLength, int maxLength, int alphabetSize)
        {
            return 1.0 / (GeometricSeriesSum(maxLength, alphabetSize) - GeometricSeriesSum(minLength - 1, alphabetSize));
        }

        /// <summary>
        /// Computes the geometric sum
        /// <paramref name="p"/> + <paramref name="p"/>^2 + ... + <paramref name="p"/>^<paramref name="n"/>.
        /// </summary>
        /// <param name="n">The number of terms in the geometric sum.</param>
        /// <param name="p">The first term in the geometric sum.</param>
        /// <returns>The value of the sum.</returns>
        public static double GeometricSeriesSum(int n, double p)
        {
            return (1 - Math.Pow(p, n + 1)) / (1 - p);
        }

        /// <summary>
        /// Computes the number of ways to split a given number
        /// into a sum of no more than <paramref name="maxSumTermCount"/> non-negative numbers.
        /// </summary>
        /// <param name="number">The number to split.</param>
        /// <param name="maxSumTermCount">The maximum number of sum terms.</param>
        /// <returns>The number of ways to split the given number in the described way.</returns>
        public static int Partitions(int number, int maxSumTermCount)
        {
            int[,] table = new int[number + 1, maxSumTermCount + 1];
            for (int k = 1; k <= maxSumTermCount; ++k)
            {
                table[0, k] = 1;
                for (int n = 1; n <= number; ++n)
                {
                    table[n, k] += table[n - 1, k] + table[n, k - 1];
                }
            }

            return table[number, maxSumTermCount];
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Tests whether the support of a given distribution includes or excludes given values.
        /// </summary>
        /// <param name="distribution">The distribution.</param>
        /// <param name="values">The values to test.</param>
        /// <param name="testInclusion">Specifies whether the inclusion test must be performed instead of the exclusion test.</param>
        private static void TestInclusionExclusion<T>(IDistribution<T> distribution, T[] values, bool testInclusion)
        {
            Console.WriteLine("Testing distribution: " + distribution.ToString());
            foreach (var val in values)
            {
                var excluded = double.IsNegativeInfinity(distribution.GetLogProb(val));
                var msg = val + (excluded ? " was not in " : " was in ");
                Assert.True(excluded != testInclusion, msg + " " + distribution.ToString());
                Console.WriteLine(msg + "distribution");
            }
        }

        /// <summary>
        /// Clones an automaton by consecutively serializing and deserializing it.
        /// </summary>
        /// <param name="automaton">Automaton to clone.</param>
        /// <returns>Cloned automaton.</returns>
        public static StringAutomaton Clone(StringAutomaton automaton)
        {
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            using (var reader = new BinaryReader(ms))
            {
                automaton.Write(writer);
                writer.Flush();
                ms.Position = 0;
                var clone = StringAutomaton.Read(reader);
                return clone;
            }
        }

        /// <summary>
        /// Clones a string distribution by consecutively serializing and deserializing
        /// its automaton representation.
        /// </summary>
        /// <param name="stringDistribution">Distribution to clone.</param>
        /// <returns>Cloned distribution.</returns>
        public static StringDistribution Clone(StringDistribution stringDistribution)
        {
            if (stringDistribution.IsPointMass) return StringDistribution.String(stringDistribution.Point);
            var clonedAutomaton = Clone(stringDistribution.ToAutomaton());
            var dist = new StringDistribution();
            dist.SetWeightFunction(clonedAutomaton);
            return dist;
        }
        #endregion
    }
}

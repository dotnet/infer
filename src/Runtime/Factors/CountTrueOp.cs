// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using Distributions;
    using Factors.Attributes;
    using Utilities;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CountTrueOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "CountTrue")]
    [Quality(QualityBand.Preview)]
    [Buffers("PoissonBinomialTable")]
    public static class CountTrueOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CountTrueOp"]/message_doc[@name="PoissonBinomialTableInit(IList{Bernoulli})"]/*'/>
        public static double[,] PoissonBinomialTableInit(IList<Bernoulli> array)
        {
            return PoissonBinomialTable(array);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CountTrueOp"]/message_doc[@name="PoissonBinomialTable(IList{Bernoulli})"]/*'/>
        [Fresh]
        public static double[,] PoissonBinomialTable(IList<Bernoulli> array)
        {
            return PoissonBinomialForwardPass(array);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CountTrueOp"]/message_doc[@name="CountAverageConditional(double[,])"]/*'/>
        /// <remarks><para>
        /// Marginal distribution of count is known as Poisson Binomial.
        /// It can be found in O(n^2) time using dynamic programming, where n is the length of the array.
        /// </para></remarks>
        public static Discrete CountAverageConditional(double[,] poissonBinomialTable)
        {
            int tableSize = poissonBinomialTable.GetLength(0);
            return new Discrete(Util.ArrayInit(tableSize, i => poissonBinomialTable[tableSize - 1, i]));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CountTrueOp"]/message_doc[@name="ArrayAverageConditional{TBernoulliArray}(TBernoulliArray, Discrete, double[,], TBernoulliArray)"]/*'/>
        /// <remarks><para>
        /// Poison Binomial for a given list of Bernoulli random variables with one variable excluded can be computed
        /// in linear time given Poisson Binomial for the whole list of items computed in a forward pass, as well as a special table
        /// containing averaged Poisson Binomial, which can be computed in a backward pass.
        /// Both tables can be computed in O(n^2) time, where n is a size of the list, so the time complexity of this message operator
        /// is also O(n^2).
        /// </para></remarks>
        /// <typeparam name="TBernoulliArray">The type of messages from/to 'array'.</typeparam>
        public static TBernoulliArray ArrayAverageConditional<TBernoulliArray>(
            TBernoulliArray array, [SkipIfUniform] Discrete count, double[,] poissonBinomialTable, TBernoulliArray result)
            where TBernoulliArray : IList<Bernoulli>
        {
            int i = array.Count - 1;
            foreach (double[] backwardPassTableRow in AveragedPoissonBinomialBackwardPassTableRows(array, count))
            {
                double probTrue = 0, probFalse = 0;
                for (int j = 0; j <= i; ++j)
                {
                    probTrue += poissonBinomialTable[i, j] * backwardPassTableRow[j + 1];
                    probFalse += poissonBinomialTable[i, j] * backwardPassTableRow[j];
                }

                Debug.Assert(probTrue + probFalse > 1e-10, "The resulting distribution should be well-defined.");
                result[i] = new Bernoulli(probTrue / (probFalse + probTrue));

                --i;
            }

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CountTrueOp"]/message_doc[@name="ArrayAverageConditional{TBernoulliArray}(TBernoulliArray, int, double[,], TBernoulliArray)"]/*'/>
        /// <typeparam name="TBernoulliArray">The type of messages from/to 'array'.</typeparam>
        public static TBernoulliArray ArrayAverageConditional<TBernoulliArray>(
            TBernoulliArray array, int count, double[,] poissonBinomialTable, TBernoulliArray result)
            where TBernoulliArray : IList<Bernoulli>
        {
            Discrete mass = Discrete.PointMass(count, array.Count + 1);
            return ArrayAverageConditional(array, mass, poissonBinomialTable, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CountTrueOp"]/message_doc[@name="LogEvidenceRatio(Discrete)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Discrete count)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CountTrueOp"]/message_doc[@name="LogEvidenceRatio(int, double[,])"]/*'/>
        public static double LogEvidenceRatio(int count, double[,] poissonBinomialTable)
        {
            int tableSize = poissonBinomialTable.GetLength(0);
            return Math.Log(poissonBinomialTable[tableSize - 1, count]);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CountTrueOp"]/message_doc[@name="LogEvidenceRatio(int, bool[])"]/*'/>
        public static double LogEvidenceRatio(int count, bool[] array)
        {
            return (count == Factor.CountTrue(array)) ? 0.0 : double.NegativeInfinity;
        }

        #region Helpers

        /// <summary>
        /// Compute Poisson Binomial table for a given list of Bernoulli-distributed random variables in a forward pass.
        /// </summary>
        /// <param name="array">List of Bernoulli random variables.</param>
        /// <returns>Table A, such that A[i, j] = P(sum(<paramref name="array"/>[1], ..., <paramref name="array"/>[i]) = j).</returns>
        private static double[,] PoissonBinomialForwardPass(IList<Bernoulli> array)
        {
            var result = new double[array.Count + 1, array.Count + 1];

            result[0, 0] = 1.0;
            for (int i = 1; i <= array.Count; ++i)
            {
                double probTrue = array[i - 1].GetProbTrue();
                result[i, 0] = result[i - 1, 0] * (1 - probTrue);
                for (int j = 1; j <= i; ++j)
                {
                    result[i, j] = (result[i - 1, j] * (1 - probTrue)) + (result[i - 1, j - 1] * probTrue);
                }
            }

            return result;
        }

        /// <summary>
        /// Enumerate rows of averaged Poisson Binomial table for a given list of Bernoulli-distributed random variables in a backward pass.
        /// </summary>
        /// <param name="array">List of Bernoulli random variables.</param>
        /// <param name="averager">Distribution over sum of values in <paramref name="array"/> used to average the Poisson Binomial.</param>
        /// <returns>
        /// <para>
        /// Rows of table A, such that A[i, j] =
        /// \sum_c P(<paramref name="averager"/> = c) P(sum(<paramref name="array"/>[i + 1], ..., <paramref name="array"/>[n]) + j = c).
        /// Rows are returned last-to-first, the very first row is omitted.
        /// </para>
        /// </returns>
        private static IEnumerable<double[]> AveragedPoissonBinomialBackwardPassTableRows(IList<Bernoulli> array, Discrete averager)
        {
            Debug.Assert(averager.Dimension == array.Count + 1, "'averager' should represent a distribution over the sum of the elements of 'array'.");
            var prevRow = new double[array.Count + 1];
            var currentRow = new double[array.Count + 1];
            for (int j = 0; j <= array.Count; ++j)
            {
                currentRow[j] = averager[j];
            }

            for (int i = array.Count - 1; i >= 0; --i)
            {
                yield return currentRow;
                double[] temp = currentRow;
                currentRow = prevRow;
                prevRow = temp;

                double probTrue = array[i].GetProbTrue();
                currentRow[array.Count] = prevRow[array.Count] * (1 - probTrue);
                for (int j = array.Count - 1; j >= 0; --j)
                {
                    currentRow[j] = (prevRow[j] * (1 - probTrue)) + (prevRow[j + 1] * probTrue);
                }
            }
        }

        #endregion
    }
}

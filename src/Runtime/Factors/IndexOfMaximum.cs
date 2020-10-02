// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Collections;
    using Distributions;
    using Math;
    using Attributes;
    using Utilities;
    using GaussianArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;

    /// <summary>
    /// Holds messages for IndexOfMaximumOp
    /// </summary>
    public class IndexOfMaximumBuffer
    {
        public IList<Gaussian> MessagesToMax;
        public IList<Gaussian> to_list;
    }


    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumOp"]/doc/*'/>
    [FactorMethod(typeof(MMath), "IndexOfMaximumDouble")]
    [Quality(QualityBand.Experimental)]
    [Buffers("Buffer")]
    public static class IndexOfMaximumOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumOp"]/message_doc[@name="BufferInit{GaussianList}(GaussianList)"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static IndexOfMaximumBuffer BufferInit<GaussianList>([IgnoreDependency] GaussianList list)
            where GaussianList : IList<Gaussian>
        {
            return new IndexOfMaximumBuffer
            {
                MessagesToMax = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray()),
                to_list = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray())
            };
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumOp"]/message_doc[@name="Buffer{GaussianList}(IndexOfMaximumBuffer, GaussianList, int)"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static IndexOfMaximumBuffer Buffer<GaussianList>(
            IndexOfMaximumBuffer Buffer, GaussianList list, int IndexOfMaximumDouble) // redundant parameters required for correct dependency graph
            where GaussianList : IList<Gaussian>
        {
            var max_marginal = Buffer.to_list[IndexOfMaximumDouble] * list[IndexOfMaximumDouble];
            Gaussian product = Gaussian.Uniform();
            //var order = Rand.Perm(list.Count); 
            for (int i = 0; i < list.Count; i++)
            {
                //int c = order[i]; 
                int c = i;
                if (c != IndexOfMaximumDouble)
                {
                    var msg_to_sum = max_marginal / Buffer.MessagesToMax[c];

                    var msg_to_positiveop = DoublePlusOp.AAverageConditional(Sum: msg_to_sum, b: list[c]);
                    var msgFromPositiveOp = IsPositiveOp.XAverageConditional(true, msg_to_positiveop);
                    Buffer.MessagesToMax[c] = DoublePlusOp.SumAverageConditional(list[c], msgFromPositiveOp);
                    Buffer.to_list[c] = DoublePlusOp.AAverageConditional(Sum: msg_to_sum, b: msgFromPositiveOp);
                    max_marginal = msg_to_sum * Buffer.MessagesToMax[c];
                    product.SetToProduct(product, Buffer.MessagesToMax[c]);
                }
            }
            //Buffer.to_list[IndexOfMaximumDouble] = max_marginal / list[IndexOfMaximumDouble];
            Buffer.to_list[IndexOfMaximumDouble] = product;
            return Buffer;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumOp"]/message_doc[@name="listAverageConditional{GaussianList}(IndexOfMaximumBuffer, GaussianList, int)"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static GaussianList listAverageConditional<GaussianList>(
            IndexOfMaximumBuffer Buffer, GaussianList to_list, int IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            to_list.SetTo(Buffer.to_list);
            return to_list;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumOp"]/message_doc[@name="LogAverageFactor{GaussianList}(IndexOfMaximumBuffer, GaussianList, int)"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static double LogAverageFactor<GaussianList>(IndexOfMaximumBuffer Buffer, GaussianList list, int IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            double evidence = 0;
            var max_marginal = list[IndexOfMaximumDouble] * Buffer.to_list[IndexOfMaximumDouble];
            for (int c = 0; c < list.Count; c++)
            {
                if (c != IndexOfMaximumDouble)
                {
                    var msg_to_sum = max_marginal / Buffer.MessagesToMax[c];
                    var msg_to_positiveop = DoublePlusOp.AAverageConditional(Sum: msg_to_sum, b: list[c]);
                    evidence += IsPositiveOp.LogEvidenceRatio(true, msg_to_positiveop);
                    // sum operator does not contribute because no projection is involved
                    // the x[index]-x[c] variable does not contribute because it only connects to two factors
                    evidence -= msg_to_sum.GetLogAverageOf(Buffer.MessagesToMax[c]);
                    if (max_marginal.IsPointMass)
                        evidence += Buffer.MessagesToMax[c].GetLogAverageOf(max_marginal);
                    else
                        evidence -= Buffer.MessagesToMax[c].GetLogNormalizer();
                }
            }
            //evidence += ReplicateOp.LogEvidenceRatio<Gaussian>(MessagesToMax, list[IndexOfMaximumDouble], MessagesToMax.Select(o => max_marginal / o).ToArray());
            if (!max_marginal.IsPointMass)
                evidence += max_marginal.GetLogNormalizer() - list[IndexOfMaximumDouble].GetLogNormalizer();
            //evidence -= Buffer.MessagesToMax.Sum(o => o.GetLogNormalizer());
            //evidence -= Buffer.MessagesToMax.Sum(o => (max_marginal / o).GetLogAverageOf(o));
            return evidence;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumOp"]/message_doc[@name="LogEvidenceRatio{GaussianList}(IndexOfMaximumBuffer, GaussianList, int)"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static double LogEvidenceRatio<GaussianList>(IndexOfMaximumBuffer Buffer, GaussianList list, int IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            return LogAverageFactor(Buffer, list, IndexOfMaximumDouble);
        }
    }


    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumStochasticOp"]/doc/*'/>
    [FactorMethod(typeof(MMath), "IndexOfMaximumDouble", Default = true)]
    [Quality(QualityBand.Experimental)]
    [Buffers("Buffers")]
    public static class IndexOfMaximumStochasticOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumStochasticOp"]/message_doc[@name="BuffersInit{GaussianList}(GaussianList, Discrete)"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static IndexOfMaximumBuffer[] BuffersInit<GaussianList>([IgnoreDependency] GaussianList list, [IgnoreDependency] Discrete IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            return list.Select(o => IndexOfMaximumOp.BufferInit(list)).ToArray();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumStochasticOp"]/message_doc[@name="Buffers{GaussianList}(IndexOfMaximumBuffer[], GaussianList, Discrete)"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        [Fresh]
        public static IndexOfMaximumBuffer[] Buffers<GaussianList>(IndexOfMaximumBuffer[] Buffers, [SkipIfUniform] GaussianList list, [IgnoreDependency] Discrete IndexOfMaximumDouble)
            where GaussianList : IList<Gaussian>
        {
            for (int i = 0; i < list.Count; i++)
            {
                Buffers[i].to_list = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray());
                Buffers[i].to_list[i] = Buffers[i].MessagesToMax.Aggregate((p, q) => p * q);
                Buffers[i] = IndexOfMaximumOp.Buffer(Buffers[i], list, i);
            }
            return Buffers;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumStochasticOp"]/message_doc[@name="ListAverageConditional{GaussianList}(IndexOfMaximumBuffer[], GaussianList, Discrete, GaussianList)"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static GaussianList ListAverageConditional<GaussianList>(
            [SkipIfUniform] IndexOfMaximumBuffer[] Buffers, GaussianList list, [SkipIfUniform] Discrete IndexOfMaximumDouble, GaussianList result)
            where GaussianList : DistributionStructArray<Gaussian, double> // IList<Gaussian>
        {
            int count = list.Count;
            int[] indices = Util.ArrayInit(count, i => i);
            // TODO: check if Index is a point mass
            var enterPartial = Util.ArrayInit(count, i => Util.ArrayInit(count, j => Buffers[j].to_list[i]));
            for (int i = 0; i < count; i++)
            {
                result[i] = GateEnterPartialOp<double>.ValueAverageConditional(enterPartial[i], IndexOfMaximumDouble, list[i], indices, result[i]);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumStochasticOp"]/message_doc[@name="IndexOfMaximumDoubleAverageConditional{GaussianList}(GaussianList, IndexOfMaximumBuffer[])"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static Discrete IndexOfMaximumDoubleAverageConditional<GaussianList>(GaussianList list, IndexOfMaximumBuffer[] Buffers)
            where GaussianList : DistributionStructArray<Gaussian, double>
        {
            // var results = list.Select(o => list.Select(p => Gaussian.Uniform()).ToList()).ToArray();
            // TODO: if IndexOfMaximumDouble is uniform we will never call this routine so buffers will not get set, so messages to IndexOfMaximumDouble will be incorrect
            var evidences = new double[list.Count];
            for (int i = 0; i < list.Count; i++)
            {
                //var res = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray());
                //res[i] = Buffer[i].MessagesToMax.Aggregate((p, q) => p * q);
                evidences[i] = IndexOfMaximumOp.LogAverageFactor(Buffers[i], list, i);
            }
            return new Discrete(MMath.Softmax(evidences));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumStochasticOp"]/message_doc[@name="LogAverageFactor{GaussianList}(GaussianList, GaussianList, IndexOfMaximumBuffer[], Discrete)"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static double LogAverageFactor<GaussianList>(
            [SkipIfUniform] GaussianList list, GaussianList to_list, IndexOfMaximumBuffer[] Buffers, Discrete IndexOfMaximumDouble)
            where GaussianList : DistributionStructArray<Gaussian, double>
        {
            var evidences = new double[list.Count];
            var tempBuffer = new IndexOfMaximumBuffer();
            for (int i = 0; i < list.Count; i++)
            {
                tempBuffer.to_list = new DistributionStructArray<Gaussian, double>(list.Select(o => Gaussian.Uniform()).ToArray());
                tempBuffer.to_list[i] = Buffers[i].MessagesToMax.Aggregate((p, q) => p * q);
                tempBuffer.MessagesToMax = new DistributionStructArray<Gaussian, double>(Buffers[i].MessagesToMax.Select(o => (Gaussian)o.Clone()).ToArray());
                evidences[i] = IndexOfMaximumOp.LogAverageFactor(tempBuffer, list, i) + IndexOfMaximumDouble.GetLogProb(i);
                ;
            }
            return MMath.LogSumExp(evidences);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumStochasticOp"]/message_doc[@name="LogEvidenceRatio{GaussianList}(GaussianList, GaussianList, Discrete, Discrete, IndexOfMaximumBuffer[])"]/*'/>
        /// <typeparam name="GaussianList">The type of an incoming message from <c>list</c>.</typeparam>
        public static double LogEvidenceRatio<GaussianList>(
            GaussianList list, GaussianList to_list, Discrete IndexOfMaximumDouble, Discrete to_IndexOfMaximumDouble, IndexOfMaximumBuffer[] Buffers)
            where GaussianList : DistributionStructArray<Gaussian, double>
        {
            return LogAverageFactor(list, to_list, Buffers, IndexOfMaximumDouble) - IndexOfMaximumDouble.GetLogAverageOf(to_IndexOfMaximumDouble);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumOp_Fast"]/doc/*'/>
    [FactorMethod(typeof(MMath), "IndexOfMaximumDouble", Default = false)]
    [Quality(QualityBand.Experimental)]
    public static class IndexOfMaximumOp_Fast
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumOp_Fast"]/message_doc[@name="IndexOfMaximumDoubleAverageConditional(IList{Gaussian}, Discrete)"]/*'/>
        public static Discrete IndexOfMaximumDoubleAverageConditional([SkipIfAnyUniform, Proper] IList<Gaussian> list, Discrete result)
        {
            // Fast approximate calculation of downward message

            if (list.Count <= 1)
                return result;
            Gaussian[] maxOfOthers = new Gaussian[list.Count];
            MaxOfOthersOp.MaxOfOthers(list, maxOfOthers);
            Vector probs = result.GetWorkspace();
            for (int i = 0; i < list.Count; i++)
            {
                probs[i] = ProbGreater(list[i], maxOfOthers[i]).GetProbTrue();
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IndexOfMaximumOp_Fast"]/message_doc[@name="IndexOfMaximumDoubleAverageConditional(IList{Gaussian}, Discrete)"]/*'/>
        public static Discrete IndexOfMaximumDoubleAverageConditional2(IList<Gaussian> list, Discrete result)
        {
            // Fast approximate calculation of downward message

            // TODO: sort list first
            // best accuracy is achieved by processing in decreasing order of means
            Gaussian max = list[0];
            Vector probs = result.GetWorkspace();
            probs[0] = 1.0;
            for (int i = 1; i < list.Count; i++)
            {
                Gaussian A = max;
                Gaussian B = list[i];
                double pMax = ProbGreater(A, B).GetProbTrue();
                for (int j = 0; j < i; j++)
                {
                    probs[j] *= pMax;
                }
                probs[i] = 1 - pMax;
                max = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), A, B);
            }
            result.SetProbs(probs);
            return result;
        }

        /// <summary>
        /// Returns the probability that A>B
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        private static Bernoulli ProbGreater(Gaussian A, Gaussian B)
        {
            Gaussian diff = DoublePlusOp.AAverageConditional(Sum: A, b: B);
            return IsPositiveOp.IsPositiveAverageConditional(diff);
        }
    }

    [FactorMethod(typeof(Factor), "MaxOfOthers", Default = true)]
    [Quality(QualityBand.Experimental)]
    public static class MaxOfOthersOp
    {
        public static ArrayType MaxOfOthersAverageConditional<ArrayType>(ArrayType maxOfOthers, [SkipIfAnyUniform] IList<Gaussian> array, ArrayType result)
            where ArrayType : IList<Gaussian>, SettableToUniform
        {
            if (!maxOfOthers.IsUniform())
                throw new ArgumentException("Incoming message from maxOfOthers is not uniform.  The child of this factor must be barren.");
            if (array.Count != result.Count)
                throw new ArgumentException($"array.Count ({array.Count}) != result.Count ({result.Count})");
            MaxOfOthers(array, result);
            return result;
        }

        public static ArrayType ArrayAverageConditional<ArrayType>([SkipIfAllUniform] IList<Gaussian> maxOfOthers, ArrayType array, ArrayType result)
            where ArrayType : IList<Gaussian>
        {
            throw new NotImplementedException();
        }

        public static void MaxOfOthers(IList<Gaussian> array, IList<Gaussian> result)
        {
            var array2 = array.ToArray();
            var indices = Util.ArrayInit(array2.Length, i => i);
            // sort in descending order
            Array.Sort(array2, indices, Comparer<Gaussian>.Create((a, b) => b.GetMean().CompareTo(a.GetMean())));
            MaxOfOthers_Quadratic(array2, result);
            // put result into the original order
            Unpermute(result, indices);
        }

        public static void Unpermute<T>(IList<T> array, int[] indices)
        {
            for (int i = 0; i < array.Count; i++)
            {
                int j = indices[i];
                while (j != i)
                {
                    // swap array[i] with array[j], and their corresponding indices
                    T temp = array[i];
                    array[i] = array[j];
                    array[j] = temp;
                    int k = indices[j];
                    indices[i] = k;
                    indices[j] = j;
                    j = k;
                }
            }
        }

        public static void MaxOfOthers_MonteCarlo(IList<Gaussian> array, IList<Gaussian> result)
        {
            if (array.Count == 0)
                return;
            if (array.Count == 1)
            {
                result[0] = Gaussian.Uniform();
                return;
            }
            int iterCount = 1000000;
            double[] x = new double[array.Count];
            var est = new ArrayEstimator<GaussianEstimator, IList<Gaussian>, Gaussian, double>(array.Count, i => new GaussianEstimator());
            for (int iter = 0; iter < iterCount; iter++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    x[i] = array[i].Sample();
                }
                double[] maxOfOthers = Factor.MaxOfOthers(x);
                est.Add(maxOfOthers);
            }
            est.GetDistribution(result);
        }

        public static void MaxOfOthers_Quadratic(IList<Gaussian> array, IList<Gaussian> result)
        {
            if (array.Count == 0)
                return;
            if (array.Count == 1)
            {
                result[0] = Gaussian.Uniform();
                return;
            }
            var maxBefore = Gaussian.Uniform();
            for (int i = 0; i < array.Count-1; i++)
            {
                // maxBefore is max(array[..i-1])
                Gaussian maxBeforeAndNext, maxBeforeAndCurrent;
                if (i == 0)
                {
                    maxBeforeAndNext = array[i + 1];
                    maxBeforeAndCurrent = array[i];
                }
                else
                {
                    maxBeforeAndNext = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), maxBefore, array[i + 1]);
                    maxBeforeAndCurrent = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), maxBefore, array[i]);
                }
                // maxBeforeAndNext is max(array[..i-1], array[i+1])
                var max = maxBeforeAndNext;
                for (int j = i+2; j < array.Count; j++)
                {
                    // max is max(array[..j-1] except i)
                    max = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), max, array[j]);
                }
                result[i] = max;
                maxBefore = maxBeforeAndCurrent;
            }
            result[array.Count - 1] = maxBefore;
        }

        public static void MaxOfOthers_Linear(IList<Gaussian> array, IList<Gaussian> result)
        {
            if (array.Count == 0)
                return;
            if (array.Count == 1)
            {
                result[0] = Gaussian.Uniform();
                return;
            }
            var maxBefore = result;
            // initialize maxBefore[i] to max(array[0..i-1])
            maxBefore[1] = array[0];
            for (int i = 2; i < array.Count; i++)
            {
                maxBefore[i] = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), maxBefore[i - 1], array[i - 1]);
            }
            Gaussian maxAfter = array[array.Count - 1];
            for (int i = array.Count - 2; i >= 1; i--)
            {
                result[i] = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), maxAfter, maxBefore[i]);
                maxAfter = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), maxAfter, array[i]);
            }
            result[0] = maxAfter;
        }
    }
}

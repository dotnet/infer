// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using Factors.Attributes;

    /// <summary>
    /// Marker interface for classes which wrap distributions
    /// </summary>
    public interface IsDistributionWrapper
    {
    }

    /// <summary>
    /// An Accumulator that adds each element to a collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class AccumulateIntoCollection<T> : Accumulator<T>
    {
        public ICollection<T> Collection;

        public AccumulateIntoCollection(ICollection<T> collection)
        {
            this.Collection = collection;
        }

        public void Add(T item)
        {
            Collection.Add(item);
        }

        public void Clear()
        {
            Collection.Clear();
        }
    }

    /// <summary>
    /// Sample List
    /// </summary>
    /// <typeparam name="T">Domain type for sample list</typeparam>
    public class SampleList<T> : Accumulator<T>
    {
        private List<T> samples = new List<T>();

        /// <summary>
        /// Samples
        /// </summary>
        public IList<T> Samples
        {
            get { return samples.AsReadOnly(); }
        }

        /// <summary>
        /// Add a sample to the sample list
        /// </summary>
        /// <param name="item"></param>
        public void Add(T item)
        {
            samples.Add(item);
        }

        /// <summary>
        /// Clears out all the samples
        /// </summary>
        public void Clear()
        {
            samples.Clear();
        }
    }

    /// <summary>
    /// Conditional List
    /// </summary>
    /// <typeparam name="TDist">Distribution type for conditional list</typeparam>
    public class ConditionalList<TDist> : Accumulator<TDist>
        where TDist : ICloneable
    {
        private List<TDist> conditionals = new List<TDist>();

        /// <summary>
        /// Samples
        /// </summary>
        public IList<TDist> Conditionals
        {
            get { return conditionals.AsReadOnly(); }
        }

        /// <summary>
        /// Add a sample to the sample list
        /// </summary>
        /// <param name="item"></param>
        public void Add(TDist item)
        {
            conditionals.Add((TDist) item.Clone());
        }

        /// <summary>
        /// Clears out all the samples
        /// </summary>
        public void Clear()
        {
            conditionals.Clear();
        }
    }

    /// <summary>
    /// Gibbs marginal - wraps underlying estimator, provides burn-in and thinning,
    /// and maintains thinned samples and conditionals
    /// </summary>
    /// <typeparam name="TDist">The distribution type</typeparam>
    /// <typeparam name="T">The domain type</typeparam>
    [Quality(QualityBand.Preview)]
    public class GibbsMarginal<TDist, T> :
        IsDistributionWrapper,
        Estimator<TDist>
        where TDist : IDistribution<T>, Sampleable<T>
    {
        /// <summary>
        /// Constructor from a distribution prototype, and burn in and thin parameters
        /// </summary>
        /// <param name="distPrototype">Prototype distribution</param>
        /// <param name="burnIn">Burn in - number of sample discarded initially</param>
        /// <param name="thin">Thinning parameter - only every 'thin' samples returned</param>
        /// <param name="estimateMarginal"></param>
        /// <param name="collectSamples"></param>
        /// <param name="collectDistributions"></param>
        [Skip]
        public GibbsMarginal(TDist distPrototype, int burnIn, int thin, bool estimateMarginal, bool collectSamples, bool collectDistributions)
        {
            this.LastConditional = (TDist) distPrototype.Clone();
            this.LastSample = default(T);
            this.resultWorkspace = (TDist) distPrototype.Clone();
            this.Estimator = null;

            if (collectSamples)
            {
                // The sample list
                Accumulator<T> sampAcc = new SampleList<T>();
                // Add to the list of sample accumulators. 
                // This is this first in the list - do not change as some code depends on this
                this.sampleAccumulators.Accumulators.Add(new BurnInAccumulator<T>(burnIn, thin, sampAcc));
            }
            if (collectDistributions)
            {
                // The conditional list
                Accumulator<TDist> condAcc = new ConditionalList<TDist>();
                // Add to the list of distribution accumulators. 
                // This is this first in the list - do not change as some code depends on this
                this.distribAccumulators.Accumulators.Add(new BurnInAccumulator<TDist>(burnIn, thin, condAcc));
            }

            if (estimateMarginal)
            {
                // Try to create an estimator where we can add distributions. This should usually
                // be the case. If not, create an estimator where we add samples
                try
                {
                    this.Estimator = ArrayEstimator.CreateEstimator<TDist, T>(distPrototype, true);
                }
                catch (Exception)
                {
                }
                if (this.Estimator == null)
                {
                    this.Estimator = ArrayEstimator.CreateEstimator<TDist, T>(distPrototype, false);
                    Accumulator<T> acc = (Accumulator<T>)this.Estimator;
                    // Thinning is always 1 for estimators
                    this.sampleAccumulators.Accumulators.Add(new BurnInAccumulator<T>(burnIn, 1, acc));
                }
                else
                {
                    Accumulator<TDist> acc = (Accumulator<TDist>)this.Estimator;
                    // Thinning is always 1 for estimators
                    this.distribAccumulators.Accumulators.Add(new BurnInAccumulator<TDist>(burnIn, 1, acc));
                }
            }
            Clear();
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        /// <remarks>This does a soft copy of the estimator and sample and conditional lists</remarks>
        public GibbsMarginal(GibbsMarginal<TDist, T> that)
        {
            Estimator = that.Estimator;
            sampleAccumulators = that.sampleAccumulators;
            distribAccumulators = that.distribAccumulators;
            LastSample = that.LastSample;
            LastConditional = that.LastConditional;
        }

        /// <summary>
        /// The embedded estimator
        /// </summary>
        public Estimator<TDist> Estimator { get; }

        private AccumulatorList<T> sampleAccumulators = new AccumulatorList<T>();
        private AccumulatorList<TDist> distribAccumulators = new AccumulatorList<TDist>();

        // Needed for GetDistribution
        private TDist resultWorkspace;

        /// <summary>
        /// Thinned samples
        /// </summary>
        public IList<T> Samples
        {
            get
            {
                // Samples list is first in the list of sample accumulators
                return ((SampleList<T>) ((BurnInAccumulator<T>) sampleAccumulators.Accumulators[0]).Accumulator).Samples;
            }
        }

        /// <summary>
        /// Last sample added. If no samples, returns default(T)
        /// </summary>
        public T LastSample;

        /// <summary>
        /// Thinned conditionals
        /// </summary>
        public IList<TDist> Conditionals
        {
            get
            {
                // Conditionals list is first in the list of distribution accumulators
                return ((ConditionalList<TDist>) ((BurnInAccumulator<TDist>) distribAccumulators.Accumulators[0]).Accumulator).Conditionals;
            }
        }

        /// <summary>
        /// Last conditional distribution added. If no conditionals, returns uniform
        /// </summary>
        public TDist LastConditional;

        /// <summary>
        /// Clears out all the samples and clears the accumulators
        /// </summary>
        public void Clear()
        {
            sampleAccumulators.Clear();
            distribAccumulators.Clear();
        }

        #region Estimator<TDist> Members

        /// <summary>
        /// Get the estimated distribution from the samples
        /// </summary>
        /// <param name="result"></param>
        /// <returns></returns>
        public TDist GetDistribution(TDist result)
        {
            return Estimator.GetDistribution(result);
        }

        #endregion

        /// <summary>
        /// The marginal
        /// </summary>
        public TDist Distribution
        {
            get { return Estimator.GetDistribution(resultWorkspace); }
        }

        /// <summary>
        /// Perform an update by adding a sample from the last conditional
        /// </summary>
        public void PostUpdate()
        {
            // Cannot use the LastSample as result because then we would need to clone it for SampleList, and T does not always implement ICloneable
            // e.g. T=double does not implement ICloneable
            LastSample = LastConditional.Sample();
            sampleAccumulators.Add(LastSample);
            distribAccumulators.Add(LastConditional);
        }

        /// <summary>
        /// Shows the GibbsMarginal in string form
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string sampleString = "";
            if (!ReferenceEquals(LastSample, null)) sampleString = ", " + LastSample.ToString();
            return LastConditional.ToString() + sampleString;
        }
    }
}
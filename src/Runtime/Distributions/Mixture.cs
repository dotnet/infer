// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System.Runtime.Serialization;

    /// <summary>
    /// An estimator which is a mixture of distributions of the same type 
    /// </summary>
    /// <typeparam name="TDist">The distribution type</typeparam>
    [DataContract]
    public class MixtureEstimator<TDist> : Estimator<MixtureEstimator<TDist>>, Accumulator<TDist>
    {
        /// <summary>
        /// The components
        /// </summary>
        [DataMember]
        protected List<TDist> components;

        /// <summary>
        /// The mixing weight of each component.  Does not necessarily sum to 1.
        /// </summary>
        [DataMember]
        protected List<double> weights;

        /// <summary>
        /// The components of the mixture
        /// </summary>
        public IReadOnlyList<TDist> Components { get { return components; } }

        /// <summary>
        /// The weights of the mixture
        /// </summary>
        public IReadOnlyList<double> Weights { get { return weights; } }

        /// <summary>
        /// Add a component to the mixture with a given weight
        /// </summary>
        /// <param name="item">The component to add</param>
        /// <param name="weight">The weight</param>
        public void Add(TDist item, double weight)
        {
            components.Add(item);
            weights.Add(weight);
        }

        /// <summary>
        /// Add a component to the mixture. A weight of 1 is assumed
        /// </summary>
        /// <param name="item">The component to add</param>
        public void Add(TDist item)
        {
            Add(item, 1.0);
        }

        /// <summary>
        /// The sum of the component weights
        /// </summary>
        /// <returns></returns>
        public double WeightSum()
        {
            double sum = 0.0;
            foreach (double w in weights)
            {
                sum += w;
            }
            return sum;
        }

        /// <summary>
        /// Normalize the weights to add to 1
        /// </summary>
        public void Normalize()
        {
            double sum = WeightSum();
            if (sum > 0)
            {
                for (int i = 0; i < weights.Count; i++)
                {
                    weights[i] /= sum;
                }
            }
        }

        /// <summary>
        /// Create a mixture model with no components
        /// </summary>
        public MixtureEstimator()
        {
            components = new List<TDist>();
            weights = new List<double>();
        }

        /// <summary>
        /// Sets the mixture to zero, by removing all components.
        /// </summary>
        public void SetToZero()
        {
            components.Clear();
            weights.Clear();
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        void Accumulator<TDist>.Clear()
        {
            SetToZero();
        }

        /// <summary>
        /// The the resulting mixture
        /// </summary>
        /// <param name="result">Where to put the resulting mixture</param>
        /// <returns></returns>
        public MixtureEstimator<TDist> GetDistribution(MixtureEstimator<TDist> result)
        {
            Normalize();
            return this;
        }

    }


    /// <summary>
    /// A mixture of distributions of the same type 
    /// </summary>
    /// <typeparam name="TDist">The distribution type</typeparam>
    /// <typeparam name="TDomain">The domain type</typeparam>
    /// <typeparam name="TThis">For use when subclassing, the type of the subclass</typeparam>    
    public class Mixture<TDist, TDomain, TThis> : MixtureEstimator<TDist>, IDistribution<TDomain>, CanGetLogProb<TDomain>, Sampleable<TDomain>,
        SettableToProduct<TThis>, CanGetLogAverageOf<TThis>, CanGetAverageLog<TThis>
        where TDist : CanGetLogAverageOf<TDist>, CanGetLogProb<TDomain>, Sampleable<TDomain>, IDistribution<TDomain>
        where TThis : Mixture<TDist,TDomain, TThis>
    {
        //public double GetAverageLogOfComponent(TDist that)
        //{
        //    double sum = 0;
        //    double weightSum = 0;
        //    for (int i = 0; i < weights.Count; i++)
        //    {
        //        weightSum += weights[i];
        //        sum += weights[i]*components[i].GetAverageLog(that);
        //    }
        //    return sum/weightSum;
        //}

        public double GetLogProb(TDomain value)
        {
            double logProb = double.NegativeInfinity;
            double weightSum = 0;
            for (int i = 0; i < weights.Count; i++)
            {
                weightSum += weights[i];
                logProb = MMath.LogSumExp(logProb, System.Math.Log(weights[i]) + components[i].GetLogProb(value));
            }
            return logProb - System.Math.Log(weightSum);
        }

        public TDomain Sample()
        {
            int i = Rand.Sample(weights, WeightSum());
            return components[i].Sample();
        }

        public TDomain Sample(TDomain result)
        {
            int i = Rand.Sample(weights, WeightSum());
            return components[i].Sample(result);
        }

        public virtual object Clone()
        {
            throw new NotImplementedException();
        }

        public TDomain Point
        {
            get
            {
                if (!IsPointMass) throw new InvalidOperationException("Not a point mass");
                return components[0].Point;
            }
            set
            {
                if (components.Count == 0) throw new NotImplementedException("NYI: Setting a zero mixture to a point value.");
                var pt = (TDist)components[0].Clone();
                pt.Point = value;
                SetToZero();
                Add(pt);
            }
        }

        public bool IsPointMass
        {
            get {
                // todo: handle multiple co-incident point mass components
                return components.Count == 1 && components[0].IsPointMass;
            }
        }

        public double MaxDiff(object that)
        {
            throw new NotImplementedException();
        }

        public void SetToUniform()
        {
            if (components.Count==0) throw new NotImplementedException("NYI: Setting a zero mixture to uniform.");
            var unif = (TDist)components[0].Clone();
            unif.SetToUniform();
            SetToZero();
            Add(unif);            
        }

        public bool IsUniform()
        {
            return components.Count == 1 && components[0].IsUniform();
        }

        /// <summary>
        /// Replaces the components of this distribution with the specified components (with equal weight).
        /// </summary>
        /// <param name="components">The new mixture components</param>
        public void SetComponents(IEnumerable<TDist> components)
        {
            SetToZero();
            foreach (var comp in components)
            {
                Add(comp);
            }
        }

        public override string ToString()
        {
            return StringUtil.CollectionToString(Components, "/");
        }

        public virtual double GetLogAverageOf(TThis that)
        {
            throw new NotImplementedException();
        }

        public virtual void SetToProduct(TThis a, TThis b)
        {
            throw new NotImplementedException();
        }

        public double GetLogAverageOf(TDist that)
        {
            throw new NotImplementedException();
        }

        public double GetAverageLog(TThis that)
        {
            throw new NotImplementedException();
        }
    }
}
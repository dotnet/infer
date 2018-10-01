// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Base class for SparseGP message passing

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;

    /// <summary>
    /// A base class for Gaussian process distributions
    /// </summary>
    [Serializable]
    [DataContract]
    public class GaussianProcess : IGaussianProcess, Sampleable<IFunction>
    {
        /// <summary>
        /// Mean function
        /// </summary>
        [DataMember]
        public IFunction mean;

        /// <summary>
        /// Covariance function
        /// </summary>
        [DataMember]
        public IKernelFunction kernel;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="mean">Mean function</param>
        /// <param name="kernel">Kernel function</param>
        [Construction("mean", "kernel")]
        public GaussianProcess(IFunction mean, IKernelFunction kernel)
        {
            this.mean = mean;
            this.kernel = kernel;
        }

        /// <summary>
        /// This base class just returns the mean function
        /// </summary>
        /// <returns></returns>
        public IFunction Sample()
        {
            return mean;
        }

        /// <summary>
        /// This base class just returns the mean function
        /// </summary>
        /// <param name="result"></param>
        /// <returns></returns>
        public IFunction Sample(IFunction result)
        {
            return Sample();
        }

        /// <summary>
        /// Mean at a given point
        /// </summary>
        /// <param name="X"></param>
        /// <returns></returns>
        public double Mean(Vector X)
        {
            return mean.Evaluate(X);
        }

        /// <summary>
        /// Mean at a given list of points
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>Predictive mean vector</returns>
        public Vector Mean(IList<Vector> XList)
        {
            int numPoints = XList.Count;
            Vector result = Vector.Zero(numPoints);
            for (int i = 0; i < numPoints; i++)
            {
                result[i] = Mean(XList[i]);
            }
            return result;
        }

        /// <summary>
        /// Predictive Variance at a given point
        /// </summary>
        /// <param name="X">Input</param>
        /// <returns>Predictive variance</returns>
        public double Variance(Vector X)
        {
            return kernel.EvaluateX(X);
        }

        /// <summary>
        /// Predictive covariance at a given pair of points
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public double Covariance(Vector x, Vector y)
        {
            return kernel.EvaluateX1X2(x, y);
        }

        /// <summary>
        /// Predictive coariance at a given list of points
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>Predictive covariance</returns>
        public PositiveDefiniteMatrix Covariance(IList<Vector> XList)
        {
            int numPoints = XList.Count;
            PositiveDefiniteMatrix kXX = new PositiveDefiniteMatrix(numPoints, numPoints);
            for (int i = 0; i < numPoints; i++)
            {
                kXX[i, i] = kernel.EvaluateX(XList[i]);
                for (int j = i + 1; j < numPoints; j++)
                {
                    kXX[i, j] = kernel.EvaluateX1X2(XList[i], XList[j]);
                    kXX[j, i] = kXX[i, j];
                }
            }
            return kXX;
        }

        /// <summary>
        /// Predictive distribution at a given point
        /// </summary>
        /// <param name="X">Input</param>
        /// <returns>Predictive distribution</returns>
        public Gaussian Marginal(Vector X)
        {
            return new Gaussian(Mean(X), Variance(X));
        }

        /// <summary>
        /// Predictive distribution at a given list of points
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>Predictive distribution</returns>
        public VectorGaussian Joint(IList<Vector> XList)
        {
            return new VectorGaussian(Mean(XList), Covariance(XList));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            GaussianProcess that = obj as GaussianProcess;
            if (that == null) return false;
            return Equals(mean, that.mean) && Equals(kernel, that.kernel);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, mean.GetHashCode());
            hash = Hash.Combine(hash, kernel.GetHashCode());
            return hash;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "GaussianProcess(" + mean + "," + kernel + ")";
        }
    }
}
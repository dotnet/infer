// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// White noise kernel. 

using System;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions.Kernels
{
    /// <summary>
    /// White noise kernel function. This can be added to other kernels using the
    /// SummationKernel class to provide measurement noise
    /// </summary>
    [Serializable]
    public class WhiteNoise : KernelFunction
    {
        private static int version = 1; // version for read/write
        private double noiseVar;

        #region Constructors

        /// <summary>
        /// Default constructor for white noise kernel function
        /// </summary>
        public WhiteNoise()
            : base(new string[] {"NoiseSD"})
        {
            this[0] = 0.0;
        }

        public double LogNoiseSD
        {
            get { return this[0]; }
        }

        /// <summary>
        /// Construct white noise kernel function from log noise standard deviation
        /// </summary>
        /// <param name="logNoiseSD">Log noise standard deviation</param>
        [Construction("LogNoiseSD")]
        public WhiteNoise(double logNoiseSD)
            : this()
        {
            this[0] = logNoiseSD;
        }

        #endregion

        public override string ToString()
        {
            return "WhiteNoise(" + this[0] + ")";
        }

        #region IKernelFunction Members

        /// <summary>
        /// Evaluates the kernel for a pair of vectors
        /// </summary>
        /// <param name="x1">First vector</param>
        /// <param name="x2">Second vector</param>
        /// <param name="x1Deriv">Derivative of the kernel value with respect to x1 input vector</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        public override double EvaluateX1X2(Vector x1, Vector x2, ref Vector x1Deriv, ref Vector logThetaDeriv)
        {
            if (object.ReferenceEquals(x1, x2))
            {
                return EvaluateX(x1, ref x1Deriv, ref logThetaDeriv);
            }
            else
            {
                double result = 0.0;
                if (((object) logThetaDeriv) != null)
                    logThetaDeriv[0] = 0.0;
                if (((object) x1Deriv) != null)
                {
                    for (int i = 0; i < x1Deriv.Count; i++)
                    {
                        x1Deriv[i] = 0.0;
                    }
                }
                return result;
            }
        }

        /// <summary>
        /// Evaluates the kernel for a single vector (which is used for both slots)
        /// </summary>
        /// <param name="x">Vector</param>
        /// <param name="xDeriv">Derivative of the kernel value with respect to x</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        public override double EvaluateX(Vector x, ref Vector xDeriv, ref Vector logThetaDeriv)
        {
            double result = noiseVar;
            if (((object) logThetaDeriv) != null)
            {
                logThetaDeriv[0] = 2.0*noiseVar;
            }

            if (((object) xDeriv) != null)
            {
                for (int i = 0; i < xDeriv.Count; i++)
                    xDeriv[i] = 0.0;
            }
            return result;
        }

        /// <summary>
        /// Sets or gets a log hyper-parameter by index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public override double this[int index]
        {
            get { return base[index]; }
            set
            {
                //// Put a limit on log standard deviation. We should be using different
                //// units if this is more
                //if (value > 5.0)
                //    value = 5.0;

                // Convert from log ...
                noiseVar = System.Math.Exp(value);

                // ... and square to get a variance
                noiseVar *= noiseVar;

                base[index] = value;
            }
        }

        /// <summary>
        /// The static version for the derived class
        /// </summary>
        public override int TypeVersion
        {
            get { return version; }
        }

        #endregion
    }
}
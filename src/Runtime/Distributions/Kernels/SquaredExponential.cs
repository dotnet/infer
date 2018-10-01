// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Squared exponential kernel. This will typically be added to a white noise
// kernel using the Summation kernel

using System;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions.Kernels
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;

    /// <summary>
    /// Squared Exponential kernel function: k(x,y) = exp(-0.5*(x-y)^2/exp(2*logLength))
    /// </summary>
    [Serializable]
    [DataContract]
    public class SquaredExponential : KernelFunction
    {
        private static int version = 1; // version for read/write
        [DataMember]
        private double lenMult;
        [DataMember]
        private double signalVar;

        /// <summary>
        /// Constructs the kernel k(x,y) = exp(2*logSignalSD - 0.5*(x-y)^2/exp(2*logLength))
        /// </summary>
        /// <param name="logLengthScale">Log length</param>
        /// <param name="logSignalSD">Log signal variance</param>
        [Construction("LogLengthScale", "LogSignalSD")]
        public SquaredExponential(double logLengthScale, double logSignalSD)
            : base(new string[] {"Length", "SignalSD"})
        {
            this.LogLengthScale = logLengthScale;
            this.LogSignalSD = logSignalSD;
        }

        /// <summary>
        /// Checks if this object is equal to <paramref name="other"/>.
        /// </summary>
        private bool Equals(SquaredExponential other)
        {
            return lenMult.Equals(other.lenMult) && signalVar.Equals(other.signalVar);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((SquaredExponential)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (lenMult.GetHashCode() * 397) ^ signalVar.GetHashCode();
            }
        }

        /// <summary>
        /// Checks if <paramref name="left"/> object is equal to <paramref name="right"/>.
        /// </summary>
        public static bool operator ==(SquaredExponential left, SquaredExponential right)
        {
            return Equals(left, right);
        }

        /// <summary>
        /// Checks if <paramref name="left"/> object is equal to <paramref name="right"/>.
        /// </summary>
        public static bool operator !=(SquaredExponential left, SquaredExponential right)
        {
            return !Equals(left, right);
        }

        /// <summary>
        /// Constructs the kernel k(x,y) = exp(- 0.5*(x-y)^2/exp(2*logLength))
        /// </summary>
        /// <param name="logLengthScale"></param>
        public SquaredExponential(double logLengthScale) : this(logLengthScale, 0.0)
        {
        }

        /// <summary>
        /// Constructs the kernel k(x,y) = exp(- 0.5*(x-y)^2)
        /// </summary>
        public SquaredExponential()
            : this(0.0, 0.0)
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "SquaredExponential(" + LogLengthScale + "," + LogSignalSD + ")";
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
                Vector dvec = Vector.Zero(x1.Count);
                dvec.SetToDifference(x1, x2);
                double d = lenMult*dvec.Inner(dvec);
                double de = System.Math.Exp(d);
                double result = signalVar*de;
                if (((object) logThetaDeriv) != null)
                {
                    logThetaDeriv[0] = -2.0*result*d;
                    logThetaDeriv[1] = 2.0*signalVar*de;
                }
                if (((object) x1Deriv) != null)
                {
                    x1Deriv.SetToProduct(dvec, result*2.0*lenMult);
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
            double result = signalVar;
            if (((object) logThetaDeriv) != null)
            {
                logThetaDeriv[0] = 0.0;
                logThetaDeriv[1] = 2.0*signalVar;
            }
            if (((object) xDeriv) != null)
            {
                xDeriv.SetAllElementsTo(0.0);
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
                if (index == 0)
                {
                    LogLengthScale = value;
                }
                else if (index == 1)
                {
                    LogSignalSD = value;
                }
            }
        }

        /// <summary>
        /// Sets/gets log of the length scale
        /// </summary>
        public double LogLengthScale
        {
            get { return 0.5* System.Math.Log(-0.5/ lenMult); }
            set
            {
                double len = System.Math.Exp(value);
                lenMult = -0.5/(len*len);
                base[0] = value;
            }
        }

        /// <summary>
        /// Gets/sets log of the signal variance
        /// </summary>
        public double LogSignalSD
        {
            get { return 0.5* System.Math.Log(signalVar); }
            set
            {
                signalVar = System.Math.Exp(2* value);
                base[1] = value;
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
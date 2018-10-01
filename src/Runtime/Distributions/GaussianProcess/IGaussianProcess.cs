// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;

    /// <summary>
    /// Function interface - used for distributions over a function domain
    /// </summary>
    public interface IFunction
    {
        /// <summary>
        /// Evaluate a function
        /// </summary>
        /// <param name="X">Input</param>
        /// <returns>Output</returns>
        double Evaluate(Vector X);
    }

    /// <summary>
    /// Class implementing the constant function. Used as a domain prototype
    /// for distributions over functions
    /// </summary>
    [Serializable]
    [DataContract]
    public class ConstantFunction : IFunction
    {
        /// <summary>
        /// Checks if this object is equal to <paramref name="other"/>.
        /// </summary>
        private bool Equals(ConstantFunction other)
        {
            return constValue.Equals(other.constValue);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ConstantFunction)obj);
        }

        public override int GetHashCode()
        {
            return constValue.GetHashCode();
        }

        /// <summary>
        /// Checks if <paramref name="left"/> object is equal to <paramref name="right"/>.
        /// </summary>
        public static bool operator ==(ConstantFunction left, ConstantFunction right)
        {
            return Equals(left, right);
        }

        /// <summary>
        /// Checks if <paramref name="left"/> object is not equal to <paramref name="right"/>.
        /// </summary>
        public static bool operator !=(ConstantFunction left, ConstantFunction right)
        {
            return !Equals(left, right);
        }

        [DataMember]
        private double constValue;

        /// <summary>
        /// The constant value of the function
        /// </summary>
        public double ConstantValue
        {
            get { return constValue; }
            set { constValue = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "ConstantFunction(" + ConstantValue + ")";
        }

        #region IFunction Members

        /// <summary>
        /// Evaluate the function
        /// </summary>
        /// <param name="X"></param>
        /// <returns></returns>
        public double Evaluate(Vector X)
        {
            return ConstantValue;
        }

        /// <summary>
        /// Constructor for constant function - default value is 0.0
        /// </summary>
        public ConstantFunction()
        {
            constValue = 0.0;
        }

        /// <summary>
        /// Constructor for constant function
        /// </summary>
        [Construction("ConstantValue")]
        public ConstantFunction(double value)
        {
            constValue = value;
        }

        #endregion
    }

    /// <summary>
    /// Basic GP interface
    /// </summary>
    public interface IGaussianProcess
    {
        /// <summary>
        /// Mean of f at a given point
        /// </summary>
        /// <param name="X">Input</param>
        /// <returns>E[f(x)]</returns>
        double Mean(Vector X);

        /// <summary>
        /// Mean of f at a list of points
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>E[f(x_i)]</returns>
        Vector Mean(IList<Vector> XList);

        /// <summary>
        /// Variance of f at a given point
        /// </summary>
        /// <param name="X">Input</param>
        /// <returns>var(f(x))</returns>
        double Variance(Vector X);

        /// <summary>
        /// Covariance of f at two points
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns>cov(f(x),f(y))</returns>
        double Covariance(Vector x, Vector y);

        /// <summary>
        /// Covariance matrix of f at a list of points
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>cov(f(x_1),f(x_2),...)</returns>
        PositiveDefiniteMatrix Covariance(IList<Vector> XList);

        /// <summary>
        /// Marginal distribution of f at a given point
        /// </summary>
        /// <param name="X">Input</param>
        /// <returns>p(f(x))</returns>
        Gaussian Marginal(Vector X);

        /// <summary>
        /// Joint distribution of f at a list of points
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>p(f(x_1),f(x_2),...)</returns>
        VectorGaussian Joint(IList<Vector> XList);
    }
}
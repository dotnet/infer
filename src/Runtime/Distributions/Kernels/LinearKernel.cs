// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions.Kernels
{
    /// <summary>
    /// Linear kernel
    /// </summary>
    [Serializable]
    public class LinearKernel : KernelFunction
    {
        private static int version = 1; // version for read/write
        private Vector variances;

        private void initialise()
        {
            variances = null;
        }

        /// <summary>
        /// Default constructor
        /// </summary>
        public LinearKernel() : base()
        {
            initialise();
        }

        /// <summary>
        /// Sets up names and values of parameters
        /// </summary>
        /// <param name="logVariances">Log of the weight variances</param>
        public void SetupParams(double[] logVariances)
        {
            int numInputs = logVariances.Length;

            string[] hyperNames = new string[numInputs];
            for (int i = 0; i < numInputs; i++)
            {
                hyperNames[i] = "V" + i.ToString(CultureInfo.InvariantCulture);
            }
            base.HyperNames = hyperNames;
            variances = Vector.Zero(numInputs);
            for (int i = 0; i < numInputs; i++)
            {
                this[i] = logVariances[i];
            }
        }

        /// <summary>
        /// Constructs a linear kernel from vector of log variances
        /// </summary>
        /// <param name="logVariances">Log of the weight variances</param>
        public LinearKernel(double[] logVariances)
            : this()
        {
            SetupParams(logVariances);
        }

        /// <summary>
        /// Initialises the parameters from data. The variance is
        /// set as the square of the inverse of the 'length' of the
        /// input feature. Note that the variance we are trying to set
        /// up here corresponds to the variance of the weight parameters
        /// in a linear model, not to the variance of the input feature.
        /// </summary>
        /// <param name="X">X data - initialises variances</param>
        public void InitialiseFromData(IList<Vector> X)
        {
            int ND = X.Count;
            int NL = X[0].Count;
            double[] logVars = new double[NL];

            GaussianEstimator[] gex = new GaussianEstimator[NL];
            for (int l = 0; l < NL; l++)
            {
                gex[l] = new GaussianEstimator();
            }

            for (int d = 0; d < ND; d++)
            {
                Vector x = X[d];
                for (int l = 0; l < NL; l++)
                {
                    gex[l].Add(x[l]);
                }
            }

            double dimMult = System.Math.Sqrt((double) NL);
            for (int l = 0; l < NL; l++)
            {
                Gaussian g = gex[l].GetDistribution(new Gaussian());
                double length = 0.5* dimMult * System.Math.Sqrt(g.GetVariance());

                // If the variance is zero, set to some nominal value.
                // This is different from the situation where length
                // is very small
                if (length < 0.000000001)
                    length = 1.0;

                // variance = inv(sq(length))
                logVars[l] = -2.0* System.Math.Log(length);
            }
            SetupParams(logVars);
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
            if (Object.ReferenceEquals(x1, x2))
            {
                return EvaluateX(x1, ref x1Deriv, ref logThetaDeriv);
            }
            else
            {
                int numInputs = variances.Count;
                Vector dvec = Vector.Zero(numInputs);
                dvec.SetToProduct(variances, x2);
                double result = x1.Inner(dvec);

                if (((object) logThetaDeriv) != null)
                {
                    logThetaDeriv.SetToProduct(x1, dvec);
                }

                if (((object) x1Deriv) != null)
                {
                    x1Deriv.SetTo(dvec);
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
            int numInputs = variances.Count;
            Vector dvec = Vector.Zero(numInputs);
            dvec.SetToProduct(variances, x);
            double result = x.Inner(dvec);

            if (((object) logThetaDeriv) != null)
            {
                logThetaDeriv.SetToProduct(x, dvec);
            }

            if (((object) xDeriv) != null)
            {
                xDeriv.SetToProduct(dvec, 2.0);
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
                int numInputs = variances.Count;
                variances[index] = System.Math.Exp(value);
                base[index] = value;
            }
        }


        /// <summary>
        /// Reads the parameters in from a stream
        /// </summary>
        /// <param name="sr">Stream reader</param>
        public override void Read(StreamReader sr)
        {
            string typeName = this.GetType().Name;
            string start = "<" + typeName + ">";
            string end = "</" + typeName + ">";

            string str;
            str = sr.ReadLine();
            if (str != start)
                throw new IOException("Unexpected section start");

            // Version
            str = sr.ReadLine();
            int vers = int.Parse(str);

            str = sr.ReadLine();
            int thcnt = int.Parse(str);

            if (thcnt > 0)
            {
                double[] logVars = new double[thcnt];
                for (int i = 0; i < thcnt; i++)
                {
                    str = sr.ReadLine();
                    logVars[i] = double.Parse(str);
                }
                SetupParams(logVars);
            }
            else
            {
                initialise();
            }

            str = sr.ReadLine();
            if (str != end)
                throw new IOException("Unexpected section end");
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
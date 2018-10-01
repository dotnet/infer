// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions.Kernels
{
    /// <summary>
    /// Automatic Relevance Determination Kernel
    /// </summary>
    [Serializable]
    public class ARD : KernelFunction
    {
        private static int version = 1; // version for read/write
        private Vector invLength;
        private double signalVar;

        private void initialise()
        {
            invLength = null;
            signalVar = 1.0;
        }

        /// <summary>
        /// Default constructor
        /// </summary>
        public ARD() : base()
        {
            initialise();
        }

        /// <summary>
        /// Sets up names and values of parameters
        /// </summary>
        /// <param name="logLengths">Log of the lengths</param>
        /// <param name="logSigVar">Log of the signal variance</param>
        public void SetupParams(double[] logLengths, double logSigVar)
        {
            int numInputs = logLengths.Length;

            string[] hyperNames = new string[numInputs + 1];
            for (int i = 0; i < numInputs; i++)
            {
                hyperNames[i] = "L" + i.ToString(CultureInfo.InvariantCulture);
            }
            hyperNames[numInputs] = "SignalSD";
            base.HyperNames = hyperNames;
            invLength = Vector.Zero(numInputs);
            for (int i = 0; i < numInputs; i++)
            {
                this[i] = logLengths[i];
            }
            this[numInputs] = logSigVar;
        }

        public double[] LogLengths
        {
            get
            {
                var res = new double[invLength.Count];
                for (int i = 0; i < invLength.Count; i++)
                {
                    res[i] = this[i];
                }
                ;
                return res;
            }
        }

        public double LogSigVar
        {
            get { return this[invLength.Count]; }
        }

        /// <summary>
        /// Constructs an ARD kernel from a vector of log lengths, and a log signal variance
        /// </summary>
        /// <param name="logLengths"></param>
        /// <param name="logSigVar"></param>
        [Construction("LogLengths", "LogSigVar")]
        public ARD(double[] logLengths, double logSigVar) : this()
        {
            SetupParams(logLengths, logSigVar);
        }

        /// <summary>
        /// Initialises the parameters from data
        /// </summary>
        /// <param name="X">X data - initialises lengths</param>
        /// <param name="y">y data - initialises signal standard deviation</param>
        public void InitialiseFromData(IList<Vector> X, Vector y)
        {
            if (X.Count != y.Count)
                throw new ArgumentException("X and y data counts don't match");

            int ND = X.Count;
            int NL = X[0].Count;
            double[] logLengths = new double[NL];

            GaussianEstimator[] gex = new GaussianEstimator[NL];
            GaussianEstimator gey = new GaussianEstimator();
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
                gey.Add(y[d]);
            }

            double dimMult = System.Math.Sqrt((double) NL);
            for (int l = 0; l < NL; l++)
            {
                Gaussian g = gex[l].GetDistribution(new Gaussian());
                double length = 0.5* dimMult * System.Math.Sqrt(g.GetVariance());

                // If the variance is zero, set to some nominal value.
                // This is differnet from the situation where length
                // is very small
                if (length < 0.000000001)
                    length = 1.0;

                logLengths[l] = System.Math.Log(length);
            }
            double sd = System.Math.Sqrt(gey.GetDistribution(new Gaussian()).GetVariance());
            if (sd < 0.000000001)
                sd = 1.0;

            SetupParams(logLengths, System.Math.Log(sd));
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
                int numInputs = invLength.Count;
                Vector dvec = Vector.Zero(numInputs);
                dvec.SetToDifference(x1, x2);

                // Factor in the inverse length
                dvec.SetToProduct(dvec, invLength);

                double d = -0.5*dvec.Inner(dvec);
                double de = System.Math.Exp(d);

                double result = signalVar*de;

                if (((object) logThetaDeriv) != null)
                {
                    for (int i = 0; i < numInputs; i++)
                    {
                        logThetaDeriv[i] = result*dvec[i]*dvec[i];
                    }
                    logThetaDeriv[numInputs] = 2.0*signalVar*de;
                }

                if (((object) x1Deriv) != null)
                {
                    x1Deriv.SetToProduct(dvec, -result);
                    x1Deriv.SetToProduct(x1Deriv, invLength);
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
                int numInputs = x.Count;
                for (int i = 0; i < numInputs; i++)
                {
                    logThetaDeriv[i] = 0.0;
                }
                logThetaDeriv[numInputs] = 2.0*signalVar;
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
                int numInputs = invLength.Count;
                if (index < numInputs)
                {
                    double len = System.Math.Exp(value);
                    invLength[index] = 1.0/len;
                    base[index] = value;
                }
                else
                {
                    double sigSD = System.Math.Exp(value);
                    signalVar = sigSD*sigSD;
                    base[index] = value;
                }
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
                double[] logLengths = new double[thcnt - 1];
                for (int i = 0; i < thcnt - 1; i++)
                {
                    str = sr.ReadLine();
                    logLengths[i] = double.Parse(str);
                }
                str = sr.ReadLine();
                double logSigVar = double.Parse(str);
                SetupParams(logLengths, logSigVar);
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
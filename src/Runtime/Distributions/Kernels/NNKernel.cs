// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Reference:  (Williams. Computation with Infinite Neural Networks, 1998)

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions.Kernels
{
    /// <summary>
    /// Neural Net kernel
    /// </summary>
    [Serializable]
    public class NNKernel : KernelFunction
    {
        private static int version = 1; // version for read/write
        private const double twoOverPi = 2.0/ System.Math.PI;
        private Vector weightStdDev;
        private double biasWtStdDev;

        private void initialise()
        {
            weightStdDev = null;
        }

        /// <summary>
        /// Default constructor
        /// </summary>
        public NNKernel()
            : base()
        {
            initialise();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "NNKernel(" + Vector.FromArray(GetLogWeightVariances()) + "," + GetLogBiasWeightVariance() + ")";
        }

        /// <summary>
        /// Sets up names and values of parameters
        /// </summary>
        /// <param name="logWeightVariances">Log weight variances</param>
        /// <param name="logBiasWeightVariance">Log bias weight variance</param>
        public void SetupParams(double[] logWeightVariances, double logBiasWeightVariance)
        {
            int numInputs = logWeightVariances.Length;
            int numParams = numInputs + 1;

            string[] hyperNames = new string[numParams];
            for (int i = 0; i < numInputs; i++)
            {
                hyperNames[i] = "WtVar" + i.ToString(CultureInfo.InvariantCulture);
            }
            hyperNames[numInputs] = "BiasVar";
            base.HyperNames = hyperNames;
            weightStdDev = Vector.Zero(numInputs);
            for (int i = 0; i < numInputs; i++)
            {
                this[i] = logWeightVariances[i];
            }
            this[numInputs] = logBiasWeightVariance;
        }

        /// <summary>
        /// Gets the log weight variances from this Neural Net kernel instance
        /// </summary>
        /// <returns></returns>
        public double[] GetLogWeightVariances()
        {
            int numInputs = ThetaCount - 1;
            double[] logWeightVariances = new double[numInputs];
            for (int i = 0; i < logWeightVariances.Length; i++)
            {
                logWeightVariances[i] = this[i];
            }
            return logWeightVariances;
        }

        /// <summary>
        /// Gets the log bias variances from this Neural Net kernel instance
        /// </summary>
        /// <returns></returns>
        public double GetLogBiasWeightVariance()
        {
            int numInputs = ThetaCount - 1;
            return this[numInputs];
        }

        /// <summary>
        /// Constructs an neural net kernel from vector of log lweight variances
        /// </summary>
        /// <param name="logWeightVariances">Log weight variances</param>
        /// <param name="logBiasWeightVariance">Log bias weight variances</param>
        public NNKernel(double[] logWeightVariances, double logBiasWeightVariance)
            : this()
        {
            SetupParams(logWeightVariances, logBiasWeightVariance);
        }

        /// <summary>
        /// Initialises the parameters from data
        /// </summary>
        /// <param name="X">X data - initialises weight variances</param>
        public void InitialiseFromData(IList<Vector> X)
        {
            int ND = X.Count;
            int NL = X[0].Count;
            double[] logWeightVariances = new double[NL];

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

                double wtSDev = 1.0/length;


                logWeightVariances[l] = 2.0* System.Math.Log(wtSDev);
            }
            double logBiasWeightVariance = -System.Math.Log((double) NL);
            SetupParams(logWeightVariances, logBiasWeightVariance);
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
                int numInputs = weightStdDev.Count;
                Vector avec = Vector.Zero(numInputs);
                Vector bvec = Vector.Zero(numInputs);

                avec.SetToProduct(x1, weightStdDev);
                bvec.SetToProduct(x2, weightStdDev);
                double bwsq = biasWtStdDev*biasWtStdDev;

                double num = 2.0*(avec.Inner(bvec) + bwsq);
                double den1 = 1.0 + 2.0*(avec.Inner(avec) + bwsq);
                double den2 = 1.0 + 2.0*(bvec.Inner(bvec) + bwsq);
                double densq = den1*den2;
                double den = System.Math.Sqrt(densq);
                double denInv = 1.0/den;

                double y = num*denInv;
                // Limit y
                if (y > 0.999)
                    y = 0.999;
                else if (y < -0.999)
                    y = -0.999;

                double result = twoOverPi * System.Math.Asin(y);

                bool doThetaDeriv = (((object) logThetaDeriv) != null);
                bool doX1Deriv = (((object) x1Deriv) != null);

                if (doThetaDeriv || doX1Deriv)
                {
                    // Note that wt sdev = exp(0.5 * log variance)
                    // Client program works in terms of log variance
                    double mult = denInv* twoOverPi / (System.Math.Sqrt(1.0 - (y * y)));
                    double ydInv = y*denInv;

                    if (doThetaDeriv)
                    {
                        for (int i = 0; i < numInputs; i++)
                        {
                            logThetaDeriv[i] = mult*
                                               (2.0*avec[i]*bvec[i] - ((den2*avec[i]*avec[i] + den1*bvec[i]*bvec[i])*ydInv));
                        }
                        logThetaDeriv[numInputs] = mult*
                                                   ((2.0*bwsq) - ((den2*bwsq + den1*bwsq)*ydInv));
                    }

                    if (doX1Deriv)
                    {
                        for (int i = 0; i < numInputs; i++)
                        {
                            x1Deriv[i] = 2.0*mult*weightStdDev[i]*
                                         (bvec[i] - (den2*avec[i]*ydInv));
                        }
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
            int numInputs = weightStdDev.Count;
            Vector avec = Vector.Zero(numInputs);
            avec.SetToProduct(x, weightStdDev);
            double bwsq = biasWtStdDev*biasWtStdDev;
            double num = 2.0*(avec.Inner(avec) + bwsq);
            double denInv = 1.0/(1.0 + num);
            double y = num*denInv;
            // Limit y
            if (y > 0.999)
                y = 0.999;
            else if (y < -0.999)
                y = -0.999;

            double result = twoOverPi * System.Math.Asin(y);

            bool doThetaDeriv = (((object) logThetaDeriv) != null);
            bool doXDeriv = (((object) xDeriv) != null);
            if (doThetaDeriv || doXDeriv)
            {
                double mult = 2.0* denInv * denInv * twoOverPi / (System.Math.Sqrt(1.0 - (y * y)));
                double ydInv = y*denInv;

                if (doThetaDeriv)
                {
                    for (int i = 0; i < numInputs; i++)
                    {
                        logThetaDeriv[i] = mult*avec[i]*avec[i];
                    }
                    logThetaDeriv[numInputs] = mult*bwsq;
                }

                if (doXDeriv)
                {
                    for (int i = 0; i < numInputs; i++)
                    {
                        xDeriv[i] = 2.0*mult*avec[i]*weightStdDev[i];
                    }
                }
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
                int numInputs = weightStdDev.Count;
                if (index < numInputs)
                {
                    weightStdDev[index] = System.Math.Exp(0.5* value);
                    base[index] = value;
                }
                else if (index == numInputs)
                {
                    biasWtStdDev = System.Math.Exp(0.5* value);
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
                double[] logWeightVariances = new double[thcnt - 1];
                for (int i = 0; i < thcnt - 1; i++)
                {
                    str = sr.ReadLine();
                    logWeightVariances[i] = double.Parse(str);
                }
                str = sr.ReadLine();
                double logBiasWeightVariance = double.Parse(str);
                SetupParams(logWeightVariances, logBiasWeightVariance);
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
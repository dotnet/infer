// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// This file defines the basic interface that all kernel functions must
// implement, and also provides a virtual base implementation which provides
// the common functionality

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions.Kernels
{
    /// <summary>
    /// Interface for all Kernel functions
    /// </summary>
    public interface IKernelFunction
    {
        /// <summary>
        /// Evaluates the kernel for a pair of vectors
        /// </summary>
        /// <param name="x1">First vector</param>
        /// <param name="x2">Second vector</param>
        /// <returns></returns>
        double EvaluateX1X2(Vector x1, Vector x2);

        /// <summary>
        /// Evaluates the kernel for a single vector
        /// </summary>
        /// <param name="x">Vector</param>
        /// <returns></returns>
        double EvaluateX(Vector x);
    }

    /// <summary>
    /// Interface for Kernel functions with parameters
    /// </summary>
    public interface IKernelFunctionWithParams : IKernelFunction
    {
        /// <summary>
        /// Evaluates the kernel for a pair of vectors and, optionally, returns derivatives
        /// with respect to the parameters
        /// </summary>
        /// <param name="x1">First vector</param>
        /// <param name="x2">Second vector</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        double EvaluateX1X2(Vector x1, Vector x2, ref Vector logThetaDeriv);

        /// <summary>
        /// Evaluates the kernel for a pair of vectors and, optionally, returns derivatives
        /// with respect to each vector, and with respect to the parameters
        /// </summary>
        /// <param name="x1">First vector</param>
        /// <param name="x2">Second vector</param>
        /// <param name="x1Deriv">Derivative of the kernel value with respect to x1 input vector</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        double EvaluateX1X2(Vector x1, Vector x2, ref Vector x1Deriv, ref Vector logThetaDeriv);

        /// <summary>
        /// Evaluates the kernel for a single vector, and optionally, returns the derivatives
        /// with respect to the parameters
        /// </summary>
        /// <param name="x">Vector</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        double EvaluateX(Vector x, ref Vector logThetaDeriv);

        /// <summary>
        /// Evaluates the kernel for a single vector, and optionally, returns the derivatives
        /// with respect to the vector and with respect to the parameters
        /// </summary>
        /// <param name="x">Vector</param>
        /// <param name="xDeriv">Derivative of the kernel value with respect to x</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        double EvaluateX(Vector x, ref Vector xDeriv, ref Vector logThetaDeriv);

        /// <summary>
        /// Sets or gets a log hyper-parameter by index
        /// </summary>
        /// <param name="index">Index of the log hyper-parameter</param>
        /// <returns>The log hyper-parameter value</returns>
        double this[int index] { get; set; }

        /// <summary>
        /// Sets or gets a log hyper-parameter by name
        /// </summary>
        /// <param name="name">Name of the hyper-parameter</param>
        /// <returns>The log hyper-parameter value</returns>
        double this[string name] { get; set; }

        /// <summary>
        /// Gets the index of a specified hyperparameter name
        /// </summary>
        /// <param name="thetaName">The hyper-parameter name</param>
        /// <returns>The index</returns>
        int NameToIndex(string thetaName);

        /// <summary>
        /// Gets the name of a specified hyperparameter index
        /// </summary>
        /// <param name="thetaIndex">The hyper-parameter index</param>
        /// <returns>The name</returns>
        string IndexToName(int thetaIndex);

        /// <summary>
        /// Hyper-parameter count
        /// </summary>
        int ThetaCount { get; }

        /// <summary>
        /// Writes the function parameters out to a stream
        /// </summary>
        /// <param name="sw">String Writer</param>
        void Write(StreamWriter sw);

        /// <summary>
        /// Reads the function parameters in from a stream
        /// </summary>
        /// <param name="sr">String Reader</param>
        void Read(StreamReader sr);

        /// <summary>
        /// The version for the derived class
        /// </summary>
        int TypeVersion { get; }
    }

    /// <summary>
    /// Base class for all kernel functions
    /// </summary>
    [Serializable]
    public abstract class KernelFunction : IKernelFunctionWithParams
    {
        /// <summary>
        /// Hyper-parameter names
        /// </summary>
        protected IList<string> thetaNames;

        /// <summary>
        /// Hyper-parameter values
        /// </summary>
        protected double[] thetaValues;

        /// <summary>
        /// Dictionary that allows look-up of index from hype-parameter name
        /// </summary>
        protected Dictionary<string, int> thetaName2Index;

        private Vector nullVec = null;

        [Construction("thetaNames", "thetaValues")]
        protected KernelFunction(IList<string> hyperNames, double[] values)
        {
            HyperNames = hyperNames;
            thetaValues = values;
        }

        /// <summary>
        /// Protected constructor - derived classes pass down their list
        /// of hyper-parameter names
        /// </summary>
        /// <param name="hyperNames"></param>
        protected KernelFunction(IList<string> hyperNames)
        {
            HyperNames = hyperNames;
        }

        /// <summary>
        /// Default constructor
        /// </summary>
        protected KernelFunction()
            : this(null)
        {
        }

        /// <summary>
        /// Sets the names of the hyper-parameters. Note that this destroys any values
        /// </summary>
        public virtual IList<string> HyperNames
        {
            set
            {
                thetaNames = value;
                thetaName2Index = new Dictionary<string, int>();
                if (thetaNames != null)
                {
                    thetaValues = new double[thetaNames.Count];
                    for (int i = 0; i < thetaNames.Count; i++)
                    {
                        thetaName2Index.Add(thetaNames[i], i);
                    }
                }
            }
        }

        #region IKernelFunction Members

        /// <summary>
        /// Evaluates the kernel for a pair of vectors
        /// </summary>
        /// <param name="x1">First vector</param>
        /// <param name="x2">Second vector</param>
        /// <returns></returns>
        public virtual double EvaluateX1X2(Vector x1, Vector x2)
        {
            return EvaluateX1X2(x1, x2, ref nullVec, ref nullVec);
        }

        /// <summary>
        /// Evaluates the kernel for a pair of vectors and, optionally, returns derivatives
        /// with respect to the parameters
        /// </summary>
        /// <param name="x1">First vector</param>
        /// <param name="x2">Second vector</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        public virtual double EvaluateX1X2(Vector x1, Vector x2, ref Vector logThetaDeriv)
        {
            return EvaluateX1X2(x1, x2, ref nullVec, ref logThetaDeriv);
        }

        /// <summary>
        /// Evaluates the kernel for a pair of vectors and, optionally, returns derivatives
        /// with respect to each vector, and with respect to the parameters
        /// </summary>
        /// <param name="x1">First vector</param>
        /// <param name="x2">Second vector</param>
        /// <param name="x1Deriv">Derivative of the kernel value with respect to x1 input vector</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        public abstract double EvaluateX1X2(Vector x1, Vector x2, ref Vector x1Deriv, ref Vector logThetaDeriv);

        /// <summary>
        /// Evaluates the kernel for a single vector
        /// </summary>
        /// <param name="x">Vector</param>
        /// <returns></returns>
        public virtual double EvaluateX(Vector x)
        {
            return EvaluateX(x, ref nullVec, ref nullVec);
        }

        /// <summary>
        /// Evaluates the kernel for a single vector, and optionally, returns the derivatives
        /// with respect to the parameters
        /// </summary>
        /// <param name="x">Vector</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        public virtual double EvaluateX(Vector x, ref Vector logThetaDeriv)
        {
            Vector nullVec = null;
            return EvaluateX(x, ref nullVec, ref logThetaDeriv);
        }

        /// <summary>
        /// Evaluates the kernel for a single vector, and optionally, returns the derivatives
        /// with respect to the vector and with respect to the parameters
        /// </summary>
        /// <param name="x">Vector</param>
        /// <param name="xDeriv">Derivative of the kernel value with respect to x</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        public abstract double EvaluateX(Vector x, ref Vector xDeriv, ref Vector logThetaDeriv);

        /// <summary>
        /// Sets or gets a log hyper-parameter by index
        /// </summary>
        /// <param name="index">Index of the log hyper-parameter</param>
        /// <returns>The log hyper-parameter value</returns>
        public virtual double this[int index]
        {
            get { return thetaValues[index]; }
            set { thetaValues[index] = value; }
        }

        /// <summary>
        /// Sets or gets hyper-parameter by name. This indexer is not over-rideable
        /// </summary>
        /// <param name="name">Mame of the hyper-parameter</param>
        /// <returns>The hyper-parameter value</returns>
        public virtual double this[string name]
        {
            get
            {
                // Go through the integer indexer. This way, if a
                // derived class needs to override, it only needs
                // to override the integer version
                return this[thetaName2Index[name]];
            }
            set
            {
                // Go through the integer indexer. This way, if a
                // derived class needs to override, it only needs
                // to override the integer version
                this[thetaName2Index[name]] = value;
            }
        }

        /// <summary>
        /// Gets the index of a specified hyperparameter name
        /// </summary>
        /// <param name="thetaName">The hyper-parameter name</param>
        /// <returns>The index</returns>
        public virtual int NameToIndex(string thetaName)
        {
            if (thetaName2Index.ContainsKey(thetaName))
                return thetaName2Index[thetaName];
            else
                return -1;
        }

        /// <summary>
        /// Gets the index of a specified hyperparameter name
        /// </summary>
        /// <param name="thetaIndex">The hyper-parameter index</param>
        /// <returns>The index</returns>
        public virtual string IndexToName(int thetaIndex)
        {
            if (thetaNames != null && thetaIndex >= 0 && thetaIndex < thetaNames.Count)
                return thetaNames[thetaIndex];
            else
                return null;
        }

        /// <summary>
        /// Hyper-parameter count
        /// </summary>
        public virtual int ThetaCount
        {
            get
            {
                if (thetaName2Index == null)
                    return 0;

                return thetaName2Index.Count;
            }
        }

        /// <summary>
        /// Writes the parameters out to a stream
        /// </summary>
        /// <param name="sw">Stream writer</param>
        public virtual void Write(StreamWriter sw)
        {
            string typeName = this.GetType().Name;
            string start = "<" + typeName + ">";
            string end = "</" + typeName + ">";
            sw.WriteLine(start);
            sw.WriteLine(this.TypeVersion);

            int thcnt = this.ThetaCount;
            sw.WriteLine(thcnt);
            for (int i = 0; i < thcnt; i++)
                sw.WriteLine(this[i]);
            sw.WriteLine(end);
        }

        /// <summary>
        /// Read the parameters in from a stream
        /// </summary>
        /// <param name="sr">Stream reader</param>
        public virtual void Read(StreamReader sr)
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

            // Following assumes that the slots have been
            // set up by the constructor. This is not a valid
            // assumption for kernels with variable theta counts
            // so such kernels should implement their own Read methods
            for (int i = 0; i < thcnt; i++)
            {
                str = sr.ReadLine();
                double d = double.Parse(str);
                this[i] = d;
            }

            str = sr.ReadLine();
            if (str != end)
                throw new IOException("Unexpected section end");
        }

        /// <summary>
        /// The static version for the derived class
        /// </summary>
        public abstract int TypeVersion { get; }

        #endregion

        /// <summary>
        /// Cholesky of Kernel matrix
        /// </summary>
        /// <param name="kf">Kernel function</param>
        /// <param name="xData">Data with which to build the matrix</param>
        /// <returns></returns>
        public static LowerTriangularMatrix Cholesky(IKernelFunction kf, Dictionary<int, Vector> xData)
        {
            if (null == kf || null == xData)
                throw new ArgumentException("Unexpected null arguments");

            int nData = xData.Count;

            // Allocate and fill the Kernel matrix.
            PositiveDefiniteMatrix K = new PositiveDefiniteMatrix(nData, nData);

            for (int i = 0; i < nData; i++)
            {
                for (int j = 0; j < nData; j++)
                {
                    // Evaluate the kernel. All hyperparameters, including noise
                    // variance are handled in the kernel.
                    K[i, j] = kf.EvaluateX1X2(xData[i], xData[j]);
                }
            }

            LowerTriangularMatrix chol = new LowerTriangularMatrix(nData, nData);
            chol.SetToCholesky(K);
            K = null;
            GC.Collect(0);

            return chol;
        }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Summation kernel

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.IO;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Distributions.Kernels
{
    /// <summary>
    /// Summation kernel. This provides the management layer for adding
    /// together different kernels
    /// </summary>
    [Serializable]
    public class SummationKernel : KernelFunction
    {
        private static int version = 1; // version for read/write

        /// <summary>
        /// List of kernels in the summation
        /// </summary>
        protected List<IKernelFunctionWithParams> kernels;

        /// <summary>
        /// Index of a hyper-parameter within a participating kernel
        /// </summary>
        protected int[] indexInKernel;

        /// <summary>
        /// Index of the kernel to which the hyper-parameter belongs
        /// </summary>
        protected int[] indexOfKernel;

        /// <summary>
        /// Parameter counts of the individual kernels
        /// </summary>
        protected int[] thetaCount;

        /// <summary>
        /// Clear maps between container and containees
        /// </summary>
        protected void clearMaps()
        {
            indexInKernel = null;
            indexOfKernel = null;
            thetaCount = null;
            thetaNames = null;
            thetaName2Index = null;
        }

        /// <summary>
        /// (Re-)initialise a summation kernel. This clears
        /// all participating kernels
        /// </summary>
        protected void initialise()
        {
            kernels = new List<IKernelFunctionWithParams>();
            clearMaps();
        }

        /// <summary>
        /// Helper method for building maps between container and containees
        /// </summary>
        protected void buildMaps()
        {
            clearMaps();

            int thetaSum = 0;
            thetaCount = new int[kernels.Count];
            for (int kx = 0; kx < kernels.Count; kx++)
            {
                IKernelFunctionWithParams k = kernels[kx];
                thetaCount[kx] = k.ThetaCount;
                thetaSum += k.ThetaCount;
            }
            indexOfKernel = new int[thetaSum];
            indexInKernel = new int[thetaSum];
            thetaNames = new string[thetaSum];
            thetaName2Index = new Dictionary<string, int>();

            int idxOf = 0;
            int idx = 0;
            foreach (IKernelFunctionWithParams k in kernels)
            {
                for (int idxIn = 0; idxIn < k.ThetaCount; idxIn++, idx++)
                {
                    indexOfKernel[idx] = idxOf;
                    indexInKernel[idx] = idxIn;
                    string name = k.IndexToName(idxIn);
                    thetaNames[idx] = name;
                    // For index look-up, hoose a unique name in case there are clashes between two kernels
                    while (thetaName2Index.ContainsKey(name))
                    {
                        name += "_" + idxOf.ToString(CultureInfo.InvariantCulture);
                    }
                    thetaName2Index.Add(name, idx);
                }
                idxOf++;
            }
        }

        [Construction("Kernels")]
        public SummationKernel(List<IKernelFunctionWithParams> kernels)
        {
            this.kernels = kernels;
            buildMaps();
        }

        /// <summary>
        /// Default constructor
        /// </summary>
        public SummationKernel() : base(null)
        {
            initialise();
        }

        /// <summary>
        /// Constructs summation kernel from an initial kernel function
        /// </summary>
        /// <param name="kernel"></param>
        public SummationKernel(IKernelFunctionWithParams kernel) : this()
        {
            kernels.Add(kernel);
            buildMaps();
        }

        /// <summary>
        /// Adds two kernel functions
        /// </summary>
        /// <param name="kernelA"></param>
        /// <param name="kernelB"></param>
        /// <returns></returns>
        public static SummationKernel operator +(SummationKernel kernelA, IKernelFunctionWithParams kernelB)
        {
            SummationKernel res = new SummationKernel();
            res.kernels = kernelA.kernels;
            res.kernels.Add(kernelB);
            res.buildMaps();
            return res;
        }

        /// <summary>
        /// The participating kernels
        /// </summary>
        public virtual IList<IKernelFunctionWithParams> Kernels
        {
            get { return kernels; }
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            bool notFirst = false;
            foreach (object kernel in Kernels)
            {
                if (notFirst) sb.Append("+");
                else notFirst = true;
                sb.Append(kernel);
            }
            return sb.ToString();
        }

        #region IKernelFunction Members

        /// <summary>
        /// Number of theta parameters
        /// </summary>
        public override int ThetaCount
        {
            get
            {
                if (indexOfKernel == null)
                    return 0;
                else
                    return indexOfKernel.Length;
            }
        }

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
            int totalCount = this.ThetaCount;
            double result = 0.0;
            Vector lthd;
            Vector lx1d;

            // Check that log theta deriv is correct size
            if (((object) logThetaDeriv) != null)
            {
                if (logThetaDeriv.Count != totalCount)
                    logThetaDeriv = Vector.Zero(totalCount);
            }
            // Check that the x1 and x2 derivs are the correct size
            if (((object) x1Deriv) != null)
            {
                if (x1Deriv.Count != x1.Count)
                    x1Deriv = Vector.Zero(x1.Count);
                lx1d = Vector.Zero(x1.Count);
            }
            else
                lx1d = null;

            int derivX = 0;
            for (int kx = 0; kx < kernels.Count; kx++)
            {
                IKernelFunctionWithParams k = kernels[kx];
                int thcnt = thetaCount[kx];
                if (((object) logThetaDeriv) != null)
                    lthd = Vector.Zero(thcnt);
                else
                    lthd = null;
                result += k.EvaluateX1X2(x1, x2, ref lx1d, ref lthd);
                if (((object) lthd) != null)
                {
                    for (int ix = 0; ix < thcnt; ix++)
                    {
                        logThetaDeriv[derivX++] = lthd[ix];
                    }
                }
                if (((object) lx1d) != null)
                {
                    x1Deriv.SetToSum(x1Deriv, lx1d);
                }
            }
            return result;
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
            int totalCount = this.ThetaCount;
            double result = 0.0;
            Vector lthd;
            Vector lx1d;

            // Check that log theta deriv is correct size
            if (((object) logThetaDeriv) != null)
            {
                if (logThetaDeriv.Count != totalCount)
                    logThetaDeriv = Vector.Zero(totalCount);
            }
            // Check that the x deriv is the correct size
            // Check that log theta deriv is correct size
            if (((object) xDeriv) != null)
            {
                if (xDeriv.Count != x.Count)
                    xDeriv = Vector.Zero(x.Count);
                lx1d = Vector.Zero(x.Count);
            }
            else
                lx1d = null;

            int derivX = 0;
            for (int kx = 0; kx < kernels.Count; kx++)
            {
                IKernelFunctionWithParams k = kernels[kx];
                int thcnt = thetaCount[kx];
                if (((object) logThetaDeriv) != null)
                {
                    lthd = Vector.Zero(thcnt);
                }
                else
                    lthd = null;
                result += k.EvaluateX(x, ref lx1d, ref lthd);
                if (((object) lthd) != null)
                {
                    for (int ix = 0; ix < thcnt; ix++)
                    {
                        logThetaDeriv[derivX++] = lthd[ix];
                    }
                }
                if (((object) lx1d) != null)
                {
                    xDeriv.SetToSum(xDeriv, lx1d);
                }
            }
            return result;
        }

        /// <summary>
        /// Sets or gets a log hyper-parameter by index
        /// </summary>
        /// <param name="index">parameter index</param>
        /// <returns></returns>
        public override double this[int index]
        {
            get
            {
                int idx_Of = indexOfKernel[index];
                int idx_In = indexInKernel[index];
                return kernels[idx_Of][idx_In];
            }
            set
            {
                int idx_Of = indexOfKernel[index];
                int idx_In = indexInKernel[index];
                kernels[idx_Of][idx_In] = value;
            }
        }

        /// <summary>
        /// Set or get hyper-parameter by name. This indexer is not over-rideable
        /// </summary>
        /// <param name="name">Mame of the hyper-parameter</param>
        /// <returns>The hyper-parameter value</returns>
        public override double this[string name]
        {
            get
            {
                // Get the index
                int idx = NameToIndex(name);

                if (idx >= 0)
                {
                    int idxOf = indexOfKernel[idx];
                    int idxIn = indexInKernel[idx];

                    return kernels[idxOf][idxIn];
                }
                else
                {
                    throw new ArgumentException("Name <" + name + "> cannot be found in summation kernel");
                }
            }
            set
            {
                // Get the index
                int idx = NameToIndex(name);

                if (idx >= 0)
                {
                    int idxOf = indexOfKernel[idx];
                    int idxIn = indexInKernel[idx];

                    kernels[idxOf][idxIn] = value;
                }
                else
                {
                    throw new ArgumentException("Name <" + name + "> cannot be found in summation kernel");
                }
            }
        }

        /// <summary>
        /// Writes the function parameters out to a stream
        /// </summary>
        /// <param name="sw">Stream writer</param>
        public override void Write(StreamWriter sw)
        {
            string typeName = this.GetType().Name;
            string start = "<" + typeName + ">";
            string end = "</" + typeName + ">";

            sw.WriteLine(start);
            sw.WriteLine(this.TypeVersion);

            // Write out the kernel names
            int kcnt = this.kernels.Count;
            sw.WriteLine(kcnt);

            // Now write out the kernels
            foreach (IKernelFunctionWithParams kf in kernels)
            {
                sw.WriteLine(kf.GetType().Name);
                kf.Write(sw);
            }

            sw.WriteLine(end);
        }

        /// <summary>
        /// Reads the function parameters in from a stream
        /// </summary>
        /// <param name="sr">Stream reader</param>
        public override void Read(StreamReader sr)
        {
            string typeName = this.GetType().Name;
            string start = "<" + typeName + ">";
            string end = "</" + typeName + ">";


            KernelFactory kfact = KernelFactory.Instance;
            initialise();

            string str;
            str = sr.ReadLine();
            if (str != start)
                throw new IOException("Unexpected section start");

            // Version
            str = sr.ReadLine();
            int vers = int.Parse(str);

            // kernel count
            str = sr.ReadLine();
            int kcnt = int.Parse(str);

            // Cosntruct the kernels
            for (int i = 0; i < kcnt; i++)
            {
                str = sr.ReadLine();
                IKernelFunctionWithParams kf = kfact.CreateKernelFunction(str);
                kf.Read(sr);
                kernels.Add(kf);
            }
            // Build the maps
            buildMaps();

            // Read in the end marker
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
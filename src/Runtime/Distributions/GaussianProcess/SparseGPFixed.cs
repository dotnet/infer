// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// This maintains all the fixed parameters for a sparse GP
// - i.e. parameters which the inference does not change.

using System;
using System.Collections.Generic;

using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Collections;
    using System.Runtime.Serialization;

    /// <summary>
    /// This class maintains all the fixed parameters for a sparse GP
    /// - i.e. parameters which the inference does not change.
    /// All SparseGP messages can refer to a single SparseGPFixed
    /// class, and cloning of SparseGP instances will just copy the
    /// reference
    /// </summary>
    [Serializable]
    [DataContract]
    public class SparseGPFixed
    {
        /// <summary>
        /// field for Prior property
        /// </summary>
        [DataMember]
        protected IGaussianProcess prior;

        /// <summary>
        /// Prior distribution to which basis points are added.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public IGaussianProcess Prior
        {
            get { return prior; }
            set
            {
                prior = value;
                ClearCachedValues();
            }
        }

        /// <summary>
        /// Field for basis property
        /// </summary>
        [DataMember]
        protected IList<Vector> basis;

        /// <summary>
        /// Checks if this object is equal to <paramref name="other"/>.
        /// </summary>
        public bool Equals(SparseGPFixed other)
        {
            return StructuralComparisons.StructuralEqualityComparer.Equals(basis, other.basis) && Equals(prior, other.prior);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            var other = obj as SparseGPFixed;
            return other != null && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return ((basis?.GetHashCode() ?? 0) * 397) ^ (prior?.GetHashCode() ?? 0);
            }
        }

        /// <summary>
        /// Checks if <paramref name="left"/> object is equal to <paramref name="right"/>.
        /// </summary>
        public static bool operator ==(SparseGPFixed left, SparseGPFixed right)
        {
            return Equals(left, right);
        }

        /// <summary>
        /// Checks if <paramref name="left"/> object is not equal to <paramref name="right"/>.
        /// </summary>
        public static bool operator !=(SparseGPFixed left, SparseGPFixed right)
        {
            return !Equals(left, right);
        }

        /// <summary>
        /// List of basis vectors
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public IList<Vector> Basis
        {
            get { return basis; }
            set
            {
                basis = value;
                ClearCachedValues();
            }
        }

        /// <summary>
        /// Field for calculated KernelOf_B_B property
        /// </summary>
        private PositiveDefiniteMatrix kBB;

        /// <summary>
        /// Kernel evaluated at the basis points
        /// </summary>
        public PositiveDefiniteMatrix KernelOf_B_B
        {
            get
            {
                if (kBB == null)
                    Recalculate();
                return kBB;
            }
        }

        /// <summary>
        /// Field for calculated InvKernelOf_B_B property
        /// </summary>
        private PositiveDefiniteMatrix invKBB;

        /// <summary>
        /// Inverse of the kernel evaluated at the basis points
        /// </summary>
        public PositiveDefiniteMatrix InvKernelOf_B_B
        {
            get
            {
                if (invKBB == null)
                    Recalculate();
                return invKBB;
            }
        }

        /// <summary>
        /// Number of features - i.e. the dimension of the
        /// GP index space
        /// </summary>
        public int NumberFeatures
        {
            get
            {
                if (basis != null && basis.Count > 0)
                    return basis[0].Count;
                else
                    return 0;
            }
        }

        /// <summary>
        /// Number of basis points
        /// </summary>
        public int NumberBasisPoints
        {
            get
            {
                if (basis != null)
                    return basis.Count;
                else
                    return 0;
            }
        }

        /// <summary>
        /// Function to signal recalculation of KBB and InvKBB.
        /// This is be called by the basis and kernel
        /// function property set functions, and should
        /// also be called by any external program
        /// which directly modifies the kernel
        /// </summary>
        public void ClearCachedValues()
        {
            kBB = null;
            invKBB = null;
        }

        /// <summary>
        /// Function to recalulate KBB and InvKBB
        /// </summary>
        protected void Recalculate()
        {
            int numBas = NumberBasisPoints;
            int numFeat = NumberFeatures;

            if (prior == null || numBas <= 0 || numFeat <= 0)
            {
                ClearCachedValues();
            }
            else
            {
                kBB = prior.Covariance(basis);
                invKBB = kBB.Inverse();
                //invKBB = new PositiveDefiniteMatrix(NumberBasisPoints, NumberBasisPoints);
                //LowerTriangularMatrix lt = new LowerTriangularMatrix(NumberBasisPoints, NumberBasisPoints);
                //lt.SetToCholesky(kBB);
                //invKBB.SetToIdentity();
                //invKBB.PredivideBy(lt);
                //invKBB.PredivideByTranspose(lt);
            }
        }

        /// <summary>
        /// Evaluates the kernel of a point against the basis
        /// </summary>
        /// <param name="x">Input</param>
        /// <returns>Kernel values</returns>
        public Vector KernelOf_X_B(Vector x)
        {
            int numBasis = NumberBasisPoints;
            Vector result = Vector.Zero(numBasis);
            IList<Vector> b = Basis;
            for (int i = 0; i < numBasis; i++)
            {
                result[i] = Prior.Covariance(x, b[i]);
            }
            return result;
        }

        /// <summary>
        /// Evaluates the kernel of a list of points against the basis
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>Kernel values</returns>
        public Matrix KernelOf_X_B(IList<Vector> XList)
        {
            int numPoints = XList.Count;
            int numBasis = Basis.Count;
            Matrix kXB = new Matrix(numPoints, numBasis);
            for (int i = 0; i < numPoints; i++)
            {
                for (int j = 0; j < numBasis; j++)
                {
                    kXB[i, j] = Prior.Covariance(XList[i], Basis[j]);
                }
            }
            return kXB;
        }

        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected SparseGPFixed()
        {
        }

        /// <summary>
        /// Constructor from kernel function and basis
        /// </summary>
        public SparseGPFixed(IKernelFunction kf, IList<Vector> basis)
            : this(new GaussianProcess(new ConstantFunction(), kf), basis)
        {
        }

        /// <summary>
        /// Constructor from prior and basis
        /// </summary>
        /// <param name="prior"></param>
        /// <param name="basis"></param>
        [Construction("Prior", "Basis")]
        public SparseGPFixed(IGaussianProcess prior, IList<Vector> basis)
        {
            this.prior = prior;
            this.basis = basis;
            ClearCachedValues();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return prior + ",nbasis=" + NumberBasisPoints;
        }
    }
}
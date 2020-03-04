// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// SparseGP class. Supports full rank (alpha/beta) Sparse GP
// and calculations involving rank1 potentials.
// Reference: Snelson and Ghahramani (2006)

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Text;
    using System.Runtime.Serialization;
    using Math;
    using Utilities;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    #region Rank 1 potential

    /// <summary>
    /// Rank 1 potential for a sparse GP. This low rank parameterisation
    /// is used for messages flowing from a SparseGP evaluation factor to
    /// a function variable.
    /// </summary>
    public class Rank1Pot
    {
        #region Fields

        /// <summary>
        /// Xi
        /// </summary>
        public Vector Xi;

        /// <summary>
        /// Yi
        /// </summary>
        public double Yi;

        /// <summary>
        /// Lambda inverse
        /// </summary>
        public double LambdaInv;

        #endregion

        #region Constructors

        /// <summary>
        /// Default constructor
        /// </summary>
        public Rank1Pot()
        {
            this.Xi = null;
            this.Yi = 0.0;
            this.LambdaInv = 1.0;
            this.ClearCachedValues();
        }

        #endregion

        #region Calculated properties

        /// <summary>
        /// Field for K_B_x property
        /// </summary>
        /// 
        protected Vector kBx;

        /// <summary>
        /// K(B,x). This is a calculated Vector maintained
        /// by the class
        /// </summary>
        public Vector K_B_x(SparseGP sgpb)
        {
            if (kBx == null)
            {
                if (Xi != null)
                {
                    kBx = sgpb.FixedParameters.KernelOf_X_B(Xi);
                }
            }
            return kBx;
        }

        /// <summary>
        /// Field for P property
        /// </summary>
        protected Vector pvec;

        /// <summary>
        /// p = Inv(K(B,B)) * K(B,x). This is a calculated Vector maintained
        /// by the class
        /// </summary>
        public Vector P(SparseGP sgpb)
        {
            if (pvec == null)
            {
                Vector KBx = K_B_x(sgpb);
                if (KBx != null)
                {
                    pvec = sgpb.FixedParameters.InvKernelOf_B_B*KBx;
                }
            }
            return pvec;
        }


        /// <summary>
        /// Field for K_x_x property
        /// </summary>
        protected double kxx;

        /// <summary>
        /// k(x)
        /// </summary>
        public double K_x_x(SparseGP sgpb)
        {
            if (double.IsNaN(kxx))
                kxx = sgpb.FixedParameters.Prior.Variance(Xi);
            return kxx;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Flag recalculation of the calculated properties
        /// </summary>
        public void ClearCachedValues()
        {
            kBx = null;
            pvec = null;
            kxx = double.NaN;
        }

        #endregion
    }

    #endregion

    /// <summary>
    /// A Gaussian Process distribution over functions, represented by a GP prior times a set of regression likelihoods on basis points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This distribution family comes from the paper "Sparse-posterior Gaussian Processes for general likelihoods"
    /// by Qi et al (2010), http://event.cwi.nl/uai2010/papers/UAI2010_0283.pdf
    /// </para><para>
    /// The state of the distribution is represented by (FixedParameters, IncludePrior, InducingDist, pointFunc).
    /// The GP prior and basis point locations are stored in FixedParameters.
    /// The regression likelihoods are stored as a single VectorGaussian called InducingDist.
    /// IncludePrior=false does not include the prior in the distribution (i.e. the distribution is degenerate).
    /// If pointFunc != null, the distribution is a point mass.
    /// If InducingDist is uniform and IncludePrior is false, the distribution is uniform.
    /// The GP prior is assumed to be non-uniform.
    /// </para>
    /// </remarks>
    [Serializable]
    [Quality(QualityBand.Preview)]
    [DataContract]
    public class SparseGP :
        IGaussianProcess,
        IDistribution<IFunction>,
        SettableTo<SparseGP>,
        SettableToProduct<SparseGP>,
        SettableToRatio<SparseGP>,
        SettableToPower<SparseGP>,
        SettableToWeightedSum<SparseGP>,
        CanGetLogAverageOf<SparseGP>,
        CanGetLogAverageOfPower<SparseGP>,
        CanGetAverageLog<SparseGP>,
        CanGetMean<IFunction>,
        Sampleable<IFunction>
    {
        /// <summary>
        /// Field for FixedParameters property
        /// </summary>
        [DataMember]
        protected SparseGPFixed fixedParameters;

        /// <summary>
        /// Sets and gets the fixed sparse parameters - parameters
        /// which are not changed by inference
        /// </summary>
        public SparseGPFixed FixedParameters
        {
            get { return fixedParameters; }
            set
            {
                fixedParameters = value;
                if (InducingDist.Dimension != fixedParameters.NumberBasisPoints) InducingDist = new VectorGaussian(fixedParameters.NumberBasisPoints);
                ClearCachedValues();
            }
        }

        #region Sparse GP Parameters

        /// <summary>
        /// The regression likelihoods that modify the prior.
        /// </summary>
        /// <remarks>
        /// If this field is changed, ClearCachedValues() must be called before accessing any other property.
        /// </remarks>
        [DataMember]
        public VectorGaussian InducingDist;

        /// <summary>
        /// Whether this sparse GP includes the prior
        /// </summary>
        [DataMember]
        public bool IncludePrior; // defaults to false

        /// <summary>
        /// Use for setting point distribution
        /// </summary>
        [DataMember]
        protected IFunction pointFunc;

        /// <summary>
        /// Field for Alpha property
        /// </summary>
        [DataMember]
        protected Vector alpha;

        /// <summary>
        /// Alpha - along with beta, this encodes the posterior means
        /// and covariances of the Sparse GP
        /// </summary>
        [IgnoreDataMember]
        public Vector Alpha
        {
            get
            {
                if (alpha == null)
                {
                    alpha = FixedParameters.InvKernelOf_B_B*Var_B_B*InducingDist.MeanTimesPrecision;
                }
                return alpha;
            }
        }

        /// <summary>
        /// Field for Beta property
        /// </summary>
        [DataMember]
        protected PositiveDefiniteMatrix beta;

        /// <summary>
        /// Beta - along with alpha, this encodes the posterior means
        /// and covariances of the Sparse GP
        /// </summary>
        [IgnoreDataMember]
        public PositiveDefiniteMatrix Beta
        {
            get
            {
                if (beta == null)
                {
                    bool UseVarBB = InducingDist.Precision.Trace() < double.MaxValue;
                    if (UseVarBB)
                    {
                        beta = new PositiveDefiniteMatrix(FixedParameters.NumberBasisPoints, FixedParameters.NumberBasisPoints);
                        beta.SetToDifference(InducingDist.Precision, InducingDist.Precision * Var_B_B * InducingDist.Precision);
                    }
                    else
                    {
                        beta = GetInverse(FixedParameters.KernelOf_B_B + GetInverse(InducingDist.Precision));
                    }
                }
                return beta;
            }
        }

        private static PositiveDefiniteMatrix GetInverse(PositiveDefiniteMatrix A)
        {
            PositiveDefiniteMatrix result = new PositiveDefiniteMatrix(A.Rows, A.Cols);
            LowerTriangularMatrix L = new LowerTriangularMatrix(A.Rows, A.Cols);
            L.SetToCholesky(A);
            bool[] isZero = new bool[L.Rows];
            for (int i = 0; i < L.Rows; i++)
            {
                if (L[i, i] == 0)
                {
                    isZero[i] = true;
                    L[i, i] = 1;
                }
            }
            L.SetToInverse(L);
            result.SetToOuterTranspose(L);
            for (int i = 0; i < isZero.Length; i++)
            {
                if (isZero[i]) result[i, i] = double.PositiveInfinity;
            }
            return result;
        }

        #endregion

        #region Calculated properties

        /// <summary>
        /// Field for Mean_B property
        /// </summary>
        protected Vector meanB;

        /// <summary>
        /// m(B). This is a calculated Vector maintained
        /// by the class
        /// </summary>
        [IgnoreDataMember]
        public Vector Mean_B
        {
            get
            {
                if (meanB == null)
                {
                    if (IsUniform())
                    {
                        meanB = Vector.Zero(FixedParameters.NumberBasisPoints);
                    }
                    else
                    {
                        meanB = FixedParameters.KernelOf_B_B*Alpha;
                    }
                }
                return meanB;
            }
        }

        /// <summary>
        /// Field for Var_B_B property
        /// </summary>
        protected PositiveDefiniteMatrix varBB;

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// var(B, B). This is a calculated matrix maintained
        /// by the class
        /// </summary>
        [IgnoreDataMember]
        public PositiveDefiniteMatrix Var_B_B
        {
            get
            {
                if (varBB == null)
                {
                    varBB = new PositiveDefiniteMatrix(FixedParameters.NumberBasisPoints, FixedParameters.NumberBasisPoints);
                    if (IsUniform())
                    {
                        varBB.SetAllElementsTo(double.PositiveInfinity);
                    }
                    else
                    {
                        if (false)
                        {
                            PositiveDefiniteMatrix K = FixedParameters.KernelOf_B_B;
                            varBB.SetToDifference(K, K*Beta*K);
                        }
                        else
                        {
                            varBB.SetToInverse(FixedParameters.InvKernelOf_B_B + InducingDist.Precision);
                        }
                    }
                }
                return varBB;
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        #endregion

        #region Flag Recalculate

        /// <summary>
        /// Function to signal recalculation of calculated parameters.
        /// This is called automatically if the fixed parameter
        /// class is swapped out, or if the kernel is changed, or
        /// if parameters are changed. It should also be called
        /// by any external program modifies the kernel or other
        /// fixed parameters in place
        /// </summary>
        public void ClearCachedValues()
        {
            alpha = null;
            beta = null;
            meanB = null;
            varBB = null;
        }

        #endregion

        #region IGPPrediction Members

        /// <summary>
        /// Mean at a given point
        /// </summary>
        /// <param name="X"></param>
        /// <returns></returns>
        public double Mean(Vector X)
        {
            if (IsUniform())
            {
                return Gaussian.Uniform().GetMean();
            }
            else
            {
                Vector kxB = FixedParameters.KernelOf_X_B(X);
                return kxB.Inner(Alpha) + FixedParameters.Prior.Mean(X);
            }
        }

        /// <summary>
        /// Mean at a given list of points
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>Predictive mean vector</returns>
        public Vector Mean(IList<Vector> XList)
        {
            if (IsUniform())
            {
                VectorGaussian temp = new VectorGaussian(XList.Count);
                temp.SetToUniform();
                return temp.GetMean();
            }
            else
            {
                int numPoints = XList.Count;
                Vector result = Vector.Zero(numPoints);
                for (int i = 0; i < numPoints; i++)
                {
                    result[i] = Mean(XList[i]);
                }
                return result;
            }
        }

        /// <summary>
        /// Predictive Variance at a given point
        /// </summary>
        /// <param name="X">Input</param>
        /// <returns>Predictive variance</returns>
        public double Variance(Vector X)
        {
            if (IsUniform())
            {
                Gaussian temp = new Gaussian();
                temp.SetToUniform();
                return temp.GetVariance();
            }
            else
            {
                Vector kxB = FixedParameters.KernelOf_X_B(X);
                return FixedParameters.Prior.Variance(X) - Beta.QuadraticForm(kxB);
            }
        }

        /// <summary>
        /// Predictive covariance at a given pair of points
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public double Covariance(Vector x, Vector y)
        {
            double kxy = FixedParameters.Prior.Covariance(x, y);
            Vector kxB = FixedParameters.KernelOf_X_B(x);
            Vector kyB = FixedParameters.KernelOf_X_B(y);
            return kxy - kxB.Inner(Beta*kyB);
        }

        /// <summary>
        /// Predictive coariance at a given list of points
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>Predictive covariance</returns>
        public PositiveDefiniteMatrix Covariance(IList<Vector> XList)
        {
            if (IsUniform())
            {
                VectorGaussian temp = new VectorGaussian(XList.Count);
                temp.SetToUniform();
                return temp.GetVariance();
            }
            else
            {
                PositiveDefiniteMatrix kXX = FixedParameters.Prior.Covariance(XList);
                Matrix kXB = FixedParameters.KernelOf_X_B(XList);
                kXX.SetToDifference(kXX, kXB*Beta*kXB.Transpose());
                return kXX;
            }
        }

        /// <summary>
        /// Predictive distribution at a given point
        /// </summary>
        /// <param name="X">Input</param>
        /// <returns>Predictive distribution</returns>
        public Gaussian Marginal(Vector X)
        {
            if (IsUniform())
            {
                return Gaussian.Uniform();
            }
            else
            {
                double kxx = FixedParameters.Prior.Variance(X);
                Vector kxb = FixedParameters.KernelOf_X_B(X);
                Gaussian result = new Gaussian(
                    kxb.Inner(Alpha) + FixedParameters.Prior.Mean(X), Math.Max(0, kxx - Beta.QuadraticForm(kxb)));
                return result;
            }
        }

        /// <summary>
        /// Predictive distribution at a given list of points
        /// </summary>
        /// <param name="XList">List of inputs</param>
        /// <returns>Predictive distribution</returns>
        public VectorGaussian Joint(IList<Vector> XList)
        {
            if (IsUniform())
            {
                return VectorGaussian.Uniform(XList.Count);
            }
            else
            {
                PositiveDefiniteMatrix kXX = FixedParameters.Prior.Covariance(XList);
                Matrix kXB = FixedParameters.KernelOf_X_B(XList);
                kXX.SetToDifference(kXX, kXB*Beta*kXB.Transpose());
                VectorGaussian result = new VectorGaussian(kXB*Alpha + FixedParameters.Prior.Mean(XList), kXX);
                return result;
            }
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected SparseGP()
        {
        }

        /// <summary>
        /// Constructs sparse GP, given basis etc
        /// </summary>
        /// <param name="spgf">The fixed parameters</param>
        public SparseGP(SparseGPFixed spgf) : this(spgf, true)
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="spgf">Fixed parameters</param>
        /// <param name="includePrior">Whether this instance includes the prior</param>
        public SparseGP(SparseGPFixed spgf, bool includePrior)
        {
            fixedParameters = spgf;
            InducingDist = new VectorGaussian(spgf.NumberBasisPoints);
            IncludePrior = includePrior;
        }

        /// <summary>
        /// Constructor from full specification
        /// </summary>
        /// <param name="spgf">Fixed parameters</param>
        /// <param name="includePrior">Whether this instance includes the prior</param>
        /// <param name="InducingDist">Inducing distribution</param>
        /// <param name="pointFunc">If not null, set this as a point distribution</param>
        [Construction("FixedParameters", "IncludePrior", "InducingDist", "Point")]
        public SparseGP(SparseGPFixed spgf, bool includePrior, VectorGaussian InducingDist, IFunction pointFunc)
        {
            fixedParameters = spgf;
            IncludePrior = includePrior;
            this.InducingDist = VectorGaussian.Copy(InducingDist);
            this.pointFunc = pointFunc;
        }

        /// <summary>
        /// Creates a uniform sparse GP
        /// </summary>
        /// <param name="sgpf"></param>
        /// <returns></returns>
        public static SparseGP Uniform(SparseGPFixed sgpf)
        {
            return new SparseGP(sgpf, false);
        }

        /// <summary>
        /// Creates a sparse GP point mass - i.e. all the mass is at a given function
        /// </summary>
        /// <param name="sgpf"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static SparseGP PointMass(SparseGPFixed sgpf, IFunction value)
        {
            SparseGP sgp = new SparseGP(sgpf, false);
            sgp.Point = value;
            return sgp;
        }

        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="that"></param>
        public SparseGP(SparseGP that)
        {
            InducingDist = new VectorGaussian(that.FixedParameters.NumberBasisPoints);
            SetTo(that);
        }

        #endregion Constructors

        #region ICloneable Members

        /// <summary>
        /// Clone. Note that the fixed parameters and the rank1 list
        /// are just referenced
        /// </summary>
        /// <returns>The cloned object</returns>
        public object Clone()
        {
            return new SparseGP(this);
        }

        #endregion

        #region SettableTo<SparseGP> Members

        /// <summary>
        /// Sets one sparse GP to another. Everything is copied
        /// except the FixedParameters and the lsit of rank 1 potentials
        /// which are referenced.
        /// </summary>
        /// <param name="that">The sparse GP to copy</param>
        public void SetTo(SparseGP that)
        {
            fixedParameters = that.FixedParameters;
            InducingDist.SetTo(that.InducingDist);
            IncludePrior = that.IncludePrior;
            pointFunc = that.pointFunc;

            if (that.alpha != null)
                alpha = Vector.Copy(that.alpha);
            else
                alpha = null;

            if (that.beta != null)
                beta = new PositiveDefiniteMatrix(that.beta as Matrix);
            else
                beta = null;

            if (that.meanB != null)
                meanB = Vector.Copy(that.meanB);
            else
                meanB = null;

            if (that.varBB != null)
                varBB = new PositiveDefiniteMatrix(that.varBB as Matrix);
            else
                varBB = null;
        }

        #endregion

        /// <summary>
        /// Sets this instance to the product of two sparse GPs.
        /// </summary>
        /// <param name="a">Sparse GP</param>
        /// <param name="b">Sparse GP</param>
        public void SetToProduct(SparseGP a, SparseGP b)
        {
            if (a.FixedParameters != b.FixedParameters)
                throw new ArgumentException("SparseGPs do not have the same FixedParameters.  a.FixedParameters = " + a.FixedParameters + ", b.FixedParameters = " +
                                            b.FixedParameters);
            FixedParameters = a.FixedParameters;
            if (a.IncludePrior && b.IncludePrior) throw new ArgumentException("Both SparseGPs include the prior.  Cannot multiply.");
            IncludePrior = a.IncludePrior || b.IncludePrior;
            if (a.IsPointMass)
            {
                if (b.IsPointMass && !a.Point.Equals(b.Point))
                {
                    throw new AllZeroException();
                }
                Point = a.Point;
            }
            else if (b.IsPointMass)
            {
                Point = b.Point;
            }
            else
            {
                InducingDist.SetToProduct(a.InducingDist, b.InducingDist);
                pointFunc = null;
                ClearCachedValues();
            }
        }

        /// <summary>
        /// Creates a new SparseGP which the product of two other SparseGPs
        /// </summary>
        /// <param name="a">First SparseGP</param>
        /// <param name="b">Second SparseGP</param>
        /// <returns>Result</returns>
        public static SparseGP operator *(SparseGP a, SparseGP b)
        {
            SparseGP result = new SparseGP(a.FixedParameters);
            result.SetToProduct(a, b);
            return result;
        }

        #region SettableToRatio<SparseGPBase> Members

        /// <summary>
        /// Sets this instance to the ratio of two sparse GPs.
        /// </summary>
        /// <param name="numerator">Sparse GP</param>
        /// <param name="denominator">Sparse GP</param>
        /// <param name="forceProper"></param>
        public void SetToRatio(SparseGP numerator, SparseGP denominator, bool forceProper = false)
        {
            if (numerator.FixedParameters != denominator.FixedParameters)
                throw new ArgumentException("SparseGPs do not have the same FixedParameters.  numerator.FixedParameters = " + numerator.FixedParameters +
                                            ", denominator.FixedParameters = " + denominator.FixedParameters);
            FixedParameters = numerator.FixedParameters;
            if (numerator.IncludePrior) IncludePrior = !denominator.IncludePrior;
            else if (denominator.IncludePrior) throw new ArgumentException("Only the denominator includes the prior.  Cannot divide.");
            else IncludePrior = false; // neither include the prior
            if (numerator.IsPointMass)
            {
                if (denominator.IsPointMass)
                {
                    if (numerator.Point.Equals(denominator.Point))
                    {
                        SetToUniform();
                    }
                    else
                    {
                        throw new DivideByZeroException();
                    }
                }
                else
                {
                    Point = numerator.Point;
                }
            }
            else if (denominator.IsPointMass)
            {
                throw new DivideByZeroException();
            }
            else
            {
                // neither is point mass
                InducingDist.SetToRatio(numerator.InducingDist, denominator.InducingDist, forceProper);
                pointFunc = null;
                ClearCachedValues();
            }
        }

        /// <summary>
        /// Creates a new SparseGP which the ratio of two other SparseGPs
        /// </summary>
        /// <param name="numerator">numerator SparseGP</param>
        /// <param name="denominator">denominator SparseGP</param>
        /// <returns>Result</returns>
        public static SparseGP operator /(SparseGP numerator, SparseGP denominator)
        {
            SparseGP result = new SparseGP(numerator.FixedParameters);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        #endregion

        /// <summary>
        /// Sets this sparse GP the the power of another sparse GP
        /// </summary>
        /// <param name="dist"></param>
        /// <param name="exponent"></param>
        public void SetToPower(SparseGP dist, double exponent)
        {
            if (exponent == 1.0) SetTo(dist);
            else
            {
                FixedParameters = dist.FixedParameters;
                if (exponent == 0.0) SetToUniform();
                else if (dist.IsPointMass)
                {
                    if (exponent < 0)
                    {
                        throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                    }
                    else
                    {
                        Point = dist.Point;
                    }
                }
                else if (dist.IncludePrior) throw new ArgumentException("Cannot raise prior to a power.");
                else
                {
                    IncludePrior = dist.IncludePrior;
                    InducingDist.SetToPower(dist.InducingDist, exponent);
                    pointFunc = null;
                    ClearCachedValues();
                }
            }
        }

        #region CanGetLogProb<IFunction> Members

        /// <summary>
        /// Gets the log density for a given value
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double GetLogProb(IFunction value)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        #endregion

        #region HasPoint<IFunction> Members

        /// <summary>
        /// Sets or Gets a point. If not a point function,
        /// the get returns the mean function of the sparse GP
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public IFunction Point
        {
            get { return pointFunc; }
            set
            {
                pointFunc = value;
                ClearCachedValues();
            }
        }

        /// <summary>
        /// Asks the distribution whether it is a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return (pointFunc == null ? false : true); }
        }

        #endregion

        /// <summary>
        /// Evaluates the mean function of the GP
        /// </summary>
        /// <param name="X">Input variable</param>
        /// <returns>Evaulated function value</returns>
        public double EvaluateMean(Vector X)
        {
            if (IsPointMass)
                return Point.Evaluate(X);
            else
                return Mean(X);
        }

        #region SettableToUniform Members

        /// <summary>
        /// Sets to uniform
        /// </summary>
        public void SetToUniform()
        {
            IncludePrior = false;
            InducingDist.SetToUniform();
            pointFunc = null;
            ClearCachedValues();
        }

        /// <summary>
        /// Asks the distribution whether it is uniform
        /// </summary>
        /// <returns>True or false</returns>
        public bool IsUniform()
        {
            return !IncludePrior && InducingDist.IsUniform();
        }

        #endregion

        #region Diffable Members

        /// <summary>
        /// Max difference between two sparse GPs - used for
        /// convergence testing
        /// </summary>
        /// <param name="thatd">That sparse GP which will be compared to this sparse GP</param>
        /// <returns></returns>
        public double MaxDiff(object thatd)
        {
            SparseGP that = thatd as SparseGP;
            // Prior mean and kernel references should be the same
            // Low rank lists are ignored
            if (that == null ||
                this.FixedParameters != that.FixedParameters ||
                this.IncludePrior != that.IncludePrior ||
                this.IsPointMass != that.IsPointMass ||
                this.IsUniform() != that.IsUniform())
                return double.PositiveInfinity;
            if (this.IsUniform() && that.IsUniform())
                return 0.0;
            if (this.IsPointMass)
            {
                // both point masses
                if (this.Point is Diffable) return ((Diffable) this.Point).MaxDiff(that.Point);
                else return (this.Point == that.Point) ? 0.0 : double.PositiveInfinity;
            }
            return this.InducingDist.MaxDiff(that.InducingDist);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            return (MaxDiff(obj) == 0.0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            if (IsPointMass) return Point.GetHashCode();
            int hash = Hash.Start;
            hash = Hash.Combine(hash, FixedParameters.GetHashCode());
            hash = Hash.Combine(hash, IncludePrior.GetHashCode());
            hash = Hash.Combine(hash, pointFunc == null ? Hash.Start : pointFunc.GetHashCode());
            hash = Hash.Combine(hash, InducingDist.GetHashCode());
            return hash;
        }

        #endregion

        #region CanGetMean<IFunction> Members

        /// <summary>
        /// Gets the mean function for the Sparse GP
        /// </summary>
        /// <returns>The mean function</returns>
        public IFunction GetMean()
        {
            return this as IFunction;
        }

        #endregion

        #region Sampleable<IFunction> Members

        /// <summary>
        /// Samples from the Sparse Gaussian distribution
        /// This is only implemented for a 1-dimensional input space,
        /// and returns a simple linear spline function 
        /// </summary>
        /// <returns>Sample function</returns>
        [Stochastic]
        public IFunction Sample()
        {
            if (this.FixedParameters.NumberFeatures > 1)
                throw new Exception("Sampling of a Sparse Gaussian Process is not supported for input spaces of dimension > 1.");

            if (this.FixedParameters.NumberBasisPoints <= 0)
                return ((Sampleable<IFunction>) this.FixedParameters.Prior).Sample();

            // Try to find a reasonable range to sample from
            int numSamplePoints = 51;
            double maxabs = 1.0;
            for (int i = 0; i < this.FixedParameters.NumberBasisPoints; i++)
            {
                double absb = Math.Abs(this.FixedParameters.Basis[i][0]);
                if (maxabs < absb)
                    maxabs = absb;
            }
            maxabs *= 1.5; // Go beyond the basis points
            List<Vector> x = new List<Vector>(numSamplePoints);
            double increm = (2.0*maxabs)/((double) (numSamplePoints - 1));
            double start = -maxabs;
            double currx = start;
            for (int i = 0; i < numSamplePoints; i++)
            {
                Vector xv = Vector.Zero(1);
                xv[0] = currx;
                x.Add(xv);
                currx += increm;
            }

            // x now contains the set of input points at which we'll sample the
            // posterior function distribution
            VectorGaussian vg = VectorGaussian.FromMeanAndVariance(Mean(x), Covariance(x));
            // Sample to get the outputs
            Vector y = vg.Sample();

            // Build the spline
            LinearSpline ls = new LinearSpline();
            ls.KnotStart = start;
            ls.KnotIncrem = increm;
            ls.YPoints = y;

            return ls as IFunction;
        }

        /// <summary>
        /// Samples from the Sparse Gaussian distribution
        /// This is only implemented for a 1-dimensional input space,
        /// and returns a simple linear spline function. result is ignored
        /// <param name="result">This argument is ignored</param>
        /// </summary>
        /// <returns>Sample function</returns>
        [Stochastic]
        public IFunction Sample(IFunction result)
        {
            return Sample();
        }

        #endregion

        #region ToString override

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns>String representation of instance</returns>
        /// <exclude/>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            if (IsPointMass)
            {
                sb.Append("SparseGP.PointMass(");
                sb.Append(pointFunc);
                sb.Append(")");
            }
            else if (IsUniform())
            {
                sb.Append("SparseGP.Uniform");
            }
            else
            {
                sb.Append("SparseGP(");
                sb.Append(FixedParameters);
                sb.Append(")");
                if (Alpha != null)
                {
                    sb.Append("[alpha=");
                    sb.Append(Alpha);
                    sb.Append("]");
                }
            }
            return sb.ToString();
        }

        #endregion

        /// <summary>
        /// Sets this SparseGP distribution to the weighted sum of two other such distributions
        /// </summary>
        /// <param name="weight1"></param>
        /// <param name="value1"></param>
        /// <param name="weight2"></param>
        /// <param name="value2"></param>
        /// <remarks>Not yet implemented</remarks>
        public void SetToSum(double weight1, SparseGP value1, double weight2, SparseGP value2)
        {
            if (value1.FixedParameters != value2.FixedParameters)
                throw new ArgumentException("SparseGPs do not have the same FixedParameters.  a.FixedParameters = " + value1.FixedParameters + ", b.FixedParameters = " +
                                            value2.FixedParameters);
            FixedParameters = value1.FixedParameters;
            if (value1.IncludePrior != value2.IncludePrior) throw new ArgumentException("One Sparse GP includes a prior, the other does not.  Cannot add.");
            IncludePrior = value1.IncludePrior;

            InducingDist.SetToSum(weight1, value1.InducingDist, weight2, value2.InducingDist);
            // The only time the result is a point mass is if both sources are the same point mass
            if (InducingDist.IsPointMass)
                pointFunc = value1.pointFunc;
            else
                pointFunc = null;
            ClearCachedValues();
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Gets the log of the integral of the product of this SparseGP and that SparseGP
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        public double GetLogAverageOf(SparseGP that)
        {
            if (this.FixedParameters != that.FixedParameters)
                throw new ArgumentException("SparseGPs do not have the same FixedParameters.  this.FixedParameters = " + this.FixedParameters + ", that.FixedParameters = " +
                                            that.FixedParameters);
            if (this.IncludePrior && that.IncludePrior) throw new ArgumentException("Both SparseGPs include the prior");
            if (that.IsPointMass) return GetLogProb(that.Point);
            if (this.IsPointMass) return that.GetLogProb(this.Point);
            if (this.IncludePrior && !that.IncludePrior)
            {
                // gBB is the distribution of the function on the basis
                VectorGaussian gBB;
                if (true)
                {
                    gBB = new VectorGaussian(InducingDist.Dimension);
                    gBB.Precision.SetToSum(FixedParameters.InvKernelOf_B_B, InducingDist.Precision);
                    gBB.MeanTimesPrecision.SetTo(InducingDist.MeanTimesPrecision); // since prior has zero mean
                }
                else
                {
                    // equivalent but slower
                    gBB = VectorGaussian.FromMeanAndVariance(Mean_B, Var_B_B);
                }
                return gBB.GetLogAverageOf(that.InducingDist);
            }
            if (!this.IncludePrior && that.IncludePrior) return that.GetLogAverageOf(this);
            throw new NotImplementedException();
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(SparseGP that, double power)
        {
            throw new NotImplementedException();
            if (IsPointMass)
            {
                return power*that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                if (power < 0) throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                return this.GetLogProb(that.Point);
            }
            else
            {
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>
        /// The expected logarithm of that distribution under this distribution
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        /// <remarks>Not yet implemented</remarks>
        public double GetAverageLog(SparseGP that)
        {
            throw new NotImplementedException();
        }
    }

    #region Simple spline class for sampling

    /// <summary>
    /// Very simple 1-D linear spline class which implements IFunction.
    /// Assumes knots at regular positions - given by a start and increment.
    /// The vector of knot values defines how many knots.
    /// </summary>
    public class LinearSpline : IFunction
    {
        private double knotStart;

        /// <summary>
        /// Knot start position
        /// </summary>
        public double KnotStart
        {
            get { return knotStart; }
            set { knotStart = value; }
        }

        private double knotIncrem;
        private double knotIncremInv;

        /// <summary>
        /// Knot position increment
        /// </summary>
        public double KnotIncrem
        {
            get { return knotIncrem; }
            set
            {
                knotIncrem = value;
                knotIncremInv = 1.0/knotIncrem;
            }
        }

        private Vector ypoints;

        /// <summary>
        /// Y points
        /// </summary>
        public Vector YPoints
        {
            get { return ypoints; }
            set
            {
                if (value.Count < 2) throw new ArgumentException($"value.Count ({value.Count}) < 2", nameof(value));
                ypoints = value.Clone();
            }
        }

        #region IFunction Members

        /// <summary>
        /// Evaluate the linear spline at a given point. Only
        /// 1-D input spaces are supported - so only the first element
        /// of X is considered
        /// </summary>
        /// <param name="X"></param>
        /// <returns></returns>
        public double Evaluate(Vector X)
        {
            double x = X[0];
            int n = ypoints.Count;

            // Find knot position index
            double dx = Math.Floor(knotIncremInv*(x - knotStart));
            if (dx <= 0.0)
                dx = 0.0;
            if (dx >= ((double) (n - 1)))
                dx = (double) (n - 1);
            int ix = (int) dx;
            double xi = KnotStart + (dx*KnotIncrem);
            return ypoints[ix] + (knotIncremInv*(ypoints[ix + 1] - ypoints[ix])*(x - xi));
        }

        #endregion
    }

    #endregion
}
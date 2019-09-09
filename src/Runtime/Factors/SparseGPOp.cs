// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "FunctionEvaluate", typeof(IFunction), typeof(Vector))]
    [Quality(QualityBand.Preview)]
    public static class SparseGPOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/message_doc[@name="LogAverageFactor(double, IFunction, Vector)"]/*'/>
        public static double LogAverageFactor(double y, IFunction func, Vector x)
        {
            return (y == func.Evaluate(x)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/message_doc[@name="LogEvidenceRatio(double, IFunction, Vector)"]/*'/>
        public static double LogEvidenceRatio(double y, IFunction func, Vector x)
        {
            return LogAverageFactor(y, func, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/message_doc[@name="AverageLogFactor(double, IFunction, Vector)"]/*'/>
        public static double AverageLogFactor(double y, IFunction func, Vector x)
        {
            return LogAverageFactor(y, func, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/message_doc[@name="LogAverageFactor(double, SparseGP, Vector)"]/*'/>
        public static double LogAverageFactor(double y, SparseGP func, Vector x)
        {
            Gaussian to_y = YAverageConditional(func, x);
            return to_y.GetLogProb(y);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/message_doc[@name="LogEvidenceRatio(double, SparseGP, Vector)"]/*'/>
        public static double LogEvidenceRatio(double y, SparseGP func, Vector x)
        {
            return LogAverageFactor(y, func, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/message_doc[@name="LogEvidenceRatio(Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian y)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/message_doc[@name="YAverageConditional(SparseGP, Vector)"]/*'/>
        public static Gaussian YAverageConditional([SkipIfUniform] SparseGP func, Vector x)
        {
            return func.Marginal(x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/message_doc[@name="FuncAverageConditional(Gaussian, SparseGP, Vector, SparseGP)"]/*'/>
        public static SparseGP FuncAverageConditional([SkipIfUniform] Gaussian y, [SkipIfUniform] SparseGP func, Vector x, SparseGP result)
        {
            if (y.IsUniform() || func.IsUniform())
            {
                result.SetToUniform();
                return result;
            }
            result.FixedParameters = func.FixedParameters;
            result.IncludePrior = false;

            Vector kbx = func.FixedParameters.KernelOf_X_B(x);
            Vector proj = func.FixedParameters.InvKernelOf_B_B * kbx;
            // To avoid computing Var_B_B:
            // vf - func.Var_B_B.QuadraticForm(proj) = kxx - kbx'*beta*kbx - kbx'*inv(K)*Var_B_B*inv(K)*kbx
            // = kxx - kbx'*(beta + inv(K)*Var_B_B*inv(K))*kbx
            // = kxx - kbx'*(beta + inv(K)*(K - K*Beta*K)*inv(K))*kbx
            // = kxx - kbx'*inv(K)*kbx
            // Since Var_B_B = K - K*Beta*K
            double kxx = func.FixedParameters.Prior.Variance(x);
            if (y.Precision == 0)
            {
                result.InducingDist.Precision.SetAllElementsTo(0);
                result.InducingDist.MeanTimesPrecision.SetTo(proj);
                result.InducingDist.MeanTimesPrecision.Scale(y.MeanTimesPrecision);
            }
            else
            {
                double my, vy;
                y.GetMeanAndVariance(out my, out vy);
                double prec = 1.0 / (vy + kxx - kbx.Inner(proj));
                //Console.WriteLine($"{vf - func.Var_B_B.QuadraticForm(proj)} {func.FixedParameters.Prior.Variance(x) - func.FixedParameters.InvKernelOf_B_B.QuadraticForm(kbx)}");
                if (prec > double.MaxValue || prec < 0)
                {
                    int i = proj.IndexOfMaximum();
                    result.InducingDist.Precision.SetAllElementsTo(0);
                    result.InducingDist.Precision[i, i] = double.PositiveInfinity;
                    result.InducingDist.MeanTimesPrecision.SetAllElementsTo(0);
                    result.InducingDist.MeanTimesPrecision[i] = my;
                }
                else
                {
                    result.InducingDist.Precision.SetToOuter(proj, proj);
                    result.InducingDist.Precision.Scale(prec);
                    result.InducingDist.MeanTimesPrecision.SetTo(proj);
                    result.InducingDist.MeanTimesPrecision.Scale(prec * my);
                }
            }
            result.ClearCachedValues();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseGPOp"]/message_doc[@name="FuncAverageConditional(double, SparseGP, Vector, SparseGP)"]/*'/>
        public static SparseGP FuncAverageConditional(double y, [SkipIfUniform] SparseGP func, Vector x, SparseGP result)
        {
            return FuncAverageConditional(Gaussian.PointMass(y), func, x, result);
        }

#if false
    /// <summary>
    /// EP message to 'x'.
    /// </summary>
    /// <param name="y">Incoming message from 'y'.</param>
    /// <param name="func">Incoming message from 'func'.</param>
    /// <param name="x">Constant value for 'x'.</param>
    /// <param name="result">Modified to contain the outgoing message.</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the integral of the factor times incoming messages, over all arguments except 'x'.
    /// The formula is <c>int f(x,x) q(x) dx</c> where <c>x = (y,func)</c>.
    /// </para></remarks>
        public static VectorGaussian XAverageConditional(Gaussian y, SparseGP func, Vector x, VectorGaussian result)
        {
            // Doesn't matter what message we return as we are only supporting a point x
            result.SetToUniform();
            return result;
        }
#endif
    }
}

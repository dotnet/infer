// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "GaussianFromMeanAndVariance")]
    [Quality(QualityBand.Stable)]
    public static class GaussianFromMeanAndVarianceOp
    {
        public static bool ForceProper = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageConditional(double, double, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian SampleAverageConditional(double mean, double variance, TruncatedGaussian result)
        {
            return TruncatedGaussian.FromGaussian(Gaussian.FromMeanAndVariance(mean, variance));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="MeanAverageConditional(double, double, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian MeanAverageConditional(double sample, double variance, TruncatedGaussian result)
        {
            return SampleAverageConditional(sample, variance, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, double)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, double variance)
        {
            if (sample.IsUniform() || mean.IsUniform())
                return 0.0;
            double xm, xv, mm, mv;
            sample.GetMeanAndVariance(out xm, out xv);
            mean.GetMeanAndVariance(out mm, out mv);
            return Gaussian.GetLogProb(xm, mm, variance + xv + mv);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(double, Gaussian, double)"]/*'/>
        public static double LogAverageFactor(double sample, [SkipIfUniform] Gaussian mean, double variance)
        {
            return LogAverageFactor(Gaussian.PointMass(sample), mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(Gaussian, double, double)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian sample, double mean, double variance)
        {
            return LogAverageFactor(sample, Gaussian.PointMass(mean), variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian sample, Gaussian mean, double variance)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, double, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian sample, double mean, double variance)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, [SkipIfUniform] Gaussian mean, double variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double sample, double mean, double variance)
        {
            return Gaussian.GetLogProb(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, double mean, double variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double sample, double mean, double variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageConditional(double, double)"]/*'/>
        public static Gaussian SampleAverageConditional(double mean, double variance)
        {
            return new Gaussian(mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian SampleAverageConditional([SkipIfUniform] Gaussian mean, double variance)
        {
            return GaussianOp.SampleAverageConditional(mean, 1 / variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gamma)"]/*'/>
        public static double LogAverageFactor(Gaussian sample, Gaussian mean, Gamma variance)
        {
            if (variance.IsPointMass)
                return LogAverageFactor(sample, mean, variance.Point);
            double mx, vx, mm, vm;
            double a, b;
            sample.GetMeanAndVariance(out mx, out vx);
            mean.GetMeanAndVariance(out mm, out vm);
            a = variance.Shape;
            b = variance.Rate;
            double c = Math.Sqrt(2 * b);
            double m = c * (mx - mm);
            double v = c * c * (vx + vm);
            double mu, vu;
            if (a == 1)
            {
                double logZ;
                LaplacianTimesGaussianMoments(m, v, out logZ, out mu, out vu);
                logZ += Math.Log(c);
                return logZ;
            }
            throw new NotImplementedException("shape != 1 is not yet implemented");
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gamma, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static double LogEvidenceRatio(Gaussian sample, Gaussian mean, Gamma variance, Gaussian to_sample)
        {
            return LogAverageFactor(sample, mean, variance) - to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageConditional(Gaussian, double, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian SampleAverageConditional([NoInit] Gaussian sample, double mean, [Proper] Gamma variance)
        {
            return SampleAverageConditional(sample, Gaussian.PointMass(mean), variance);
        }

        /// <summary>
        /// Compute moments of 0.5*exp(-abs(x))*N(x;m,v)
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="logZ"></param>
        /// <param name="mu"></param>
        /// <param name="vu"></param>
        public static void LaplacianTimesGaussianMoments(double m, double v, out double logZ, out double mu, out double vu)
        {
            if (Double.IsPositiveInfinity(v))
            {
                // moments of Laplacian only
                logZ = 0;
                mu = 0;
                vu = 2;
                return;
            }
            double invV = 1 / v;
            double invSqrtV = Math.Sqrt(invV);
            double mPlus = (m - v) * invSqrtV;
            double mMinus = (-m - v) * invSqrtV;
            double Z0, Z1, Z2;
            if (mPlus > 30)
            {
                double[] moments = NormalCdfMoments(2, m - v, v);
                Z0 = moments[0];
                Z1 = moments[1];
                Z2 = moments[2];
                logZ = Math.Log(0.5 * Z0) - m + 0.5 * v;
            }
            else if (mMinus > 30)
            {
                double[] moments = NormalCdfMoments(2, -m - v, v);
                Z0 = moments[0];
                Z1 = -moments[1];
                Z2 = moments[2];
                logZ = Math.Log(0.5 * Z0) + m + 0.5 * v;
            }
            else
            {
                Z0 = MMath.NormalCdfRatio(mPlus) + MMath.NormalCdfRatio(mMinus);
                Z1 = Math.Sqrt(v) * (MMath.NormalCdfMomentRatio(1, mPlus) - MMath.NormalCdfMomentRatio(1, mMinus));
                Z2 = 2 * v * (MMath.NormalCdfMomentRatio(2, mPlus) + MMath.NormalCdfMomentRatio(2, mMinus));
                logZ = Math.Log(0.5 * Z0) - MMath.LnSqrt2PI - 0.5 * m * m * invV;
            }
            mu = Z1 / Z0;
            double mu2 = Z2 / Z0;
            vu = mu2 - mu * mu;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 429
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageConditional(Gaussian, Gaussian, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian SampleAverageConditional([NoInit] Gaussian sample, [SkipIfUniform] Gaussian mean, [Proper] Gamma variance)
        {
            if (variance.IsPointMass)
                return SampleAverageConditional(mean, variance.Point);
            if (mean.Precision == 0)
                return Gaussian.Uniform();
            double mx, vx, mm, vm;
            double a, b;
            sample.GetMeanAndVariance(out mx, out vx);
            mean.GetMeanAndVariance(out mm, out vm);
            a = variance.Shape;
            b = variance.Rate;
            if (sample.IsUniform())
                return new Gaussian(mm, vm + a / b);
            if (sample.IsPointMass)
            {
                throw new NotImplementedException();
            }
            double c = Math.Sqrt(2 * b);
            double m = c * (mx - mm);
            double v = c * c * (vx + vm);
            double mu, vu;
            if (a < 1)
                throw new ArgumentException("variance.shape < 1");
            double m2p = m * m / v;
            if (false && Math.Abs(m * m / v) > 30)
            {
                double invV = 1 / v;
                double invSqrtV = Math.Sqrt(invV);
                double mPlus = (m - v) * invSqrtV;
                double mMinus = (-m - v) * invSqrtV;
                double Zplus = MMath.NormalCdfRatio(mPlus);
                double Zminus = MMath.NormalCdfRatio(mMinus);
                double Z0 = Zminus + Zplus;
                double alpha = (Zminus - Zplus) / Z0;
                double beta = -(1 - 2 / Z0 * invSqrtV - alpha * alpha);
                alpha *= c;
                beta *= c * c;
                double weight = beta / (sample.Precision - beta);
                Gaussian result = new Gaussian();
                result.Precision = sample.Precision * weight;
                result.MeanTimesPrecision = weight * (sample.MeanTimesPrecision + alpha) + alpha;
                if (!result.IsProper())
                    throw new ImproperMessageException(result);
                if (double.IsNaN(result.Precision) || double.IsNaN(result.MeanTimesPrecision))
                    throw new InferRuntimeException($"result is NaN.  sample={sample}, mean={mean}, variance={variance}");
                return result;
            }
            else
            {
                //double logZ;
                //if(a == 1) LaplacianTimesGaussianMoments(m, v, out logZ, out mu, out vu);
                VarianceGammaTimesGaussianMoments5(a, m, v, out mu, out vu);
                //Console.WriteLine("mu = {0}, vu = {1}", mu, vu);
                //VarianceGammaTimesGaussianMoments(a, m, v, out mu, out vu);
                //Console.WriteLine("mu = {0}, vu = {1}", mu, vu);
                double r = vx / (vx + vm);
                double mp = r * (mu / c + mm) + (1 - r) * mx;
                double vp = r * r * vu / (c * c) + r * vm;
                if (Double.IsNaN(vp))
                    throw new Exception("vp is NaN");
                Gaussian result = Gaussian.FromMeanAndVariance(mp, vp);
                result.SetToRatio(result, sample, true);
                return result;
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 429
#endif

        public static int nWeights = 10;

        public static void VarianceGammaTimesGaussianMoments2(double a, double m, double v, out double mu, out double vu)
        {
            // compute weights
            Matrix laplacianMoments = new Matrix(nWeights, nWeights);
            DenseVector exactMoments = DenseVector.Constant(laplacianMoments.Rows, 1.0);
            // a=10: 7-1
            // a=15: 8-1
            // a=20: 10-1
            // a=21: 10-1
            // a=30: 12-1
            // get best results if the lead term has flat moment ratio
            int jMax = Math.Max(laplacianMoments.Cols, (int)Math.Round(a - 10)) - 1;
            jMax = laplacianMoments.Cols - 1;
            for (int i = 0; i < exactMoments.Count; i++)
            {
                //int ii = jMax-i;
                int ii = i;
                double logMoment = MMath.GammaLn(ii + a) - MMath.GammaLn(a) - MMath.GammaLn(ii + 1);
                for (int j = 0; j < laplacianMoments.Cols; j++)
                {
                    int jj = jMax - j;
                    laplacianMoments[i, j] = Math.Exp(MMath.GammaLn(2 * ii + jj + 1) - MMath.GammaLn(2 * ii + 1) - MMath.GammaLn(jj + 1) - logMoment);
                }
            }
            //Console.WriteLine("exactMoments = {0}, laplacianMoments = ", exactMoments);
            //Console.WriteLine(laplacianMoments);
            (new LuDecomposition(laplacianMoments)).Solve(exactMoments);
            DenseVector weights = exactMoments;
            Console.WriteLine("weights = {0}", weights);
            double Z0Plus = 0, Z1Plus = 0, Z2Plus = 0;
            double Z0Minus = 0, Z1Minus = 0, Z2Minus = 0;
            double sqrtV = Math.Sqrt(v);
            double InvSqrtV = 1 / sqrtV;
            double mPlus = (m - v) * InvSqrtV;
            double mMinus = (-m - v) * InvSqrtV;
            for (int j = 0; j < weights.Count; j++)
            {
                int jj = jMax - j;
                Z0Plus += weights[j] * MMath.NormalCdfMomentRatio(0 + jj, mPlus) * Math.Pow(sqrtV, 0 + jj);
                Z1Plus += weights[j] * MMath.NormalCdfMomentRatio(1 + jj, mPlus) * (1 + jj) * Math.Pow(sqrtV, 1 + jj);
                Z2Plus += weights[j] * MMath.NormalCdfMomentRatio(2 + jj, mPlus) * (1 + jj) * (2 + jj) * Math.Pow(sqrtV, 2 + jj);
                Z0Minus += weights[j] * MMath.NormalCdfMomentRatio(0 + jj, mMinus) * Math.Pow(sqrtV, 0 + jj);
                Z1Minus += weights[j] * MMath.NormalCdfMomentRatio(1 + jj, mMinus) * (1 + jj) * Math.Pow(sqrtV, 1 + jj);
                Z2Minus += weights[j] * MMath.NormalCdfMomentRatio(2 + jj, mMinus) * (1 + jj) * (2 + jj) * Math.Pow(sqrtV, 2 + jj);
            }
            double Z0 = Z0Plus + Z0Minus;
            double Z1 = Z1Plus - Z1Minus;
            double Z2 = Z2Plus + Z2Minus;
            mu = Z1 / Z0;
            vu = Z2 / Z0 - mu * mu;
        }

        // returns (E[x],var(x)) where p(x) =propto VG(x;a) N(x;m,v).
        public static void VarianceGammaTimesGaussianMoments3(double a, double m, double v, out double mu, out double vu)
        {
            // compute weights
            // termMoments[i,j] is the ith moment of the jth term
            Matrix termMoments = new Matrix(nWeights, nWeights);
            DenseVector exactMoments = DenseVector.Constant(termMoments.Rows, 1.0);
            for (int i = 0; i < exactMoments.Count; i++)
            {
                // ii is half of the exponent
                int ii = i;
                double logMoment = MMath.GammaLn(ii + a) - MMath.GammaLn(a) - MMath.GammaLn(ii + 1);
                for (int j = 0; j < termMoments.Cols; j++)
                {
                    // jj is the term shape
                    int jj = j + 1;
                    termMoments[i, j] = Math.Exp(MMath.GammaLn(ii + jj) - MMath.GammaLn(ii + 1) - MMath.GammaLn(jj) - logMoment);
                }
            }
            //Console.WriteLine("exactMoments = {0}, termMoments = ", exactMoments);
            //Console.WriteLine(termMoments);
            (new LuDecomposition(termMoments)).Solve(exactMoments);
            DenseVector weights = exactMoments;
            Console.WriteLine("weights = {0}", weights);
            double Z0Plus = 0, Z1Plus = 0, Z2Plus = 0;
            double Z0Minus = 0, Z1Minus = 0, Z2Minus = 0;
            for (int j = 0; j < weights.Count; j++)
            {
                int jj = j + 1;
                Z0Plus += weights[j] * NormalVGMomentRatio(0, jj, m - v, v);
                Z1Plus += weights[j] * NormalVGMomentRatio(1, jj, m - v, v);
                Z2Plus += weights[j] * NormalVGMomentRatio(2, jj, m - v, v);
                Z0Minus += weights[j] * NormalVGMomentRatio(0, jj, -m - v, v);
                Z1Minus += weights[j] * NormalVGMomentRatio(1, jj, -m - v, v);
                Z2Minus += weights[j] * NormalVGMomentRatio(2, jj, -m - v, v);
            }
            double Z0 = Z0Plus + Z0Minus;
            double Z1 = Z1Plus - Z1Minus;
            double Z2 = Z2Plus + Z2Minus;
            mu = Z1 / Z0;
            vu = Z2 / Z0 - mu * mu;
            //Console.WriteLine("mu = {0}, vu = {1}", mu, vu);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        // returns (E[x],var(x)) where p(x) =propto VG(x;a) N(x;m,v).
        public static void VarianceGammaTimesGaussianMoments4(double a, double m, double v, out double mu, out double vu)
        {
            // compute weights
            // termMoments[i,j] is the ith moment of the jth term
            Matrix termMoments = new Matrix(nWeights, nWeights);
            DenseVector weights = DenseVector.Constant(termMoments.Rows, 1.0);
            // choose shapes of terms to bracket the true shape 'a'
            int shapeMin = Math.Max(1, (int)Math.Round(a - nWeights / 2));
            for (int j = 0; j < termMoments.Cols; j++)
            {
                // jj is the term shape
                int jj = shapeMin + j;
                double r = jj / a;
                double moment = 1;
                for (int i = 0; i < termMoments.Rows; i++)
                {
                    termMoments[i, j] = moment;
                    moment *= r;
                }
            }
            //Console.WriteLine("exactMoments = {0}, termMoments = ", exactMoments);
            //Console.WriteLine(termMoments);
            (new LuDecomposition(termMoments)).Solve(weights);
            Console.WriteLine("weights = {0}", weights);
            if (false)
            {
                // exact formula for the weights
                for (int i = 0; i < weights.Count; i++)
                {
                    double sum = 0;
                    double term = 1;
                    double numr = a - shapeMin + 1;
                    double denomr = 0;
                    for (int j = 0; j < i; j++)
                    {
                        numr--;
                        denomr++;
                        term *= numr / denomr;
                    }
                    denomr = 0;
                    for (int j = i; j < weights.Count; j++)
                    {
                        //Console.WriteLine("term = {0}", term);
                        sum += term;
                        //Console.WriteLine("sum({0},{1}) = {2}", i,j, sum);
                        numr--;
                        denomr++;
                        term *= -numr / denomr;
                    }
                    weights[i] = sum;
                }
                Console.WriteLine("weights = {0}", weights);
            }
            double Z0Plus = 0, Z1Plus = 0, Z2Plus = 0;
            double Z0Minus = 0, Z1Minus = 0, Z2Minus = 0;
            double[][] momentsPlus = NormalVGMomentRatios(2, shapeMin + weights.Count - 1, m - v, v);
            double[][] momentsMinus = NormalVGMomentRatios(2, shapeMin + weights.Count - 1, -m - v, v);
            for (int j = 0; j < weights.Count; j++)
            {
                int jj = shapeMin + j;
                Z0Plus += weights[j] * momentsPlus[jj - 1][0];
                Z1Plus += weights[j] * momentsPlus[jj - 1][1];
                Z2Plus += weights[j] * momentsPlus[jj - 1][2];
                Z0Minus += weights[j] * momentsMinus[jj - 1][0];
                Z1Minus += weights[j] * momentsMinus[jj - 1][1];
                Z2Minus += weights[j] * momentsMinus[jj - 1][2];
            }
            if (false)
            {
                double[] binomt = new double[nWeights];
                for (int i = 0; i < binomt.Length; i++)
                {
                    binomt[i] = momentsPlus[shapeMin - 1 + i][0];
                }
                double Z0Plus2 = InterpolateBesselKMoment(a - shapeMin + 1, binomt);
                Console.WriteLine("Z0Plus  = {0}", Z0Plus);
                Console.WriteLine("Z0Plus2 = {0}", Z0Plus2);
            }
            //Console.WriteLine("Z1Plus = {0}, Z1Minus = {1}", Z1Plus, Z1Minus);
            double Z0 = Z0Plus + Z0Minus;
            double Z1 = Z1Plus - Z1Minus;
            double Z2 = Z2Plus + Z2Minus;
            mu = Z1 / Z0;
            vu = Z2 / Z0 - mu * mu;
            //Console.WriteLine("mu = {0}, vu = {1}", mu, vu);
        }

        // returns (E[x],var(x)) where p(x) =propto VG(x;a) N(x;m,v).
        public static void VarianceGammaTimesGaussianMoments5(double a, double m, double v, out double mu, out double vu)
        {
            if (false)
            {
                if (a == 1)
                {
                    if ((m - v) / Math.Sqrt(v) > 37.6)
                    {
                        mu = m - v;
                        vu = v;
                        return;
                    }
                    else if ((-m - v) / Math.Sqrt(v) > 37.6)
                    {
                        mu = m + v;
                        vu = v;
                        return;
                    }
                }
                else if (a == 2)
                {
                    if ((m - v) / Math.Sqrt(v) > 37.6)
                    {
                        double mv = m - v;
                        //mu = (mv + mv*mv + v)/(1 + mv);
                        mu = mv + v / (1 + mv);
                        vu = v;
                        return;
                    }
                    else if ((-m - v) / Math.Sqrt(v) > 37.6)
                    {
                        double mv = -m - v;
                        //mu = -(mv + mv*mv + v)/(1 + mv);
                        mu = -(mv + v / (1 + mv));
                        vu = v;
                        return;
                    }
                }
            }
            bool usePlusOnly = ((m - v) / Math.Sqrt(v) > 30);
            bool useMinusOnly = ((-m - v) / Math.Sqrt(v) > 30);
            // choose shapes of terms to bracket the true shape 'a'
            int shapeMin = Math.Max(1, (int)Math.Round(a - nWeights / 2));
            double[][] momentsPlus = NormalVGMomentRatios(2, shapeMin + nWeights - 1, m - v, v);
            if (usePlusOnly)
            {
                momentsPlus = NormalVGMoments(2, shapeMin + nWeights - 1, m - v, v);
            }
            double[] binomt = new double[nWeights];
            for (int i = 0; i < binomt.Length; i++)
            {
                binomt[i] = momentsPlus[i + shapeMin - 1][0];
            }
            double Z0Plus = InterpolateBesselKMoment(a - shapeMin + 1, binomt);
            for (int i = 0; i < binomt.Length; i++)
            {
                binomt[i] = momentsPlus[i + shapeMin - 1][1];
            }
            double Z1Plus = InterpolateBesselKMoment(a - shapeMin + 1, binomt);
            for (int i = 0; i < binomt.Length; i++)
            {
                binomt[i] = momentsPlus[i + shapeMin - 1][2];
            }
            double Z2Plus = InterpolateBesselKMoment(a - shapeMin + 1, binomt);
            if (useMinusOnly)
            {
                Z0Plus = 0;
                Z1Plus = 0;
                Z2Plus = 0;
            }
            double[][] momentsMinus = NormalVGMomentRatios(2, shapeMin + nWeights - 1, -m - v, v);
            if (useMinusOnly)
            {
                momentsMinus = NormalVGMoments(2, shapeMin + nWeights - 1, -m - v, v);
            }
            for (int i = 0; i < binomt.Length; i++)
            {
                binomt[i] = momentsMinus[i + shapeMin - 1][0];
            }
            double Z0Minus = InterpolateBesselKMoment(a - shapeMin + 1, binomt);
            for (int i = 0; i < binomt.Length; i++)
            {
                binomt[i] = momentsMinus[i + shapeMin - 1][1];
            }
            double Z1Minus = InterpolateBesselKMoment(a - shapeMin + 1, binomt);
            for (int i = 0; i < binomt.Length; i++)
            {
                binomt[i] = momentsMinus[i + shapeMin - 1][2];
            }
            double Z2Minus = InterpolateBesselKMoment(a - shapeMin + 1, binomt);
            if (usePlusOnly)
            {
                Z0Minus = 0;
                Z1Minus = 0;
                Z2Minus = 0;
            }
            double Z0 = Z0Plus + Z0Minus;
            double Z1 = Z1Plus - Z1Minus;
            double Z2 = Z2Plus + Z2Minus;
            mu = Z1 / Z0;
            vu = Z2 / Z0 - mu * mu;
            if (Double.IsNaN(vu))
                throw new InferRuntimeException("result is NaN");
            if (vu < 0)
                throw new InferRuntimeException("vu < 0");
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>
        /// Approximate a moment of VG(x;a) by interpolating its values for integer shapes
        /// </summary>
        /// <param name="a">The starting integer shape</param>
        /// <param name="binomt">The exact moment for integer shapes starting at <paramref name="a"/></param>
        /// <returns>The interpolated moment</returns>
        public static double InterpolateBesselKMoment(double a, double[] binomt)
        {
            BinomialTransform(binomt);
            double sum = 0;
            double term = 1;
            double numr = a;
            double denomr = 0;
            for (int j = 0; j < binomt.Length; j++)
            {
                sum += term * binomt[j];
                numr--;
                if (numr == 0)
                    break;
                denomr++;
                term *= -numr / denomr;
            }
            return sum;
        }

        public static void BinomialTransform(double[] x)
        {
            for (int i = 0; i < x.Length; i++)
            {
                for (int j = x.Length - 1; j > i; j--)
                {
                    x[j] = x[j - 1] - x[j];
                }
            }
        }

        // returns phi_n(m,v)/n!
        public static double NormalCdfMoment(int n, double m, double v)
        {
            double InvSqrtV = Math.Sqrt(1 / v);
            return MMath.NormalCdfMomentRatio(n, m * InvSqrtV) * MMath.InvSqrt2PI * Math.Exp(-0.5 * m * m / v) * Math.Pow(v, 0.5 * n);
        }

        public static double NormalCdfMomentRecurrence(int n, double m, double v)
        {
            double prev = Math.Exp(-MMath.LnSqrt2PI - 0.5 * Math.Log(v) - 0.5 * m * m / v);
            double cur = MMath.NormalCdf(m / Math.Sqrt(v));
            for (int i = 0; i < n; i++)
            {
                double next = (m * cur + v * prev) / (i + 1);
                prev = cur;
                cur = next;
            }
            return cur;
        }

        /// <summary>
        /// Computes int_0^Inf x^n N(x;m,v) dx / N(m/sqrt(v);0,1)
        /// </summary>
        /// <param name="nMax"></param>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double[] NormalCdfMomentRatios(int nMax, double m, double v)
        {
            double[] result = new double[nMax + 1];
            double sqrtV = Math.Sqrt(v);
            for (int i = 0; i <= nMax; i++)
            {
                // NormalCdfMomentRatio(0,37.66) = infinity
                result[i] = MMath.NormalCdfMomentRatio(i, m / sqrtV) * Math.Pow(sqrtV, i) * MMath.Gamma(i + 1);
            }
            return result;
        }

        // returns int_0^Inf x^n VG(x;a) N(x;m,v) dx / (0.5*N(m;0,1))
        public static double NormalVGMomentRatio(int n, int a, double m, double v)
        {
            if (a < 1)
                throw new ArgumentException($"{nameof(a)} < 1", nameof(a));
            if (a == 1)
            {
                double sqrtV = Math.Sqrt(v);
                return MMath.Gamma(n + 1) * MMath.NormalCdfMomentRatio(n, m / sqrtV) * Math.Pow(sqrtV, n);
            }
            else if (a == 2)
            {
                double sqrtV = Math.Sqrt(v);
                return 0.5 * NormalVGMomentRatio(n, 1, m, v) + 0.5 * MMath.Gamma(n + 2) * MMath.NormalCdfMomentRatio(n + 1, m / sqrtV) * Math.Pow(sqrtV, n + 1);
            }
            else
            {
                // a > 2
                return 0.25 / ((a - 2) * (a - 1)) * NormalVGMomentRatio(n + 2, a - 2, m, v) + (a - 1.5) / (a - 1) * NormalVGMomentRatio(n, a - 1, m, v);
            }
        }

        /// <summary>
        /// Compute int_0^Inf x^n N(x;m,v) dx for all integer n from 0 to nMax.  Loses accuracy if m &lt; -1.
        /// </summary>
        /// <param name="nMax"></param>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double[] NormalCdfMoments(int nMax, double m, double v)
        {
            double[] result = new double[nMax + 1];
            double sqrtV = Math.Sqrt(v);
            double cur = MMath.NormalCdf(m / sqrtV);
            result[0] = cur;
            if (nMax > 0)
            {
                double prev = cur;
                cur = m * prev + sqrtV * Math.Exp(-MMath.LnSqrt2PI - 0.5 * m * m / v);
                result[1] = cur;
                for (int i = 1; i < nMax; i++)
                {
                    double next = m * cur + v * prev * i;
                    prev = cur;
                    cur = next;
                    result[i + 1] = cur;
                }
            }
            return result;
        }

        /// <summary>
        /// Compute int_0^Inf x^n N(x;m+v,v) VG(x;a) dx *2*exp(m+v/2).  Loses accuracy if m &lt; -1.
        /// </summary>
        /// <param name="nMax"></param>
        /// <param name="aMax"></param>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns>NormalVGMoment[a][n] where a ranges from 1 to aMax, n ranges from 0 to nMax+aMax-a</returns>
        public static double[][] NormalVGMoments(int nMax, int aMax, double m, double v)
        {
            double[] moments1 = NormalCdfMoments(nMax + aMax - 1, m, v);
            return NormalVGMomentTable(nMax, aMax, m, v, moments1);
        }

        /// <summary>
        /// Compute int_0^Inf x^n N(x;m+v,v) VG(x;a) dx *2*exp(m+v/2)/N(m/sqrt(v);0,1)
        /// </summary>
        /// <param name="nMax"></param>
        /// <param name="aMax"></param>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns>NormalVGMoment[a][n] where a ranges from 1 to aMax, n ranges from 0 to nMax+aMax-a</returns>
        public static double[][] NormalVGMomentRatios(int nMax, int aMax, double m, double v)
        {
            double[] moments1 = NormalCdfMomentRatios(nMax + aMax - 1, m, v);
            return NormalVGMomentTable(nMax, aMax, m, v, moments1);
        }

        public static double[][] NormalVGMomentTable(int nMax, int aMax, double m, double v, double[] moments1)
        {
            double[][] moments = new double[aMax][];
            if (aMax < 1)
                throw new ArgumentException($"{nameof(aMax)} < 1", nameof(aMax));
            // if (a == 1) {
            moments[0] = moments1;
            if (aMax >= 2)
            {
                double[] moments2 = new double[nMax + aMax - 1];
                for (int i = 0; i < moments2.Length; i++)
                {
                    moments2[i] = 0.5 * (moments1[i] + moments1[i + 1]);
                }
                moments[1] = moments2;
            }
            if (aMax >= 3)
            {
                for (int a = 3; a <= aMax; a++)
                {
                    double[] momentsCur = new double[nMax + aMax - a + 1];
                    double[] momentsPrev2 = moments[a - 3]; // NormalVGMoments(nMax+2, a-2, m, v);
                    double[] momentsPrev = moments[a - 2]; // NormalVGMoments(nMax, a-1, m, v);
                    for (int i = 0; i < momentsCur.Length; i++)
                    {
                        momentsCur[i] = 0.25 / ((a - 2) * (a - 1)) * momentsPrev2[i + 2] + (a - 1.5) / (a - 1) * momentsPrev[i];
                    }
                    moments[a - 1] = momentsCur;
                }
            }
            return moments;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="MeanAverageConditional(double, double)"]/*'/>
        public static Gaussian MeanAverageConditional(double sample, double variance)
        {
            return SampleAverageConditional(sample, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="MeanAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian MeanAverageConditional([SkipIfUniform] Gaussian sample, double variance)
        {
            return SampleAverageConditional(sample, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="MeanAverageConditional(double, Gaussian, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian MeanAverageConditional(double sample, Gaussian mean, [Proper] Gamma variance)
        {
            return SampleAverageConditional(mean, sample, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="MeanAverageConditional(Gaussian, Gaussian, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian MeanAverageConditional([SkipIfUniform] Gaussian sample, [NoInit] Gaussian mean, [Proper] Gamma variance)
        {
            return SampleAverageConditional(mean, sample, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="VarianceAverageConditional(double, double)"]/*'/>
        public static GammaPower VarianceAverageConditional(double sample, double mean)
        {
            // v^(-0.5)*exp(-0.5*(sample-mean)^2/v)
            double diff = sample-mean;
            double rate = 0.5*diff*diff;
            return new GammaPower(-0.5, 1/rate, -1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="VarianceAverageConditional(double, Gaussian, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gamma VarianceAverageConditional(double sample, [SkipIfUniform] Gaussian mean, [Proper] Gamma variance)
        {
            return VarianceAverageConditional(Gaussian.PointMass(sample), mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="VarianceAverageConditional(Gaussian, double, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gamma VarianceAverageConditional([SkipIfUniform] Gaussian sample, double mean, [Proper] Gamma variance)
        {
            return VarianceAverageConditional(sample, Gaussian.PointMass(mean), variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="VarianceAverageConditional(double, double, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gamma VarianceAverageConditional(double sample, double mean, [Proper] Gamma variance)
        {
            return VarianceAverageConditional(Gaussian.PointMass(sample), Gaussian.PointMass(mean), variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="VarianceAverageConditional(Gaussian, Gaussian, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gamma VarianceAverageConditional([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [Proper] Gamma variance)
        {
            if (sample.Precision == 0 || mean.Precision == 0)
                return Gamma.Uniform();
            double mx, vx, mm, vm;
            sample.GetMeanAndVariance(out mx, out vx);
            mean.GetMeanAndVariance(out mm, out vm);
            if (variance.IsPointMass)
            {
                // f(v) = int_x int_m N(x;m,v) p(x) p(m) dm dx
                //      = N(mx;mm,v+vx+vm)
                // log f(v) = -0.5*log(v+vx+vm) - 0.5*(mx-mm)^2/(v+vx+vm)
                // dlogf = -0.5/(v+vx+vm) + 0.5*(mx-mm)^2/(v+vx+vm)^2
                // ddlogf = 0.5/(v+vx+vm)^2 - (mx-mm)^2/(v+vx+vm)^3
                double vp = variance.Point;
                double denom = 1.0 / (vp + vx + vm);
                if (MMath.AreEqual(denom, 0)) return Gamma.Uniform();
                double vdenom = vp / (vp + vx + vm);
                double mxm = Math.Abs(mx - mm);
                double mxmdenom = MMath.AreEqual(mxm, 0) ? 0 : mxm * denom;
                double mxmvdenom = MMath.AreEqual(vdenom, 0) ? (mxmdenom * vp) : (mxm * vdenom);
                double dlogf = double.IsInfinity(mxmdenom) ? double.PositiveInfinity : (-0.5 * denom + 0.5 * mxmdenom * mxmdenom);
                double xdlogf = -0.5 * vdenom + 0.5 * Math.Max(mxmdenom, mxmvdenom) * Math.Min(mxmdenom, mxmvdenom);
                double xxddlogf = 0.5 * vdenom * vdenom - (MMath.AreEqual(mxmvdenom, 0) ? 0 : denom * mxmvdenom * mxmvdenom);
                return Gamma.FromDerivatives(vp, dlogf, xdlogf, xxddlogf, ForceProper);
            }
            double a = variance.Shape;
            double b = variance.Rate;
            double c = Math.Sqrt(2 * b);
            double m = c * (mx - mm);
            double v = c * c * (vx + vm);
            double Z = VarianceGammaTimesGaussianIntegral(a, m, v);
            double Zplus1 = VarianceGammaTimesGaussianIntegral(a + 1, m, v);
            double Zplus2 = VarianceGammaTimesGaussianIntegral(a + 2, m, v);
            double vmp = a / b * Zplus1 / Z;
            double vm2p = a * (a + 1) / (b * b) * Zplus2 / Z;
            double vvp = vm2p - vmp * vmp;
            Gamma result = Gamma.FromMeanAndVariance(vmp, vvp);
            result.SetToRatio(result, variance, true);
            if (result.Shape < 1)
            {
                // force result.Shape = 1, choose Rate to match mean only
                result = Gamma.FromShapeAndRate(variance.Shape, variance.Shape / vmp);
                result.SetToRatio(result, variance, true);
            }
            return result;
        }

        private static Gamma FromDerivativesCheck(double x, double dLogP, double xdLogP, double xxddLogP, bool forceProper)
        {
            Gamma result = Gamma.FromDerivatives(x, dLogP, xdLogP, xxddLogP, forceProper);
            result.GetDerivatives(x, out double dlogp2, out double ddlogp2);
            bool bothInfinity = double.IsInfinity(dLogP) && (dLogP == dlogp2);
            if (!bothInfinity && MMath.AbsDiff(xdLogP, x*dlogp2, 1e-10) > 1e-1 && MMath.AbsDiff(2*xdLogP, 2*x * dlogp2, 1e-10) > 1e-1) throw new Exception();
            return result;
        }

        /// <summary>
        /// Compute int_{-Inf}^{Inf} N(x;m,v) VG(x;a) dx * 2/N(m/sqrt(v);0,1)
        /// </summary>
        /// <param name="a"></param>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double VarianceGammaTimesGaussianIntegral(double a, double m, double v)
        {
            bool usePlusOnly = ((m - v) / Math.Sqrt(v) > 37.6);
            bool useMinusOnly = ((-m - v) / Math.Sqrt(v) > 37.6);
            // choose shapes of terms to bracket the true shape 'a'
            int shapeMin = Math.Max(1, (int)Math.Round(a - nWeights / 2));
            // momentsPlus[i][j] = int_0^inf x^j N(x;m,v) VG(x;i+1) dx *2*exp(m-v/2) / N((m-v)/sqrt(v);0,1)
            // the scale factor is sqrt(2*pi)*exp(0.5*(m^2 -2*m*v + v^2)/v))*2*exp(m-v/2) 
            //  = sqrt(2*pi)*exp(0.5*m^2/v)*2 = 2/N(m/sqrt(v);0,1)
            double[][] momentsPlus = NormalVGMomentRatios(2, shapeMin + nWeights - 1, m - v, v);
            if (usePlusOnly)
            {
                // here the scale factor is 2*exp(m-v/2)
                momentsPlus = NormalVGMoments(2, shapeMin + nWeights - 1, m - v, v);
            }
            double[] binomt = new double[nWeights];
            for (int i = 0; i < binomt.Length; i++)
            {
                binomt[i] = momentsPlus[i + shapeMin - 1][0];
            }
            double Z0Plus = InterpolateBesselKMoment(a - shapeMin + 1, binomt);
            if (useMinusOnly)
                Z0Plus = 0;
            // momentsMinus[i][j] = int_0^inf x^j N(x;-m,v) VG(x;i+1) dx *2*exp(-m-v/2) / N((-m-v)/sqrt(v);0,1)
            // the scale factor is the same as above
            double[][] momentsMinus = NormalVGMomentRatios(2, shapeMin + nWeights - 1, -m - v, v);
            if (useMinusOnly)
            {
                momentsMinus = NormalVGMoments(2, shapeMin + nWeights - 1, -m - v, v);
            }
            for (int i = 0; i < binomt.Length; i++)
            {
                binomt[i] = momentsMinus[i + shapeMin - 1][0];
            }
            double Z0Minus = InterpolateBesselKMoment(a - shapeMin + 1, binomt);
            if (usePlusOnly)
                Z0Minus = 0;
            double Z0 = Z0Plus + Z0Minus;
            return Z0;
        }

        //-- VMP ---------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(Gaussian, Gaussian, double)"]/*'/>
        public static double AverageLogFactor([Proper] Gaussian sample, [Proper] Gaussian mean, double variance)
        {
            return GaussianOp.AverageLogFactor(sample, mean, 1.0 / variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(double, Gaussian, double)"]/*'/>
        public static double AverageLogFactor(double sample, [Proper] Gaussian mean, double variance)
        {
            return GaussianOp.AverageLogFactor(sample, mean, 1.0 / variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(Gaussian, double, double)"]/*'/>
        public static double AverageLogFactor([Proper] Gaussian sample, double mean, double variance)
        {
            return GaussianOp.AverageLogFactor(sample, mean, 1.0 / variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian SampleAverageLogarithm([SkipIfUniform] Gaussian mean, double variance)
        {
            return GaussianOp.SampleAverageLogarithm(mean, 1.0 / variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageLogarithm(double, double)"]/*'/>
        public static Gaussian SampleAverageLogarithm(double mean, double variance)
        {
            return Gaussian.FromMeanAndVariance(mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="MeanAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian MeanAverageLogarithm([SkipIfUniform] Gaussian sample, double variance)
        {
            return SampleAverageLogarithm(sample, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp"]/message_doc[@name="MeanAverageLogarithm(double, double)"]/*'/>
        public static Gaussian MeanAverageLogarithm(double sample, double variance)
        {
            return SampleAverageLogarithm(sample, variance);
        }
    }

    /// <summary>
    /// This class defines specializations for the case where variance is a point mass.
    /// These methods have fewer inputs, allowing more efficient schedules.
    /// </summary>
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp_PointVariance"]/doc/*'/>
    [FactorMethod(typeof(Factor), "GaussianFromMeanAndVariance", Default = false)]
    [Quality(QualityBand.Preview)]
    public static class GaussianFromMeanAndVarianceOp_PointVariance
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp_PointVariance"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, Gamma)"]/*'/>
        public static double LogEvidenceRatio(
            double sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma variance)
        {
            if (!variance.IsPointMass)
                throw new ArgumentException($"{nameof(variance)} is not a point mass");
            return GaussianFromMeanAndVarianceOp.LogEvidenceRatio(sample, mean, variance.Point);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp_PointVariance"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gamma)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(
            [SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma variance)
        {
            if (!variance.IsPointMass)
                throw new ArgumentException($"{nameof(variance)} is not a point mass");
            return GaussianFromMeanAndVarianceOp.LogEvidenceRatio(sample, mean, variance.Point);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp_PointVariance"]/message_doc[@name="VarianceAverageConditional(Gaussian, Gaussian, Gamma)"]/*'/>
        public static Gamma VarianceAverageConditional([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [Proper] Gamma variance)
        {
            if (!variance.IsPointMass)
                throw new ArgumentException($"{nameof(variance)} is not a point mass");
            return GaussianFromMeanAndVarianceOp.VarianceAverageConditional(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp_PointVariance"]/message_doc[@name="SampleAverageConditional(Gaussian, Gamma)"]/*'/>
        public static Gaussian SampleAverageConditional([SkipIfUniform] Gaussian mean, [Proper] Gamma variance)
        {
            if (!variance.IsPointMass)
                throw new ArgumentException($"{nameof(variance)} is not a point mass");
            return GaussianFromMeanAndVarianceOp.SampleAverageConditional(mean, variance.Point);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianFromMeanAndVarianceOp_PointVariance"]/message_doc[@name="MeanAverageConditional(Gaussian, Gamma)"]/*'/>
        public static Gaussian MeanAverageConditional([SkipIfUniform] Gaussian sample, [Proper] Gamma variance)
        {
            return SampleAverageConditional(sample, variance);
        }
    }
}

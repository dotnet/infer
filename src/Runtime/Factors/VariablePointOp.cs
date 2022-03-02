// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Diagnostics;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOpBase"]/doc/*'/>
    [Quality(QualityBand.Preview)]
    public class VariablePointOpBase
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOpBase"]/message_doc[@name="LogEvidenceRatio{TDist}(TDist, TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static double LogEvidenceRatio<TDist>(TDist use, TDist def, TDist to_marginal)
            where TDist : CanGetLogAverageOf<TDist>
        {
            //return def.GetLogAverageOf(to_marginal);
            return def.GetLogAverageOf(use) - use.GetLogAverageOf(to_marginal);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOpBase"]/message_doc[@name="MarginalAverageConditionalInit{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        [Skip]
        public static TDist MarginalAverageConditionalInit<TDist>([IgnoreDependency] TDist def)
            where TDist : ICloneable
        {
            return (TDist)def.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOpBase"]/message_doc[@name="UseAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static TDist UseAverageConditional<TDist>([IsReturned] TDist to_marginal)
        {
            return to_marginal;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOpBase"]/message_doc[@name="DefAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static TDist DefAverageConditional<TDist>([IsReturned] TDist to_marginal)
        {
            return to_marginal;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOpBase"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOpBase"]/message_doc[@name="UseAverageLogarithm{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static TDist UseAverageLogarithm<TDist>([IsReturned] TDist to_marginal)
        {
            return to_marginal;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOpBase"]/message_doc[@name="DefAverageLogarithm{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        public static TDist DefAverageLogarithm<TDist>([IsReturned] TDist to_marginal)
        {
            return to_marginal;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable.</typeparam>
    [FactorMethod(typeof(Clone), "VariablePoint<>", Default = false)]
    [Quality(QualityBand.Preview)]
    public class VariablePointOp<T> : VariablePointOpBase
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp{T}"]/message_doc[@name="MarginalAverageConditional{TDist}(TDist, TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        [SkipIfAllUniform]
        public static TDist MarginalAverageConditional<TDist>([NoInit] TDist use, TDist def, TDist result)
            where TDist : SettableToProduct<TDist>, HasPoint<T>, CanGetMode<T>
        {
            result.SetToProduct(def, use);
            result.Point = result.GetMode();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp{T}"]/message_doc[@name="MarginalAverageLogarithm{TDist}(TDist, TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        [SkipIfAllUniform]
        public static TDist MarginalAverageLogarithm<TDist>([NoInit] TDist use, TDist def, TDist result)
            where TDist : SettableToProduct<TDist>, HasPoint<T>, CanGetMode<T>
        {
            return MarginalAverageConditional(use, def, result);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp_Mean{T}"]/doc/*'/>
    [FactorMethod(typeof(Clone), "VariablePoint<>", Default = false)]
    [Quality(QualityBand.Preview)]
    public class VariablePointOp_Mean<T> : VariablePointOpBase
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp_Mean{T}"]/message_doc[@name="MarginalAverageConditional{TDist}(TDist, TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        [SkipIfAllUniform]
        public static TDist MarginalAverageConditional<TDist>([NoInit] TDist use, TDist def, TDist result)
            where TDist : SettableToProduct<TDist>, HasPoint<T>, CanGetMean<T>
        {
            result.SetToProduct(def, use);
            result.Point = result.GetMean();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp_Mean{T}"]/message_doc[@name="MarginalAverageLogarithm{TDist}(TDist, TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the marginal of the variable.</typeparam>
        [SkipIfAllUniform]
        public static TDist MarginalAverageLogarithm<TDist>([NoInit] TDist use, TDist def, TDist result)
            where TDist : SettableToProduct<TDist>, HasPoint<T>, CanGetMean<T>
        {
            return MarginalAverageConditional(use, def, result);
        }
    }

    /// <summary>
    /// Implements the Rprop root-finding algorithm.
    /// </summary>
    /// <remarks>
    /// Reference: 
    /// "A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm"
    /// Martin Riedmiller, Heinrich Braun
    /// 1993
    /// </remarks>
    public class RpropBufferData
    {
        public static bool debug;
        bool hasPrevious;
        double prevPoint;
        double prevSign2 = double.NaN, prevSign3 = double.NaN;
        public static bool EnsureConvergence = false;
        int updateCount = 0;
        int updateLimit = int.MaxValue;
        int recoveryCount = 10;
        int bounceCount = 0;
        int previousUpdateCount = 0;
        public double nextPoint;
        double scaleUp = 2;
        int boundScaleUp = 2;
        double stepsize;
        double stepsizeUpperBound = double.PositiveInfinity;
        public double lowerBound = double.NegativeInfinity, upperBound = double.PositiveInfinity;

        public void SetNextPoint(double currPoint, double currDeriv)
        {
            if (double.IsNaN(currPoint)) throw new ArgumentException("currPoint is NaN");
            if (double.IsInfinity(currPoint)) throw new ArgumentException("currPoint is infinite");
            if (double.IsNaN(currDeriv)) throw new ArgumentException("currDeriv is NaN");
            if (hasPrevious)
            {
                double prevStep = currPoint - prevPoint;
                double prevSign = Math.Sign(prevStep);
                double currSign = Math.Sign(currDeriv);
                if (prevSign != 0 && currSign != 0)
                {
                    if (prevSign != currSign)
                    {
                        scaleUp = 1.2;
                        stepsize /= 2;
                        // could return here to introduce drag
                        // does not work with Delay
                        //prevPoint = currPoint;
                        //return;
                    }
                    else
                    {
                        // prevSign == currSign
                        stepsize *= scaleUp;
                        stepsizeUpperBound = Math.Min(stepsizeUpperBound, (upperBound - lowerBound) / 100);
                        stepsize = Math.Min(stepsize, stepsizeUpperBound);
                    }
                    updateCount++;
                    if (EnsureConvergence)
                    {
                        if (!double.IsNaN(prevSign3) && currSign == prevSign && currSign == prevSign2 && currSign != prevSign3)
                        {
                            stepsize /= scaleUp;
                            stepsizeUpperBound = stepsize;
                            if (debug)
                                Trace.WriteLine($"reducing stepsizeUpperBound to {stepsizeUpperBound}, {updateCount - previousUpdateCount} updates between");
                            bounceCount++;
                            recoveryCount *= 2;
                            updateLimit = updateCount + recoveryCount;
                            previousUpdateCount = updateCount;
                        }
                        else if (updateCount >= updateLimit)
                        {
                            stepsizeUpperBound = Math.Min(stepsizeUpperBound * boundScaleUp, (upperBound - lowerBound) / 100);
                            if (debug)
                                Trace.WriteLine($"increasing stepsizeUpperBound to {stepsizeUpperBound}, {updateCount - previousUpdateCount} updates between");
                            updateLimit = updateCount + recoveryCount;
                        }
                        prevSign3 = prevSign2;
                        prevSign2 = prevSign;
                    }
                }
            }
            else
            {
                stepsize = StepsizeLowerBound(currPoint);
            }
            prevPoint = currPoint;
            hasPrevious = true;
            currPoint = Math.Min(Math.Max(currPoint, lowerBound), upperBound);
            while (true)
            {
                nextPoint = currPoint + Math.Sign(currDeriv) * stepsize;
                if (nextPoint >= lowerBound && nextPoint <= upperBound)
                {
                    // Valid update
                    break;
                }
                // Shrink stepsize and try again
                double nextStepsize = stepsize / 2;
                // If stepsize is too small, currPoint will never move in the future
                if (nextStepsize < StepsizeLowerBound(currPoint))
                {
                    nextPoint = Math.Min(Math.Max(nextPoint, lowerBound), upperBound);
                    break;
                }
                stepsize = nextStepsize;
            }
            if (debug)
            {
                Trace.WriteLine($"nextPoint = {nextPoint}, currPoint = {currPoint}, currDeriv = {currDeriv}, stepsize = {stepsize}, lowerBound = {lowerBound}, upperBound = {upperBound}");
                Trace.WriteLine($"bounce count = {bounceCount} out of {updateCount} updates");
            }

            double StepsizeLowerBound(double x)
            {
                return (x == 0) ? 1e-4 : Math.Max(double.Epsilon, Math.Abs(x) * 1e-4);
            }
        }
    }

    [FactorMethod(typeof(Clone), "VariablePoint<>", Default = true)]
    [Buffers("buffer")]
    [Quality(QualityBand.Preview)]
    public class VariablePointOp_Rprop : VariablePointOpBase
    {
        [Skip]
        public static RpropBufferData BufferInit()
        {
            return new RpropBufferData();
        }

        // must compute new marginal here since Marginal method cannot modify buffer
        // Buffer must be called once per iter, so cannot be fresh or have triggers
        [SkipIfAllUniform]
        public static RpropBufferData Buffer([NoInit] Gaussian use, Gaussian def, Gaussian to_marginal, RpropBufferData buffer)
        {
            var currDist = use * def;
            if (currDist.IsPointMass)
            {
                buffer.nextPoint = currDist.Point;
                return buffer;
            }
            // cannot use buffer.nextPoint as currPoint since Marginal could be initialized by user
            double currPoint;
            if (to_marginal.IsPointMass)
            {
                currPoint = to_marginal.Point;
            }
            else
            {
                currPoint = currDist.GetMean();
            }
            // deriv of -0.5*prec*x^2+pm*x
            // is -prec*x + pm
            double currDeriv = currDist.MeanTimesPrecision - currDist.Precision * currPoint;
            buffer.SetNextPoint(currPoint, currDeriv);
            return buffer;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp_Rprop"]/message_doc[@name="MarginalAverageConditional(Gaussian, Gaussian, RpropBufferData, Gaussian)"]/*'/>
        public static Gaussian MarginalAverageConditional([IgnoreDependency] Gaussian use, [IgnoreDependency] Gaussian def, [RequiredArgument] RpropBufferData buffer, Gaussian result)
        {
            result.Point = buffer.nextPoint;
            return result;
        }
    }

    [FactorMethod(typeof(Clone), "VariablePoint<>", Default = true)]
    [Buffers("bufferTG")]
    [Quality(QualityBand.Preview)]
    public class VariablePointOp_RpropTruncatedGaussian : VariablePointOpBase
    {
        [Skip]
        public static RpropBufferData BufferTGInit()
        {
            return new RpropBufferData();
        }

        // must compute new marginal here since Marginal method cannot modify buffer
        // Buffer must be called once per iter, so cannot be fresh or have triggers
        [SkipIfAllUniform]
        public static RpropBufferData BufferTG([NoInit] TruncatedGaussian use, TruncatedGaussian def, TruncatedGaussian to_marginal, RpropBufferData bufferTG)
        {
            var currDist = use * def;
            if (currDist.IsPointMass)
            {
                if (double.IsInfinity(currDist.Point)) throw new ArgumentOutOfRangeException(nameof(use), "infinite point mass");
                bufferTG.nextPoint = currDist.Point;
                return bufferTG;
            }
            // cannot use buffer.nextPoint as currPoint since Marginal could be initialized by user
            double currPoint;
            if (to_marginal.IsPointMass)
            {
                currPoint = to_marginal.Point;
            }
            else
            {
                currPoint = currDist.GetMean();
            }
            // deriv of -0.5*prec*x^2+pm*x
            // is -prec*x + pm
            double currDeriv = currDist.Gaussian.MeanTimesPrecision - currDist.Gaussian.Precision * currPoint;
            bufferTG.lowerBound = currDist.LowerBound;
            bufferTG.upperBound = currDist.UpperBound;
            bufferTG.SetNextPoint(currPoint, currDeriv);
            return bufferTG;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp_RpropTruncatedGaussian"]/message_doc[@name="MarginalAverageConditional(TruncatedGaussian, TruncatedGaussian, RpropBufferData, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian MarginalAverageConditional([IgnoreDependency] TruncatedGaussian use, [IgnoreDependency] TruncatedGaussian def, [RequiredArgument] RpropBufferData bufferTG, TruncatedGaussian result)
        {
            result.Point = bufferTG.nextPoint;
            return result;
        }
    }

    [FactorMethod(typeof(Clone), "VariablePoint<>", Default = true)]
    [Buffers("bufferTGa")]
    [Quality(QualityBand.Preview)]
    public class VariablePointOp_RpropTruncatedGamma : VariablePointOpBase
    {
        [Skip]
        public static RpropBufferData BufferTGaInit()
        {
            var result = new RpropBufferData();
            if (!VariablePointOp_RpropGamma.UseMean)
                result.lowerBound = 0;
            return result;
        }

        // must compute new marginal here since Marginal method cannot modify buffer
        // Buffer must be called once per iter, so cannot be fresh or have triggers
        [SkipIfAllUniform]
        public static RpropBufferData BufferTGa([NoInit] TruncatedGamma use, TruncatedGamma def, TruncatedGamma to_marginal, RpropBufferData bufferTGa)
        {
            var currDist = use * def;
            if (currDist.IsPointMass)
            {
                if (double.IsInfinity(currDist.Point) || double.IsNaN(currDist.Point))
                    throw new ArgumentOutOfRangeException();
                if (VariablePointOp_RpropGamma.UseMean)
                    bufferTGa.nextPoint = Math.Log(currDist.Point);
                else
                    bufferTGa.nextPoint = currDist.Point;
                return bufferTGa;
            }
            // cannot use buffer.nextPoint as currPoint since Marginal could be initialized by user
            double currPoint;
            if (to_marginal.IsPointMass)
            {
                currPoint = to_marginal.Point;
            }
            else
            {
                currPoint = currDist.GetMean();
            }
            double currDeriv, currDeriv2;
            if (VariablePointOp_RpropGamma.UseMean)
            {
                bufferTGa.lowerBound = Math.Log(currDist.LowerBound);
                bufferTGa.upperBound = Math.Log(currDist.UpperBound);
                currDeriv = currDist.Gamma.Shape - currDist.Gamma.Rate * currPoint;
                //Trace.WriteLine($"use deriv = {(use.Gamma.Shape-1) - use.Gamma.Rate*currPoint} def deriv = {def.Gamma.Shape - def.Gamma.Rate*currPoint} total deriv = {currDeriv}");
                if (currPoint <= 0)
                    throw new ArgumentException($"currPoint ({currPoint}) <= 0");
                bufferTGa.SetNextPoint(Math.Log(currPoint), currDeriv);
            }
            else
            {
                bufferTGa.lowerBound = currDist.LowerBound;
                bufferTGa.upperBound = currDist.UpperBound;
                currDist.Gamma.GetDerivatives(currPoint, out currDeriv, out currDeriv2);
                bufferTGa.SetNextPoint(currPoint, currDeriv);
            }
            return bufferTGa;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp_RpropTruncatedGamma"]/message_doc[@name="MarginalAverageConditional(TruncatedGamma, TruncatedGamma, RpropBufferData, TruncatedGamma)"]/*'/>
        public static TruncatedGamma MarginalAverageConditional([IgnoreDependency] TruncatedGamma use, [IgnoreDependency] TruncatedGamma def, [RequiredArgument] RpropBufferData bufferTGa, TruncatedGamma result)
        {
            if (VariablePointOp_RpropGamma.UseMean)
            {
                result.Point = Math.Exp(bufferTGa.nextPoint);
            }
            else
            {
                result.Point = bufferTGa.nextPoint;
            }
            return result;
        }
    }

    [FactorMethod(typeof(Clone), "VariablePoint<>", Default = true)]
    [Buffers("buffer0")]
    [Quality(QualityBand.Preview)]
    public class VariablePointOp_RpropGamma : VariablePointOpBase
    {
        public static bool UseMean;

        [Skip]
        public static RpropBufferData Buffer0Init()
        {
            var result = new RpropBufferData();
            if (!UseMean)
                result.lowerBound = 0;
            return result;
        }

        // must compute new marginal here since Marginal method cannot modify buffer
        // Buffer must be called once per iter, so cannot be fresh or have triggers
        [SkipIfAllUniform]
        public static RpropBufferData Buffer0([NoInit] Gamma use, Gamma def, Gamma to_marginal, RpropBufferData buffer0)
        {
            var currDist = use * def;
            if (currDist.IsPointMass)
            {
                if (UseMean)
                    buffer0.nextPoint = Math.Log(currDist.Point);
                else
                    buffer0.nextPoint = currDist.Point;
                return buffer0;
            }
            // cannot use buffer.nextPoint as currPoint since Marginal could be initialized by user
            double currPoint;
            if (to_marginal.IsPointMass)
            {
                currPoint = to_marginal.Point;
            }
            else
            {
                currPoint = currDist.GetMean();
            }
            double currDeriv, currDeriv2;
            if (UseMean)
            {
                currDeriv = currDist.Shape - currDist.Rate * currPoint;
                if (currPoint <= 0)
                    throw new ArgumentException($"currPoint ({currPoint}) <= 0");
                buffer0.SetNextPoint(Math.Log(currPoint), currDeriv);
            }
            else
            {
                currDist.GetDerivatives(currPoint, out currDeriv, out currDeriv2);
                buffer0.SetNextPoint(currPoint, currDeriv);
            }
            return buffer0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp_RpropGamma"]/message_doc[@name="MarginalAverageConditional(Gamma, Gamma, RpropBufferData, Gamma)"]/*'/>
        public static Gamma MarginalAverageConditional([IgnoreDependency] Gamma use, [IgnoreDependency] Gamma def, [RequiredArgument] RpropBufferData buffer0, Gamma result)
        {
            if (UseMean)
            {
                result.Point = Math.Exp(buffer0.nextPoint);
            }
            else
            {
                result.Point = buffer0.nextPoint;
            }
            return result;
        }
    }

    [FactorMethod(typeof(Clone), "VariablePoint<>", Default = true)]
    [Buffers("bufferBeta")]
    [Quality(QualityBand.Preview)]
    public class VariablePointOp_RpropBeta : VariablePointOpBase
    {
        [Skip]
        public static RpropBufferData BufferBetaInit()
        {
            return new RpropBufferData() { lowerBound = 0, upperBound = 1 };
        }

        // must compute new marginal here since Marginal method cannot modify buffer
        // Buffer must be called once per iter, so cannot be fresh or have triggers
        [SkipIfAllUniform]
        public static RpropBufferData BufferBeta([NoInit] Beta use, Beta def, Beta to_marginal, RpropBufferData bufferBeta)
        {
            var currDist = use * def;
            if (currDist.IsPointMass)
            {
                bufferBeta.nextPoint = currDist.Point;
                return bufferBeta;
            }
            // cannot use buffer.nextPoint as currPoint since Marginal could be initialized by user
            double currPoint;
            if (to_marginal.IsPointMass)
            {
                currPoint = to_marginal.Point;
            }
            else
            {
                currPoint = currDist.GetMean();
            }
            double currDeriv, currDeriv2;
            currDist.GetDerivatives(currPoint, out currDeriv, out currDeriv2);
            bufferBeta.SetNextPoint(currPoint, currDeriv);
            return bufferBeta;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="VariablePointOp_RpropBeta"]/message_doc[@name="MarginalAverageConditional(Beta, Beta, RpropBufferData, Beta)"]/*'/>
        public static Beta MarginalAverageConditional([IgnoreDependency] Beta use, [IgnoreDependency] Beta def, [RequiredArgument] RpropBufferData bufferBeta, Beta result)
        {
            result.Point = bufferBeta.nextPoint;
            return result;
        }
    }
}

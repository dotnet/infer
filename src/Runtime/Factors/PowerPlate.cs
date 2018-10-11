// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <summary>
    /// Power plate factor method
    /// </summary>
    [Hidden]
    public static class PowerPlate
    {
        /// <summary>
        /// Copy a value from outside to the inside of a power plate.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="exponent"></param>
        /// <returns>A copy of value.</returns>
        public static T Enter<T>([IsReturned] T value, double exponent)
        {
            return value;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerPlateOp"]/doc/*'/>
    [FactorMethod(typeof(PowerPlate), "Enter<>")]
    [Quality(QualityBand.Preview)]
    public static class PowerPlateOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerPlateOp"]/message_doc[@name="ValueAverageConditional{T}(T, double, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        public static T ValueAverageConditional<T>([SkipIfUniform] T enter, double exponent, T result)
            where T : SettableToPower<T>
        {
            result.SetToPower(enter, exponent);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerPlateOp"]/message_doc[@name="EnterAverageConditional{T}(T, T, double, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        [SkipIfAllUniform("enter", "value")]
        public static T EnterAverageConditional<T>([NoInit, Cancels] T enter, T value, double exponent, T result)
            where T : SettableToPower<T>, SettableToProduct<T>
        {
            if (exponent == 0)
            {
                // it doesn't matter what we return in this case, so we return something proper
                // to avoid spurious improper message exceptions
                result.SetToPower(value, 1.0);
            }
            else
            {
                // to_enter = value*enter^(exponent-1)
                result.SetToPower(enter, exponent - 1);
                result.SetToProduct(value, result);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerPlateOp"]/message_doc[@name="EnterAverageConditionalInit{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        [Skip]
        public static T EnterAverageConditionalInit<T>([IgnoreDependency] T value)
            where T : ICloneable
        {
            return (T)value.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerPlateOp"]/message_doc[@name="LogEvidenceRatio{T}(T, T, double, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        public static double LogEvidenceRatio<T>(T enter, T value, double exponent, [Fresh] T to_enter)
            where T : CanGetLogAverageOf<T>, CanGetLogAverageOfPower<T>
        {
            // qnot(x) =propto q(x)/m_out(x)
            // qnot2(x) =propto q(x)/m_out(x)^n
            // the interior of the plate sends (int_x qnot(x) f(x) dx)^n
            // which is missing (int_x qnot2(x) m_out(x)^n dx)/(int_x qnot(x) m_out(x) dx)^n
            // this factor sends the missing piece, where:
            // enter = m_out(x)
            // to_enter = qnot(x)
            // value = qnot2(x)
            return value.GetLogAverageOfPower(enter, exponent) - exponent * to_enter.GetLogAverageOf(enter);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerPlateOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerPlateOp"]/message_doc[@name="ValueAverageLogarithm{T}(T, double, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        public static T ValueAverageLogarithm<T>([SkipIfUniform] T enter, double exponent, T result)
            where T : SettableToPower<T>
        {
            return ValueAverageConditional<T>(enter, exponent, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerPlateOp"]/message_doc[@name="EnterAverageLogarithm{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        public static T EnterAverageLogarithm<T>([IsReturned] T value)
        {
            return value;
        }
    }

    /// <summary>
    /// Damp factor methods
    /// </summary>
    [Hidden]
    public static class Damp
    {
        /// <summary>
        /// Copy a value and damp the backward message.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="stepsize">1.0 means no damping, 0.0 is infinite damping.</param>
        /// <returns></returns>
        /// <remarks>
        /// If you use this factor, be sure to increase the number of algorithm iterations appropriately.
        /// The number of iterations should increase according to the reciprocal of stepsize.
        /// </remarks>
        public static T Backward<T>([IsReturned] T value, double stepsize)
        {
            return value;
        }

        /// <summary>
        /// Copy a value and damp the forward message.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="stepsize">1.0 means no damping, 0.0 is infinite damping.</param>
        /// <returns></returns>
        /// <remarks>
        /// If you use this factor, be sure to increase the number of algorithm iterations appropriately.
        /// The number of iterations should increase according to the reciprocal of stepsize.
        /// </remarks>
        public static T Forward<T>([IsReturned] T value, double stepsize)
        {
            return value;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DampBackwardOp"]/doc/*'/>
    [FactorMethod(typeof(Damp), "Backward<>")]
    [Quality(QualityBand.Preview)]
    public static class DampBackwardOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DampBackwardOp"]/message_doc[@name="LogEvidenceRatio()"]/*'/>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        // /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DampBackwardOp"]/message_doc[@name="ValueAverageConditional{Distribution}(Distribution, double, Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution ValueAverageConditional<Distribution>(
            [SkipIfUniform] Distribution backward, double stepsize, Distribution to_value)
            where Distribution : SettableToPower<Distribution>, SettableToProduct<Distribution>
        {
            // damp the backward message.
            // to_value holds the last message to value.
            // result = to_value^(1-stepsize) * backward^stepsize
            Distribution result = to_value;
            result.SetToPower(to_value, (1 - stepsize) / stepsize);
            result.SetToProduct(result, backward);
            result.SetToPower(result, stepsize);
            return result;
        }

        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution ValueAverageLogarithm<Distribution>(
            [SkipIfUniform] Distribution backward, double stepsize, Distribution to_value)
            where Distribution : SettableToPower<Distribution>, SettableToProduct<Distribution>
        {
            return ValueAverageConditional(backward, stepsize, to_value);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DampBackwardOp"]/message_doc[@name="BackwardAverageConditional{Distribution}(Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution BackwardAverageConditional<Distribution>([IsReturned] Distribution value)
        {
            return value;
        }

        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution BackwardAverageLogarithm<Distribution>([IsReturned] Distribution value)
        {
            return value;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DampForwardOp"]/doc/*'/>
    [FactorMethod(typeof(Damp), "Forward<>")]
    [Quality(QualityBand.Preview)]
    public static class DampForwardOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DampForwardOp"]/message_doc[@name="LogEvidenceRatio()"]/*'/>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        // /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DampForwardOp"]/message_doc[@name="ForwardAverageConditional{Distribution}(Distribution, double, Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution ForwardAverageConditional<Distribution>(
            [SkipIfUniform] Distribution value, double stepsize, Distribution to_forward)
            where Distribution : SettableToPower<Distribution>, SettableToProduct<Distribution>, HasPoint<T>, SettableToWeightedSum<Distribution>, SettableToUniform
        {
            // damp the backward message.
            // to_forward holds the last message to value.
            // result = to_forward^(1-stepsize) * backward^stepsize
            //        = (to_forward^(1-stepsize)/stepsize * backward)^stepsize
            Distribution result = to_forward;
            if ((value.IsPointMass && !to_forward.IsUniform()) || to_forward.IsPointMass)
            {
                result.SetToSum(1 - stepsize, to_forward, stepsize, value);
            }
            else
            {
                result.SetToPower(to_forward, (1 - stepsize) / stepsize);
                result.SetToProduct(result, value);
                result.SetToPower(result, stepsize);
            }
            return result;
        }

        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution ForwardAverageLogarithm<Distribution>(
            [SkipIfUniform] Distribution value, double stepsize, Distribution to_forward)
            where Distribution : SettableToPower<Distribution>, SettableToProduct<Distribution>, HasPoint<T>, SettableToWeightedSum<Distribution>, SettableToUniform
        {
            return ForwardAverageConditional(value, stepsize, to_forward);
        }

        // /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DampForwardOp"]/message_doc[@name="ValueAverageConditional{Distribution}(Distribution, Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution ValueAverageConditional<Distribution>([IsReturned] Distribution forward)
        {
            return forward;
        }

        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution ValueAverageLogarithm<Distribution>([IsReturned] Distribution forward)
        {
            return forward;
        }
    }
}

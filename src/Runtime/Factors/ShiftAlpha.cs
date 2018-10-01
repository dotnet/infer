// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <summary>
    /// Factors that change a message channel's alpha factor
    /// </summary>
    [Hidden]
    public static class ShiftAlpha
    {
        /// <summary>
        /// Changes a message channel's alpha value, going to a factor.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        [ParameterNames("factor", "variable", "variableAlpha", "factorAlpha")]
        public static T ToFactor<T>(T variable, double variableAlpha, double factorAlpha)
        {
            return variable;
        }

        /// <summary>
        /// Changes a message channel's alpha value, coming from a factor.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        [ParameterNames("variable", "factor", "factorAlpha", "variableAlpha")]
        public static T FromFactor<T>(T factor, double factorAlpha, double variableAlpha)
        {
            return factor;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ShiftAlphaToFactorOp"]/doc/*'/>
    [FactorMethod(typeof(ShiftAlpha), "ToFactor<>")]
    [Quality(QualityBand.Experimental)]
    public static class ShiftAlphaToFactorOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ShiftAlphaToFactorOp"]/message_doc[@name="FactorAverageConditional{T}(T, T, double, double, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T FactorAverageConditional<T>(T factor, [SkipIfUniform] T variable, double variableAlpha, double factorAlpha, T result)
            where T : SettableToPower<T>, SettableToProduct<T>
        {
            result.SetToPower(factor, variableAlpha - factorAlpha);
            result.SetToProduct(result, variable);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ShiftAlphaToFactorOp"]/message_doc[@name="VariableAverageConditional{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T VariableAverageConditional<T>([SkipIfUniform] T factor, T result)
            where T : SettableTo<T>
        {
            result.SetTo(factor);
            return result;
        }

        /// <summary>
        /// Evidence message for EP.
        /// </summary>
        /// <param name="factor">Incoming message from 'factor'.</param>
        /// <param name="variable">Incoming message from 'variable'.</param>
        /// <param name="variableAlpha">Constant value for 'variableAlpha'.</param>
        /// <param name="factorAlpha">Constant value for 'factorAlpha'.</param>
        /// <returns><c>log(int f(x) qnotf(x) dx / int ftilde(x) qnotf(x) dx)</c></returns>
        /// <remarks><para>
        /// The formula for the result is <c>log(int f(x) qnotf(x) dx / int ftilde(x) qnotf(x) dx)</c>
        /// where <c>x = (factor,variable,variableAlpha,factorAlpha)</c>.
        /// </para></remarks>
        public static double LogEvidenceRatioOld<T>(T factor, T variable, double variableAlpha, double factorAlpha)
            where T : ICloneable, CanGetAverageLog<T>, SettableToPower<T>, SettableToProduct<T>
        {
            if (variableAlpha == 1 && factorAlpha == 0)
            {
                // EP variable to VMP factor
                T to_factor = FactorAverageConditional(factor, variable, variableAlpha, factorAlpha, (T)variable.Clone());
                return -to_factor.GetAverageLog(factor);
            }
            else if (variableAlpha == 0 && factorAlpha == 1)
            {
                // VMP variable to EP factor
                return variable.GetAverageLog(factor);
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ShiftAlphaFromFactorOp"]/doc/*'/>
    [FactorMethod(typeof(ShiftAlpha), "FromFactor<>")]
    [Quality(QualityBand.Experimental)]
    public static class ShiftAlphaFromFactorOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ShiftAlphaFromFactorOp"]/message_doc[@name="FactorAverageConditional{T}(T, T, double, double, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T FactorAverageConditional<T>(T factor, [SkipIfUniform] T variable, double factorAlpha, double variableAlpha, T result)
            where T : SettableToPower<T>, SettableToProduct<T>
        {
            result.SetToPower(factor, variableAlpha - factorAlpha);
            result.SetToProduct(result, variable);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ShiftAlphaFromFactorOp"]/message_doc[@name="VariableAverageConditional{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T VariableAverageConditional<T>([SkipIfUniform] T factor, T result)
            where T : SettableTo<T>
        {
            result.SetTo(factor);
            return result;
        }

        /// <summary>
        /// Evidence message for EP.
        /// </summary>
        /// <param name="factor">Incoming message from 'factor'.</param>
        /// <param name="variable">Incoming message from 'variable'.</param>
        /// <param name="variableAlpha">Constant value for 'variableAlpha'.</param>
        /// <param name="factorAlpha">Constant value for 'factorAlpha'.</param>
        /// <returns><c>log(int f(x) qnotf(x) dx / int ftilde(x) qnotf(x) dx)</c></returns>
        /// <remarks><para>
        /// The formula for the result is <c>log(int f(x) qnotf(x) dx / int ftilde(x) qnotf(x) dx)</c>
        /// where <c>x = (variable,factor,factorAlpha,variableAlpha)</c>.
        /// </para></remarks>
        public static double LogEvidenceRatioOld<T>(T factor, T variable, double variableAlpha, double factorAlpha)
            where T : ICloneable, CanGetAverageLog<T>, SettableToPower<T>, SettableToProduct<T>
        {
            if (variableAlpha == 1 && factorAlpha == 0)
            {
                // EP variable to VMP factor
                T to_factor = FactorAverageConditional(factor, variable, variableAlpha, factorAlpha, (T)variable.Clone());
                return -to_factor.GetAverageLog(factor);
            }
            else if (variableAlpha == 0 && factorAlpha == 1)
            {
                // VMP variable to EP factor
                return variable.GetAverageLog(factor);
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}

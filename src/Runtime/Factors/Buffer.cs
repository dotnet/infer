// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    internal static class BufferTester
    {
        [Hidden]
        public static T Copy<T>(T value)
        {
            return value;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BufferTesterCopyOp"]/doc/*'/>
    [FactorMethod(typeof(BufferTester), "Copy<>")]
    [Buffers("buffer")]
    [Quality(QualityBand.Experimental)]
    public static class BufferTesterCopyOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BufferTesterCopyOp"]/message_doc[@name="Buffer{T}(T, T, T)"]/*'/>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T Buffer<T>(T copy, T value, T result)
        {
            return value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BufferTesterCopyOp"]/message_doc[@name="BufferInit{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T BufferInit<T>(T value)
        {
            return value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BufferTesterCopyOp"]/message_doc[@name="CopyAverageConditional{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T CopyAverageConditional<T>(T value, T buffer)
        {
            return value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BufferTesterCopyOp"]/message_doc[@name="ValueAverageConditional{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageConditional<T>(T copy)
        {
            return copy;
        }
    }

    /// <summary>
    /// Buffer factors
    /// </summary>
    internal static class Buffer
    {
        /// <summary>
        /// Value factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        [Hidden]
        public static T Value<T>()
        {
            return default(T);
        }

        /// <summary>
        /// Infer factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        [Hidden]
        public static void Infer<T>(T value)
        {
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BufferOp"]/doc/*'/>
    [FactorMethod(typeof(Buffer), "Value<>")]
    [Quality(QualityBand.Mature)]
    internal static class BufferOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BufferOp"]/message_doc[@name="ValueAverageConditional{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageConditional<T>([SkipIfUniform] T value)
        {
            return value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BufferOp"]/message_doc[@name="ValueAverageLogarithm{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageLogarithm<T>([SkipIfUniform] T value)
        {
            return value;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InferOp"]/doc/*'/>
    [FactorMethod(typeof(Buffer), "Infer<>")]
    [Quality(QualityBand.Experimental)]
    internal static class InferOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InferOp"]/message_doc[@name="ValueAverageConditional{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageConditional<T>([SkipIfUniform] T value, T result)
            where T : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="InferOp"]/message_doc[@name="ValueAverageLogarithm{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageLogarithm<T>([SkipIfUniform] T value, T result)
            where T : SettableToUniform
        {
            return ValueAverageConditional(value, result);
        }
    }
}

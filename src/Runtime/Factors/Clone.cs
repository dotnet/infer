// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Attributes;
    using Utilities;

    /// <summary>
    /// Contains factor methods for cloning.
    /// </summary>
    [Quality(QualityBand.Stable)]
    public static class Clone
    {
        /// <summary>
        /// An internal factor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="def"></param>
        /// <param name="marginal"></param>
        /// <returns></returns>
        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "marginal")]
        public static T Variable<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "init", "marginal")]
        public static T VariableInit<T>(T def, T init, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "marginal")]
        public static T VariableGibbs<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "marginal")]
        public static T VariableMax<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "marginal")]
        public static T VariablePoint<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// An internal factor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="def"></param>
        /// <param name="marginal"></param>
        /// <returns></returns>
        [ParameterNames("use", "def", "marginal")]
        [Hidden]
        public static T DerivedVariable<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "init", "marginal")]
        [Hidden]
        public static T DerivedVariableInit<T>(T def, T init, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "marginal")]
        [Hidden]
        public static T DerivedVariableVmp<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "init", "marginal")]
        [Hidden]
        public static T DerivedVariableInitVmp<T>(T def, T init, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "marginal")]
        [Hidden]
        public static T DerivedVariableGibbs<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "init", "marginal")]
        [Hidden]
        public static T DerivedVariableInitGibbs<T>(T def, T init, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Create an array filled with a single value.
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="Def">The value to fill with.</param>
        /// <param name="count"></param>
        /// <param name="Marginal">Dummy argument for inferring marginals.</param>
        /// <returns>A new array with all entries set to value.</returns>
        [Stochastic]
        [ParameterNames("Uses", "Def", "count", "Marginal")]
        [Hidden]
        public static T[] UsesEqualDef<T>(T Def, int count, out T Marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Stochastic]
        [ParameterNames("Uses", "Def", "count", "burnIn", "thin", "marginal", "samples", "conditionals")]
        [Hidden]
        public static T[] UsesEqualDefGibbs<T>(T Def, int count, int burnIn, int thin, out T marginal, out T samples, out T conditionals)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Create an array filled with a single value. For reference types,
        /// the replicates all reference the same instance
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="value">The value to fill with.</param>
        /// <param name="count">Number of replicates</param>
        /// <returns>A new array with all entries set to value.</returns>
        [Hidden]
        [ParameterNames("Uses", "Def", "Count")]
        [ReturnsCompositeArray]
        [HasUnitDerivative]
        public static T[] Replicate<T>([IsReturnedInEveryElement] T value, [Constant] int count)
        {
            T[] result = new T[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        /// Create a multidimensional array filled with a single value.
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="value">The value to fill with.</param>
        /// <returns>A new array with all entries set to value.</returns>
        [Hidden]
        [ParameterNames("Uses", "Def")]
        public static Array ReplicateNd<T>(T value)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Create an array filled with a single value.
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="Def">The value to fill with.</param>
        /// <param name="count"></param>
        /// <param name="Marginal">Dummy argument for inferring marginals.</param>
        /// <returns>A new array with all entries set to value.</returns>
        [ParameterNames("Uses", "Def", "count", "Marginal")]
        [Hidden]
        public static T[] ReplicateWithMarginal<T>(T Def, int count, out T Marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("Uses", "Def", "count", "burnIn", "thin", "marginal", "samples", "conditionals")]
        [Hidden]
        public static T[] ReplicateWithMarginalGibbs<T>(T Def, int count, int burnIn, int thin, out T marginal, out T samples, out T conditionals)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Passes the input through to the output.
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="value">The value to return.</param>
        /// <returns>The supplied value.</returns>
        [Hidden]
        [Fresh] // needed for CopyPropagationTransform
        [HasUnitDerivative]
        public static T Copy<T>([SkipIfUniform] T value)
        {
            return value;
        }

    }
}

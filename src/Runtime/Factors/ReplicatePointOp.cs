// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define SpecializeArrays
#define MinimalGenericTypeParameters

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    [FactorMethod(typeof(Clone), "Replicate<>", Default = false)]
    [Buffers("toDef")]
    [Quality(QualityBand.Preview)]
    public static class ReplicatePointOp
    {
        [Skip]
        public static double LogEvidenceRatio<T>(IList<T> uses)
        {
            return 0.0;
        }

        /// <typeparam name="T">The type of the outgoing message.</typeparam>
        /// <typeparam name="TDef">The type of the incoming message from <c>Def</c>.</typeparam>
        public static T UsesAverageConditional<T, TDef>([IsReturned] TDef Def, int resultIndex, T result)
            where T : SettableTo<TDef>
        {
            // Def will always be a point mass, so no division is needed.
            result.SetTo(Def);
            return result;
        }

        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefAverageConditional<T>([IsReturned] T toDef, T result)
            where T : SettableTo<T>
        {
            result.SetTo(toDef);
            return result;
        }

        /// <typeparam name="T">The type of the messages.</typeparam>
        // SkipIfUniform on 'use' causes this line to be pruned when the backward message isn't changing
        [Fresh]
        public static T ToDefIncrement<T>(T toDef, [SkipIfUniform] T use)
            where T : SettableToProduct<T>
        {
            toDef.SetToProduct(toDef, use);
            return toDef;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_Divide"]/message_doc[@name="ToDefInit{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip] // this is needed to instruct the scheduler to treat the buffer as uninitialized
        public static T ToDefInit<T>([IgnoreDependency] T Def) // IgnoreDependency permits more optimizations in LocalAllocationTransform
            where T : ICloneable, SettableToUniform
        {
            // must construct from Def instead of Uses because Uses array may be empty
            return ArrayHelper.MakeUniform(Def);
        }

        /// <typeparam name="T">The type of the messages.</typeparam>       
        [MultiplyAll]
        [Fresh]
        public static T ToDef<T>([SkipIfAllUniform] IReadOnlyList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }

#if SpecializeArrays
        /// <typeparam name="T">The type of the messages.</typeparam>       
        [MultiplyAll]
        [Fresh]
        public static T ToDef<T>([SkipIfAllUniform] T[] Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#endif
    }
}

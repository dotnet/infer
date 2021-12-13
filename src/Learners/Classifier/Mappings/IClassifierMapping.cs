// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using Microsoft.ML.Probabilistic.Utilities;
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// A mapping used by classifier implementations of the
    /// <see cref="IPredictor{TInstanceSource, TInstance, TLabelSource, TResult, TResultDist}"/>
    /// interface to convert user data to an appropriate format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TFeatures">The type of the feature values.</typeparam>
    public interface IClassifierMapping<in TInstanceSource, TInstance, in TLabelSource, TLabel, out TFeatures> 
        : IPredictorMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>
    {
        /// <summary>
        /// Gets all class labels.
        /// </summary>
        /// <param name="instanceSource">An optional instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <returns>All possible values of a label.</returns>
        IEnumerable<TLabel> GetClassLabels(TInstanceSource instanceSource = default(TInstanceSource), TLabelSource labelSource = default(TLabelSource));

        /// <summary>
        /// Serializes a label to a string.
        /// </summary>
        /// <param name="label"></param>
        /// <returns></returns>
        string LabelToString(TLabel label);

        /// <summary>
        /// Deserializes a label from a string.  Reverses <see cref="LabelToString(TLabel)"/>.
        /// </summary>
        /// <param name="labelString">The output of <see cref="LabelToString(TLabel)"/></param>
        /// <returns>The label</returns>
        TLabel ParseLabel(string labelString);
    }

    /// <summary>
    /// Provides a default implementation of <see cref="IClassifierMapping{TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures}"/>
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TFeatures">The type of the feature values.</typeparam>
    [Serializable]
    public abstract class ClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures> :
        IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, TFeatures>
    {
        /// <inheritdoc/>
        public abstract TFeatures GetFeatures(TInstance instance, TInstanceSource instanceSource = default);

        /// <inheritdoc/>
        public abstract IEnumerable<TInstance> GetInstances(TInstanceSource instanceSource);

        /// <inheritdoc/>
        public abstract TLabel GetLabel(TInstance instance, TInstanceSource instanceSource = default, TLabelSource labelSource = default);

        /// <inheritdoc/>
        public abstract IEnumerable<TLabel> GetClassLabels(TInstanceSource instanceSource = default(TInstanceSource), TLabelSource labelSource = default(TLabelSource));

        /// <inheritdoc/>
        public virtual string LabelToString(TLabel label)
        {
            return label.ToString();
        }

        /// <inheritdoc/>
        public virtual TLabel ParseLabel(string labelString)
        {
            if (labelString is TLabel value)
            {
                return value;
            }
            else if (0 is TLabel)
            {
                return (TLabel)(object)int.Parse(labelString);
            }
            else
            {
                throw new NotImplementedException($"Cannot parse object type {StringUtil.TypeToString(typeof(TLabel))}.  To add this functionality, provide a ParseLabel implementation in your mapping class.");
            }
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Contains all the data mappings used in evaluation.
    /// </summary>
    public static class ClassifierMappings
    {
        /// <summary>
        /// Initializes static members of the <see cref="Mappings"/> class.
        /// </summary>
        static ClassifierMappings()
        {
            Classifier = new ClassifierMapping();
        }

        /// <summary>
        /// Gets the mapping from dataset format to the common classifier format.
        /// </summary>
        public static IClassifierMapping<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string, Vector> Classifier
        {
            get;
            private set;
        }
    }
}

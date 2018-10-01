// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Xml.Serialization;

    /// <summary>
    /// The configuration element that describes the rating prediction test.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "PredictRatings")]
    public class RatingPredictionTestConfigurationElement : RecommenderTestConfigurationElement
    {
        /// <summary>
        /// Gets or sets a value indicating whether the uncertain prediction metrics should also be computed.
        /// </summary>
        public bool? ComputeUncertainPredictionMetrics { get; set; }
        
        /// <summary>
        /// Creates an instance of <see cref="RatingPredictionTest"/> using the settings from this configuration element.
        /// </summary>
        /// <returns>An instance of <see cref="RatingPredictionTest"/>.</returns>
        public override RecommenderTest Create()
        {
            return new RatingPredictionTest(this.ComputeUncertainPredictionMetrics.Value);
        }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Linq;
    using System.Xml.Serialization;

    /// <summary>
    /// The configuration element that describes single run of a particular test with a particular recommender.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "Run")]
    public class RecommenderRunConfigurationElement : ConfigurationElement
    {
        /// <summary>
        /// Gets or sets the name of the run.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the dataset used for the run.
        /// </summary>
        public RecommenderDatasetConfigurationElement Dataset { get; set; }

        /// <summary>
        /// Gets or sets the splitting settings for the dataset.
        /// </summary>
        public RecommenderDatasetSplitSettingsConfigurationElement SplitSettings { get; set; }
        
        /// <summary>
        /// Gets or sets the recommender used for the run.
        /// </summary>
        public RecommenderConfigurationElement Recommender { get; set; }

        /// <summary>
        /// Gets or sets the test used for the run.
        /// </summary>
        public RecommenderTestConfigurationElement[] Tests { get; set; }

        /// <summary>
        /// Creates an instance of <see cref="RecommenderRun"/> using the settings from this configuration element.
        /// </summary>
        /// <returns>The created instance.</returns>
        public RecommenderRun Create()
        {
            return new RecommenderRun(
                this.Name,
                this.Dataset.Load(),
                this.SplitSettings.FoldCount.Value,
                () => this.SplitSettings.CreateSplittingMapping(),
                mapping => this.Recommender.Create(mapping),
                this.Tests.Select(t => t.Create()));
        }
    }
}
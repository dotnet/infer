// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Xml.Serialization;

    /// <summary>
    /// Represents the configuration of the evaluation tool.
    /// </summary>
    [XmlType(TypeName = "Configuration")]
    public class EvaluatorConfiguration : ConfigurationBase<EvaluatorConfiguration>
    {
        /// <summary>
        /// Gets or sets the list of test runs which will be performed by the evaluation tool.
        /// </summary>
        public RecommenderRunConfigurationElement[] Runs { get; set; }

        /// <summary>
        /// Checks if the configuration is correct.
        /// </summary>
        protected override void Check()
        {
            foreach (var runConfiguration in this.Runs)
            {
                // Make sure all the settings are presented for the elements of '<Runs>' section
                runConfiguration.Traverse(element => element.CheckIfAllPropertiesSet());

                // Make sure each test is specified only once in each run
                IEnumerable<Type> distinctTestTypes = runConfiguration.Tests.Select(t => t.GetType()).Distinct();
                if (distinctTestTypes.Count() != runConfiguration.Tests.Length)
                {
                    throw new InvalidConfigurationException(
                        string.Format("Two or more tests of the same type are specified for the recommender run '{0}'.", runConfiguration.Name));
                }
            }
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Xml;
    using System.Xml.Serialization;

    /// <summary>
    /// The base class for the configuration files.
    /// </summary>
    /// <typeparam name="TDerived">The type of the concrete configuration class.</typeparam>
    [Serializable]
    public abstract class ConfigurationBase<TDerived> : ConfigurationElement where TDerived : ConfigurationBase<TDerived>
    {
        /// <summary>
        /// Gets or sets the list of configuration elements that can be referenced by the runs.
        /// </summary>
        public ConfigurationElement[] Dictionary { get; set; }

        /// <summary>
        /// Loads configuration from a file.
        /// </summary>
        /// <param name="fileName">The name of the file.</param>
        /// <returns>The loaded configuration.</returns>
        /// <exception cref="InvalidConfigurationException">Thrown if the configuration file is not valid.</exception>
        public static TDerived LoadFromFile(string fileName)
        {
            var serializer = new XmlSerializer(typeof(TDerived));
            serializer.UnknownNode += SerializerOnUnknownNode;

            using (XmlReader reader = new XmlTextReader(fileName))
            {
                var configuration = (TDerived)serializer.Deserialize(reader);

                // Build id -> configuration element mapping
                var idToElement = new Dictionary<string, ConfigurationElement>();
                configuration.Traverse(
                    element =>
                    {
                        if (string.IsNullOrEmpty(element.Id))
                        {
                            return; // Skip nameless elements
                        }

                        ConfigurationElement elementWithId;
                        if (idToElement.TryGetValue(element.Id, out elementWithId))
                        {
                            if (elementWithId != element)
                            {
                                throw new InvalidConfigurationException(
                                    string.Format("Identifier '{0}' was used for more than one object in the configuration file", element.Id));
                            }
                        }
                        else
                        {
                            idToElement.Add(element.Id, element);
                        }
                    });

                // Propagate settings from parent to child
                configuration.Traverse(element => element.PropagateParentSettings(idToElement));

                // Perform custom correctness check
                configuration.Check();

                return configuration;
            }
        }

        /// <summary>
        /// Can be overridden in the derived classes to perform a custom check of the configuration after it was loaded.
        /// </summary>
        protected virtual void Check()
        {
        }

        /// <summary>
        /// Handles unknown nodes in the configuration file.
        /// </summary>
        /// <param name="sender">The sender of the event.</param>
        /// <param name="e">The arguments of the event.</param>
        /// <exception cref="InvalidConfigurationException">Thrown if an unknown node was detected in the configuration file.</exception>
        private static void SerializerOnUnknownNode(object sender, XmlNodeEventArgs e)
        {
            if (e.LocalName == "type" && e.NodeType == XmlNodeType.Attribute)
            {
                return; // For some reason type definition is considered unknown
            }

            throw new InvalidConfigurationException(string.Format("Unknown element '{0}' found in the configuration file.", e.Name));
        }
    }
}

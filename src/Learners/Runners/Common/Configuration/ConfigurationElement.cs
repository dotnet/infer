// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Reflection;
    using System.Xml.Serialization;

    /// <summary>
    /// The base class for all the classes representing elements of the configuration file.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "Element")]
    public abstract class ConfigurationElement
    {
        /// <summary>
        /// Gets or sets the identifier of the element.
        /// </summary>
        /// <remarks>The value of the property can be null which means that the element is nameless.</remarks>
        [XmlAttribute]
        public string Id { get; set; }

        /// <summary>
        /// Gets or sets the identifier of the parent element.
        /// </summary>
        /// <remarks>The value of the property can be null which indicates that the element has no parent.</remarks>
        [XmlAttribute]
        public string Parent { get; set; }

        /// <summary>
        /// Checks if all configurable properties of the element have defined values.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown if some configurable property has no value.</exception>
        public void CheckIfAllPropertiesSet()
        {
            foreach (PropertyInfo property in this.EnumerateConfigurableProperties())
            {
                if (property.GetValue(this, null) == null)
                {
                    throw new InvalidOperationException(
                        string.Format("Value of required configuration property {0} wasn't set for configuration element '{1}'.", property.Name, this.Id));
                }
            }
        }

        /// <summary>
        /// Propagates values of configurable properties which are not defined in this element from parent.
        /// </summary>
        /// <param name="idToElement">A mapping from element identifier to an object which represents the element.</param>
        public void PropagateParentSettings(IDictionary<string, ConfigurationElement> idToElement)
        {
            this.PropagateParentSettings(idToElement, new HashSet<ConfigurationElement>());
        }

        /// <summary>
        /// Performs the given action for this configuration element and all its children.
        /// </summary>
        /// <param name="action">The action to perform.</param>
        public void Traverse(Action<ConfigurationElement> action)
        {
            this.Traverse(action, new HashSet<ConfigurationElement>());
        }

        /// <summary>
        /// Checks if null is in the domain of a given type.
        /// </summary>
        /// <param name="type">The type to check.</param>
        /// <returns>True if null is in the domain of <paramref name="type"/>, false otherwise. </returns>
        private static bool IsNullInTypeDomain(Type type)
        {
            return !type.IsValueType || Nullable.GetUnderlyingType(type) != null;
        }

        /// <summary>
        /// Performs the given action for this configuration element and all its children.
        /// </summary>
        /// <param name="action">The action to perform.</param>
        /// <param name="visitedElements">The list of elements which have been traversed already.</param>
        private void Traverse(Action<ConfigurationElement> action, HashSet<ConfigurationElement> visitedElements)
        {
            visitedElements.Add(this);
            action(this);
            foreach (ConfigurationElement element in this.EnumerateChildConfigurationElements())
            {
                if (element == null)
                {
                    continue;
                }
                
                if (!visitedElements.Contains(element))
                {
                    element.Traverse(action, visitedElements);
                }
            }
        }

        /// <summary>
        /// Enumerates all the properties of this configuration element which can be defined in the configuration file.
        /// </summary>
        /// <returns>The enumeration of configurable properties.</returns>
        /// <exception cref="NotSupportedException">Thrown if the element declares a property of type which doesn't have null in its domain.</exception>
        private IEnumerable<PropertyInfo> EnumerateConfigurableProperties()
        {
            PropertyInfo[] properties = this.GetType().GetProperties();
            foreach (PropertyInfo property in properties)
            {
                if (property.GetCustomAttributes(typeof(NonSerializedAttribute), false).Length > 0)
                {
                    continue; // Skip properties which aren't serialized
                }

                if (property.GetCustomAttributes(typeof(XmlAttributeAttribute), false).Length > 0)
                {
                    continue; // Skip properties which are serialized as attributes (like Id and Parent)
                }

                if (!IsNullInTypeDomain(property.PropertyType))
                {
                    throw new NotSupportedException("Only nullable and reference types are supported in the configuration.");
                }

                yield return property;
            }
        }

        /// <summary>
        /// Enumerates all the children of this configuration element.
        /// </summary>
        /// <returns>The enumeration of child elements.</returns>
        private IEnumerable<ConfigurationElement> EnumerateChildConfigurationElements()
        {
            foreach (PropertyInfo property in this.EnumerateConfigurableProperties())
            {
                // Nested config element, return as is
                if (typeof(ConfigurationElement).IsAssignableFrom(property.PropertyType))
                {
                    yield return (ConfigurationElement)property.GetValue(this, null);
                }

                // Nested array of config elements, return each one separately
                if (property.PropertyType.IsArray && typeof(ConfigurationElement).IsAssignableFrom(property.PropertyType.GetElementType()))
                {
                    var propertyValue = (Array)property.GetValue(this, null);
                    foreach (ConfigurationElement childElement in propertyValue)
                    {
                        yield return childElement;
                    }
                }
            }
        }

        /// <summary>
        /// Propagates values of configurable properties which are not defined in this element from parent.
        /// </summary>
        /// <param name="idToElement">A mapping from element identifier to an object which represents the element.</param>
        /// <param name="visitedElements">The list of elements which have been traversed already.</param>
        private void PropagateParentSettings(IDictionary<string, ConfigurationElement> idToElement, HashSet<ConfigurationElement> visitedElements)
        {
            visitedElements.Add(this);

            if (this.Parent == null)
            {
                return; // No parent to propagate settings from
            }

            ConfigurationElement parentElement;
            if (!idToElement.TryGetValue(this.Parent, out parentElement))
            {
                throw new InvalidOperationException(string.Format("Can't find configuration element with id '{0}'", this.Parent));
            }

            if (visitedElements.Contains(parentElement))
            {
                throw new InvalidConfigurationException(string.Format("Circular dependency containing configuration element '{0}' detected.", this.Parent));
            }

            if (parentElement.GetType() != this.GetType())
            {
                throw new InvalidOperationException(string.Format("Parent configuration element '{0}' is not of the same type as child '{1}'.", this.Parent, this.Id));
            }

            parentElement.PropagateParentSettings(idToElement, visitedElements);
            this.CopyPropertyValues(parentElement);
        }

        /// <summary>
        /// Copies values of undefined configurable properties from the given element.
        /// </summary>
        /// <param name="element">The element to copy property values from.</param>
        private void CopyPropertyValues(ConfigurationElement element)
        {
            Debug.Assert(element != this, "Element should be specified.");
            Debug.Assert(element.GetType() == this.GetType(), "Parent and child should be of the same type.");
            
            foreach (PropertyInfo property in this.EnumerateConfigurableProperties())
            {
                object childValue = property.GetValue(this, null);
                if (childValue != null)
                {
                    continue;  // Do not override values that are already presented in child
                }

                object parentValue = property.GetValue(element, null);
                property.SetValue(this, parentValue, null);
            }
        }
    }
}
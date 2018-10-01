// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Serialization
{
    using System.Xml;

    /// <summary>
    /// Provides extension methods for <see cref="XmlWriter"/>.
    /// </summary>
    public static class XmlWriterExtensions
    {
        /// <summary>
        /// Writes a given text specified via a format string.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="format">The format string.</param>
        /// <param name="args">The format string arguments.</param>
        public static void WriteFormatString(this XmlWriter writer, string format, params object[] args)
        {
            writer.WriteString(string.Format(format, args));
        }

        /// <summary>
        /// Writes an element with the specified name and value. The value is specified via a format string.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="elementName">The name of the element.</param>
        /// <param name="format">The value format string.</param>
        /// <param name="args">The value format string arguments.</param>
        public static void WriteElementFormatString(this XmlWriter writer, string elementName, string format, params object[] args)
        {
            writer.WriteStartElement(elementName);
            writer.WriteFormatString(format, args);
            writer.WriteEndElement();
        }

        /// <summary>
        /// Writes an element with the specified name and a single attribute with the specified name and value.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="elementName">The name of the element.</param>
        /// <param name="attributeName">The name of the attribute.</param>
        /// <param name="attributeValue">The value of the attribute.</param>
        public static void WriteElementAttributeString(this XmlWriter writer, string elementName, string attributeName, string attributeValue)
        {
            writer.WriteStartElement(elementName);
            writer.WriteAttributeString(attributeName, attributeValue);
            writer.WriteEndElement();
        }
    }
}

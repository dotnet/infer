// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    /// <summary>
    /// Attribute used to label classes containing code examples. The examples browser
    /// will use this information to organise the code examples
    /// </summary>
    public class ExampleAttribute : Attribute
    {
        private string category;
        private string description;
        private string prefix;

        /// <summary>
        /// Creates a new Example attribute with a specified category and description
        /// </summary>
        /// <param name="category">Category of the example</param>
        /// <param name="description">Description of the example</param>
        public ExampleAttribute(string category, string description)
        {
            this.category = category;
            this.description = description;
        }

        /// <summary>
        /// Category of the example
        /// </summary>
        public string Category
        {
            get { return category; }
        }

        /// <summary>
        /// Description of the example
        /// </summary>
        public string Description
        {
            get { return description; }
        }

        /// <summary>
        /// Prefix
        /// </summary>
        public string Prefix
        {
            get { return prefix; }
            set { prefix = value; }
        }
    }
}

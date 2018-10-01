// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Globalization;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Models.Attributes
{
    /// <summary>
    /// A group of variables processed together by an inference algorithm
    /// </summary>
    public class VariableGroup
    {
        private List<Variable> variables = new List<Variable>();

        /// <summary>
        /// List of variables in the group
        /// </summary>
        public IList<Variable> Variables
        {
            get { return variables.AsReadOnly(); }
        }

        /// <summary>
        /// Name of the group
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Default constructor for when explicitly attaching GroupMember attributes
        /// </summary>
        public VariableGroup()
        {
        }

        /// <summary>
        /// Static constructor
        /// </summary>
        /// <param name="vars">List of variables</param>
        /// <returns>The variable group</returns>
        public static VariableGroup FromVariables(params Variable[] vars)
        {
            VariableGroup vg = new VariableGroup();
            foreach (Variable var in vars)
            {
                vg.variables.Add(var);
            }
            return vg;
        }


        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="that">The variable group we are copying from</param>
        public VariableGroup(VariableGroup that) : this()
        {
            foreach (Variable var in that.variables)
                variables.Add(var);
        }

        /// <summary>
        /// Returns a string representation of this variable group.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            if (variables.Count == 0)
            {
                if (Name == null || Name.Length == 0)
                    return GetHashCode().ToString(CultureInfo.InvariantCulture);
                else
                    return Name;
            }

            else
                return StringUtil.CollectionToString(variables, ",");
        }
    }
}
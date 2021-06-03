// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    /// <summary>
    /// Represents a transducer defined on pairs of strings.
    /// </summary>
    public class StringTransducer :
        Transducer<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton, StringTransducer>
    {
    }
}
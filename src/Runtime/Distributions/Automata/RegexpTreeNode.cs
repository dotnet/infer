// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a node in the tree describing a regular expression.
    /// </summary>
    /// <typeparam name="TElement">The type of a character in a language word.</typeparam>
    /// <typeparam name="TElementSet">The type of a sequence element set.</typeparam>
    /// <remarks>
    /// Although this class might look immutable, <see cref="Simplify"/>
    /// can change the tree rooted at the node by removing existing children and/or adding new.
    /// </remarks>
    public class RegexpTreeNode<TElement, TElementSet>
            where TElementSet : CanCreatePartialUniform<TElementSet>, IImmutableDistribution<TElement, TElementSet>, new()
    {
        /// <summary>
        /// The most verbose formatting settings. Used for strong language equivalence check.
        /// </summary>
        private static readonly RegexpFormattingSettings VerboseFormattingSettings = new RegexpFormattingSettings(
            putOptionalInSquareBrackets: false,
            showAnyElementAsQuestionMark: false, 
            ignoreElementDistributionDetails: false,
            truncationLength: 500,
            escapeCharacters: false,
            useLazyQuantifier: true);

        /// <summary>
        /// A node instance representing an empty string language. Shared by all regular expression trees.
        /// </summary>
        private static readonly RegexpTreeNode<TElement, TElementSet> CachedEmptyNode = new RegexpTreeNode<TElement, TElementSet> { Type = RegexpTreeNodeType.Empty };

        /// <summary>
        /// A node instance representing an empty language. Shared by all regular expression trees.
        /// </summary>
        private static readonly RegexpTreeNode<TElement, TElementSet> CachedNothingNode = new RegexpTreeNode<TElement, TElementSet> { Type = RegexpTreeNodeType.Nothing };

        /// <summary>
        /// The children of the node.
        /// </summary>
        private List<RegexpTreeNode<TElement, TElementSet>> children = new List<RegexpTreeNode<TElement, TElementSet>>();

        /// <summary>
        /// The character set associated with the node (if it is an element set node).
        /// </summary>
        private IImmutableDistribution<TElement, TElementSet> elementSet;

        /// <summary>
        /// Whether the node and its children have been simplified.
        /// </summary>
        private bool simplified;
        
        /// <summary>
        /// Cached results of <see cref="AppendToString(StringBuilder)"/> calls on this node (only for simplified nodes).
        /// </summary>
        private string toStringVerboseCached;

        /// <summary>
        /// Prevents a default instance of the <see cref="RegexpTreeNode{TElement, TElementSet}"/> class from being created.
        /// </summary>
        private RegexpTreeNode()
        {
        }

        /// <summary>
        /// Gets the type of the node.
        /// </summary>
        public RegexpTreeNodeType Type { get; private set; }

        /// <summary>
        /// Gets the character set associated with the node.
        /// </summary>
        public IImmutableDistribution<TElement, TElementSet> ElementSet
        {
            get
            {
                if (this.Type != RegexpTreeNodeType.ElementSet)
                {
                    throw new InvalidOperationException("This node doesn't represent an element set.");
                }

                return this.elementSet;
            }
        }

        /// <summary>
        /// Gets the children of the node.
        /// </summary>
        public ReadOnlyCollection<RegexpTreeNode<TElement, TElementSet>> Children
        {
            get
            {
                return this.children.AsReadOnly();
            }
        }

        /// <summary>
        /// Creates a node representing language consisting of an empty string.
        /// </summary>
        /// <returns>The created node.</returns>
        public static RegexpTreeNode<TElement, TElementSet> Empty()
        {
            return CachedEmptyNode;
        }

        /// <summary>
        /// Returns a node representing the empty language.
        /// </summary>
        /// <returns>A node representing the empty language.</returns>
        public static RegexpTreeNode<TElement, TElementSet> Nothing()
        {
            return CachedNothingNode;
        }

        /// <summary>
        /// Creates a terminal node representing a given set of sequence elements.
        /// </summary>
        /// <param name="elementSet">The distribution over sequence elements.</param>
        /// <returns>The created node.</returns>
        public static RegexpTreeNode<TElement, TElementSet> FromElementSet(Option<TElementSet> elementSet)
        {
            Argument.CheckIfNotNull(elementSet, "elementSet");

            return new RegexpTreeNode<TElement, TElementSet>
            { 
                Type = RegexpTreeNodeType.ElementSet, 
                elementSet = elementSet.HasValue ? (IImmutableDistribution<TElement, TElementSet>)elementSet.Value : null /* Distribution.CreatePartialUniform(elementSet) */
            };
        }

        /// <summary>
        /// Creates a node representing the union of the languages described by a given pair of nodes.
        /// </summary>
        /// <param name="node1">The first node.</param>
        /// <param name="node2">The second node.</param>
        /// <returns>The created node.</returns>
        public static RegexpTreeNode<TElement, TElementSet> Or(RegexpTreeNode<TElement, TElementSet> node1, RegexpTreeNode<TElement, TElementSet> node2)
        {
            Argument.CheckIfNotNull(node1, "node1");
            Argument.CheckIfNotNull(node2, "node2");

            if (node1.Type == RegexpTreeNodeType.Empty && node2.Type == RegexpTreeNodeType.Empty)
            {
                // Identical children
                return node1;
            }
            
            if (node1.Type == RegexpTreeNodeType.Nothing)
            {
                // Union with an empty language is an identity
                return node2;
            }

            if (node2.Type == RegexpTreeNodeType.Nothing)
            {
                // Union with an empty language is an identity
                return node1;
            }

            var result = new RegexpTreeNode<TElement, TElementSet> { Type = RegexpTreeNodeType.Union };
            result.children.Capacity = 2;
            result.children.Add(node1);
            result.children.Add(node2);

            return result;
        }

        /// <summary>
        /// Creates a node representing the concatenation of the languages described by a given pair of nodes.
        /// </summary>
        /// <param name="node1">The first node.</param>
        /// <param name="node2">The second node.</param>
        /// <returns>The created node.</returns>
        public static RegexpTreeNode<TElement, TElementSet> Concat(RegexpTreeNode<TElement, TElementSet> node1, RegexpTreeNode<TElement, TElementSet> node2)
        {
            Argument.CheckIfNotNull(node1, "node1");
            Argument.CheckIfNotNull(node2, "node2");

            if (node1.Type == RegexpTreeNodeType.Nothing || node2.Type == RegexpTreeNodeType.Nothing)
            {
                // Concatenation with an empty language results in an empty language
                return Nothing();
            }

            if (node1.Type == RegexpTreeNodeType.Empty)
            {
                // Concatenation with an empty string is an identity
                return node2;
            }

            if (node2.Type == RegexpTreeNodeType.Empty)
            {
                // Concatenation with an empty string is an identity
                return node1;
            }
            
            var result = new RegexpTreeNode<TElement, TElementSet> { Type = RegexpTreeNodeType.Concat };
            result.children.Capacity = 2;
            result.children.Add(node1);
            result.children.Add(node2);

            return result;
        }

        /// <summary>
        /// Creates a node representing the Kleene star of the language described by a given node.
        /// </summary>
        /// <param name="node">The node.</param>
        /// <returns>The created node.</returns>
        public static RegexpTreeNode<TElement, TElementSet> Star(RegexpTreeNode<TElement, TElementSet> node)
        {
            Argument.CheckIfNotNull(node, "node");

            if (node.Type == RegexpTreeNodeType.Nothing ||
                node.Type == RegexpTreeNodeType.Empty ||
                node.Type == RegexpTreeNodeType.Star)
            {
                // In all these cases star doesn't affect its argument
                return node;
            }

            var result = new RegexpTreeNode<TElement, TElementSet> { Type = RegexpTreeNodeType.Star };
            result.children.Capacity = 1;
            result.children.Add(node);
            
            return result;
        }

        /// <summary>
        /// Returns a string representation of the language described by the node.
        /// </summary>
        /// <param name="builder">A string builder.</param>
        /// <returns>False if the string was truncated at the maximum length.</returns>
        public bool AppendToString(StringBuilder builder)
        {
            // Use the most explicit formatting by default, so that equal strings imply equal languages.
            return this.AppendToString(builder, VerboseFormattingSettings, false);
        }

        /// <summary>
        /// Returns a string representation of the language described by the node.
        /// </summary>
        /// <param name="builder">The string builder to append to.</param>
        /// <param name="formattingSettings">The formatting settings used for conversion from an automaton to a string.</param>
        /// <returns>False if the string was truncated.</returns>
        public bool AppendToString(StringBuilder builder, RegexpFormattingSettings formattingSettings)
        {
            Argument.CheckIfNotNull(formattingSettings, "formattingSettings");
            
            return this.AppendToString(builder, formattingSettings, false);
        }

        /// <summary>
        /// Simplifies the regular expression tree rooted at this node.
        /// </summary>
        /// <param name="collapseAlternatives">
        /// Specifies whether an attempt to merge identical sub-expressions should be made.
        /// Setting it to <see langword="false"/> will improve the performance, but produce longer regular expressions.
        /// </param>
        internal void Simplify(bool collapseAlternatives)
        {
            if (this.simplified)
            {
                return;
            }
            
            // Simplify children
            foreach (RegexpTreeNode<TElement, TElementSet> child in this.children)
            {
                child.Simplify(collapseAlternatives);
            }

            // Simplify this node
            switch (this.Type)
            {
                case RegexpTreeNodeType.Star:
                    this.SimplifyStar();
                    break;
                case RegexpTreeNodeType.Concat:
                    this.SimplifyConcat();
                    break;
                case RegexpTreeNodeType.Union:
                    this.SimplifyUnion(collapseAlternatives);
                    break;
            }

            this.simplified = true;
        }

        /// <summary>
        /// Appends the character ranges for a Discrete distribution over characters to a string builder.
        /// </summary>
        /// <param name="resultBuilder">The string builder.</param>
        /// <param name="discreteChar">The Discrete distribution over characters.</param>
        private static void AppendRangesForDiscreteChar(StringBuilder resultBuilder, DiscreteChar discreteChar)
        {
            var ranges = discreteChar.Ranges;
            if (ranges.Count > 1)
            {
                resultBuilder.Append('[');
                ranges.ForEach(range => AppendCharacterRange(resultBuilder, range));
                resultBuilder.Append(']');
            }
            else if (ranges.Count == 1)
            {
                AppendCharacterRange(resultBuilder, ranges.Single());
            }
        }

        /// <summary>
        /// Appends a character range to a string builder.
        /// </summary>
        /// <param name="resultBuilder">The string builder.</param>
        /// <param name="range">The character range.</param>
        private static void AppendCharacterRange(StringBuilder resultBuilder, ImmutableDiscreteChar.CharRange range)
        {
            resultBuilder.Append(@"\u");
            resultBuilder.Append(range.StartInclusive.ToString("X4"));
            resultBuilder.Append(@"-\u");
            resultBuilder.Append((range.EndExclusive - 1).ToString("X4"));
        }

        /// <summary>
        /// Returns a string representation of the language described by the node.
        /// </summary>
        /// <param name="resultBuilder">The string builder to append to.</param>
        /// <param name="formattingSettings">The formatting settings used for conversion from an automaton to a string.</param>
        /// <param name="useCache">Whether the value caching should be used.</param>
        /// <returns>False if the max length was reached, true otherwise.</returns>
        /// <remarks>
        /// Value caching must be used with care: once cached, a value would never be invalidated.
        /// That is why this method should be called with value caching only on those nodes that have already been simplified.
        /// <see cref="SimplifyUnion"/> uses this method in exactly this way.
        /// </remarks>
        private bool AppendToString(StringBuilder resultBuilder, RegexpFormattingSettings formattingSettings, bool useCache)
        {
            Debug.Assert(formattingSettings != null, "Valid formatting settings must be provided.");
            Debug.Assert(!useCache || this.simplified, "Caching must not be used for non-simplified nodes.");

            int lengthAtStart = resultBuilder.Length;
            if (lengthAtStart > formattingSettings.TruncationLength)
            {
                return false;
            }

            if (useCache && this.toStringVerboseCached != null && formattingSettings.Equals(VerboseFormattingSettings))
            {
                resultBuilder.Append(this.toStringVerboseCached);
                return true;
            }

            ////var resultBuilder = new StringBuilder();
            switch (this.Type)
            {
                case RegexpTreeNodeType.Empty:
                    break;

                case RegexpTreeNodeType.Nothing:
                    resultBuilder.Append('Ø');
                    break;

                case RegexpTreeNodeType.ElementSet:
                    if (typeof(TElement) == typeof(string))
                    {
                        resultBuilder.Append("<");
                        if (this.elementSet is DiscreteChar)
                        {
                            ((DiscreteChar)(object)this.elementSet).AppendToString(resultBuilder);
                        }
                        else
                        {
                            var stringForm = this.elementSet.ToString();
                            ////if (stringForm.Contains('?')) stringForm = "?";
                            resultBuilder.Append(stringForm);
                        }

                        resultBuilder.Append(">");
                    }
                    else
                    {
                        if (this.elementSet.IsPointMass)
                        {
                            if (this.elementSet is DiscreteChar)
                            {
                                var dc = (DiscreteChar)(object)this.elementSet;
                                dc.AppendRegex(resultBuilder);
                            }
                            else
                            {
                                resultBuilder.Append(this.elementSet.Point);
                            }
                        }
                        else if (this.elementSet.IsUniform() || formattingSettings.IgnoreElementDistributionDetails)
                        {
                            resultBuilder.Append(formattingSettings.ShowAnyElementAsQuestionMark ? '?' : '.');
                        }
                        else
                        {
                            if (this.elementSet is DiscreteChar)
                            {
                                var dc = (DiscreteChar)(object)this.elementSet;
                                dc.AppendRegex(resultBuilder);
                            }
                            else
                            {
                                // not a discrete char... What to do here?
                                resultBuilder.Append(this.elementSet);
                            }
                        }
                    }                    
                    
                    break;

                case RegexpTreeNodeType.Union:
                    bool withSquareBrackets = formattingSettings.PutOptionalInSquareBrackets && this.IsOptional();
                    if (withSquareBrackets)
                    {
                        resultBuilder.Append('[');
                    }

                    bool hasPrevious = false;
                    for (int i = 0; i < this.children.Count; ++i)
                    {
                        if (!withSquareBrackets || this.children[i].Type != RegexpTreeNodeType.Empty)
                        {
                            if (hasPrevious)
                            {
                                resultBuilder.Append('|');
                            }

                            if (!this.children[i].AppendRegexpWithBrackets(resultBuilder, RegexpTreeNodeType.Union, formattingSettings, useCache))
                            {
                                return false;
                            }

                            hasPrevious = true;
                        }
                    }

                    if (withSquareBrackets)
                    {
                        resultBuilder.Append(']');
                    }

                    break;

                case RegexpTreeNodeType.Concat:
                    for (int i = 0; i < this.children.Count; ++i)
                    {
                        if (!this.children[i].AppendRegexpWithBrackets(resultBuilder, RegexpTreeNodeType.Concat, formattingSettings, useCache))
                        {
                            return false;
                        }
                    }

                    break;

                case RegexpTreeNodeType.Star:
                    if (!this.children[0].AppendRegexpWithBrackets(resultBuilder, RegexpTreeNodeType.Star, formattingSettings, useCache))
                    {
                        return false;
                    }

                    resultBuilder.Append(formattingSettings.UseLazyQuantifier ? "*?" : "*");
                    break;

                default:
                    Debug.Fail("Unhandled operation!");
                    break;
            }

            if (useCache && formattingSettings.Equals(VerboseFormattingSettings))
            {
            ////    this.toStringVerboseCached = resultBuilder.ToString(lengthAtStart, resultBuilder.Length-lengthAtStart);
            }

            return true;
        }

        /// <summary>
        /// Simplifies a node representing union.
        /// </summary>
        /// <param name="collapseAlternatives">
        /// Specifies whether an attempt to merge identical sub-expressions should be made.
        /// Setting it to <see langword="false"/> will improve the performance, but produce longer regular expressions.
        /// </param>
        private void SimplifyUnion(bool collapseAlternatives)
        {
            Debug.Assert(this.Type == RegexpTreeNodeType.Union, "Must be run for union nodes only.");

            //// The simplification can affect the node itself, but not its children
            //// since children are shared by multiple parents.
            
            var newChildren = new List<RegexpTreeNode<TElement, TElementSet>>();

            for (int i = 0; i < this.children.Count; ++i)
            {
                switch (this.children[i].Type)
                {
                    case RegexpTreeNodeType.Nothing:
                        Debug.Fail("Should have been optimized away.");
                        continue;
                    case RegexpTreeNodeType.Union:
                        // Move union upwards
                        newChildren.AddRange(this.children[i].children);
                        break;
                    default:
                        newChildren.Add(this.children[i]);
                        break;
                }
            }

            this.children = newChildren;
            Debug.Assert(this.children.Count > 0, "Should have been optimized away.");
            
            // Find alternatives with identical regexp representations
            if (this.children.Count > 1 && collapseAlternatives)
            {
                var regexpToNode = new SmallStringKeyDictionary<RegexpTreeNode<TElement, TElementSet>>();
                var builder = new StringBuilder();
                bool success = true;
                for (int i = 0; i < this.children.Count; ++i)
                {
                    builder.Clear();
                    this.children[i].AppendToString(builder, VerboseFormattingSettings, true);
                    if (builder.Length >= VerboseFormattingSettings.TruncationLength)
                    {
                        // Bail out if we reach the truncation length, otherwise we risk
                        // removing non-identical alternatives.
                        success = false;
                        break;
                    }

                    regexpToNode.AddIfMissing(builder.ToString(), this.children[i]);
                }

                // Remove alternatives that are definitely duplicates
                if (success)
                {
                    this.children = regexpToNode.UniqueValues;
                }
            }
            
            // Union with one argument is identity
            if (this.children.Count == 1)
            {
                this.SetTo(this.children[0]);
            }
        }

        /// <summary>
        /// Simplifies a node representing concatenation.
        /// </summary>
        private void SimplifyConcat()
        {
            Debug.Assert(this.Type == RegexpTreeNodeType.Concat, "Must be run for concatenation nodes only.");

            //// The simplification can affect the node itself, but not its children
            //// since children are shared by multiple parents.
            
            var newChildren = new List<RegexpTreeNode<TElement, TElementSet>>();

            for (int i = 0; i < this.children.Count; ++i)
            {
                switch (this.children[i].Type)
                {
                    case RegexpTreeNodeType.Nothing:
                    case RegexpTreeNodeType.Empty:
                        Debug.Fail("Should have been optimized away.");
                        break;
                    case RegexpTreeNodeType.Concat:
                        // Move concatenation upwards
                        newChildren.AddRange(this.children[i].children);
                        break;
                    default:
                        newChildren.Add(this.children[i]);
                        break;
                }
            }

            this.children = newChildren;
            Debug.Assert(this.children.Count >= 2, "Should have been optimized away");
        }

        /// <summary>
        /// Simplifies a node representing Kleene star.
        /// </summary>
        private void SimplifyStar()
        {
            Debug.Assert(this.Type == RegexpTreeNodeType.Star, "Must be run for Kleene star nodes only.");
            Debug.Assert(this.children.Count == 1, "A Kleene star node must have a single child.");

            //// The simplification can affect the node itself, but not its children
            //// since children are shared by multiple parents.

            RegexpTreeNode<TElement, TElementSet> child = this.children[0];
            switch (child.Type)
            {
                case RegexpTreeNodeType.Empty:
                case RegexpTreeNodeType.Nothing:
                case RegexpTreeNodeType.Star:
                    Debug.Fail("Should have been optimized away.");
                    break;
                case RegexpTreeNodeType.Union:
                    // 'Star' accepts the empty string anyway, no need to provide it as an alternative
                    if (child.children.Any(c => c.Type == RegexpTreeNodeType.Empty))
                    {
                        List<RegexpTreeNode<TElement, TElementSet>> nonEmptyUnionChildren =
                            child.children.Where(c => c.Type != RegexpTreeNodeType.Empty).ToList();
                        if (nonEmptyUnionChildren.Count == 0)
                        {
                            // Empty only
                            this.SetTo(RegexpTreeNode<TElement, TElementSet>.Empty());
                        }
                        else if (nonEmptyUnionChildren.Count == 1)
                        {
                            this.children.Clear();
                            this.children.AddRange(nonEmptyUnionChildren);
                            Debug.Assert(this.children.Count == 1, "There must be only one such node.");
                        }
                        else
                        {
                            // Can't modify child's children, so create a new node
                            this.children.Clear();
                            var newUnionNode = new RegexpTreeNode<TElement, TElementSet> { Type = RegexpTreeNodeType.Union, children = nonEmptyUnionChildren, simplified = true };
                            this.children.Add(newUnionNode);
                        }
                    }

                    break;
            }
        }

        /// <summary>
        /// Copies the state of this node from another node.
        /// </summary>
        /// <param name="other">The node to copy the state from.</param>
        private void SetTo(RegexpTreeNode<TElement, TElementSet> other)
        {
            Debug.Assert(other != null, "A valid node must be provided.");

            this.Type = other.Type;
            this.elementSet = other.elementSet;
            this.children.Clear();
            this.children.AddRange(other.children);
            this.simplified = other.simplified;
            this.toStringVerboseCached = other.toStringVerboseCached;
        }

        /// <summary>
        /// Checks whether this node represents an optional language (union of an empty string and some language).
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if this node represents an optional language, <see langword="false"/> otherwise.
        /// </returns>
        private bool IsOptional()
        {
            return this.Type == RegexpTreeNodeType.Union && this.children.Any(c => c.Type == RegexpTreeNodeType.Empty);
        }

        /// <summary>
        /// Appends a regular expression representation of the language described by the node to a given string builder,
        /// adding brackets if the parent operation has higher priority.
        /// </summary>
        /// <param name="resultBuilder">The string builder.</param>
        /// <param name="parentType">The type of the parent node.</param>
        /// <param name="formattingSettings">The formatting settings used for conversion from an automaton to a string.</param>
        /// <param name="useCache">Whether the value caching should be used (see <see cref="AppendToString(StringBuilder, RegexpFormattingSettings, bool)"/>).</param>
        /// <returns>Returns true if successful, false otherwise.</returns>
        private bool AppendRegexpWithBrackets(
            StringBuilder resultBuilder, RegexpTreeNodeType parentType, RegexpFormattingSettings formattingSettings, bool useCache)
        {
            Debug.Assert(formattingSettings != null, "A valid formatting settings description must be provided.");

            bool addBrackets = this.Type < parentType && !(formattingSettings.PutOptionalInSquareBrackets && this.IsOptional());
            
            if (addBrackets)
            {
                resultBuilder.Append('(');
            }

            if (!this.AppendToString(resultBuilder, formattingSettings, useCache))
            {
                return false;
            }

            if (addBrackets)
            {
                resultBuilder.Append(')');
            }

            return true;
        }

        /// <summary>
        /// A custom dictionary for mapping strings to values. A key lookup takes linear time.
        /// </summary>
        /// <remarks>
        /// Can be faster than <see cref="Dictionary{TKey, TValue}"/> when the number of strings is small, and the strings are long.
        /// There are two reasons for that. First, the linear search procedure used has almost no overhead.
        /// Second, <see cref="Dictionary{TKey, TValue}"/> needs to look through every string to compute its hash code,
        /// while the linear search procedure usually looks only at a couple characters in every string,
        /// so it can work with long strings without much performance loss.
        /// </remarks>
        /// <typeparam name="TValue">The type of a value.</typeparam>
        private class SmallStringKeyDictionary<TValue>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="SmallStringKeyDictionary{TValue}"/> class.
            /// </summary>
            public SmallStringKeyDictionary()
            {
                this.UniqueKeys = new List<string>();
                this.UniqueValues = new List<TValue>();
            }
            
            /// <summary>
            /// Gets the list of unique keys in the dictionary.
            /// </summary>
            public List<string> UniqueKeys { get; private set; }

            /// <summary>
            /// Gets the list of values corresponding to <see cref="UniqueKeys"/>.
            /// </summary>
            public List<TValue> UniqueValues { get; private set; }

            /// <summary>
            /// Adds a given key-value pair to the dictionary, if it doesn't yet contain the given key.
            /// </summary>
            /// <param name="key">The key.</param>
            /// <param name="value">The value.</param>
            public void AddIfMissing(string key, TValue value)
            {
                for (int i = 0; i < this.UniqueKeys.Count; ++i)
                {
                    if (this.UniqueKeys[i] == key)
                    {
                        return;
                    }
                }

                this.UniqueKeys.Add(key);
                this.UniqueValues.Add(value);
            }
        }
    }
}

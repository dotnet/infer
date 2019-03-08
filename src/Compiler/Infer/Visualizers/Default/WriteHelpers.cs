// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// Class for help indent text writing.
    /// </summary>
    internal class WriteHelper
    {
        // atomic shift character sequence
        private const string shift = "    ";

        /// <summary>
        /// Write an indent character sequence of needed level.
        /// </summary>
        /// <param name="writer">Writer for a text.</param>
        /// <param name="count">Level of indenting.</param>
        public static void Indent(TextWriter writer, int count)
        {
            StringBuilder space = new StringBuilder();

            for (int i = 0; i < count; i++)
            {
                space.Append(shift);
            }

            writer.Write(space);
        }

        /// <summary>
        /// Write string with indenting.
        /// </summary>
        /// <param name="writer">Writer for a text.</param>
        /// <param name="line">String for writing.</param>
        /// <param name="level">Indent level.</param>
        public static void WriteIndent(TextWriter writer, string line, int level)
        {
            Indent(writer, level);
            writer.Write(line);
        }

        /// <summary>
        /// Write string with indenting and making new line.
        /// </summary>
        /// <param name="writer">Writer for a text.</param>
        /// <param name="line">String for writing.</param>
        /// <param name="level">Indent level.</param>
        public static void WriteLineIndent(TextWriter writer, string line, int level)
        {
            Indent(writer, level);
            writer.WriteLine(line);
        }
    }



    /// <summary>
    /// Holds helping methods for writing in DOT files.
    /// </summary>
    class DotWriteHelper : WriteHelper
    {
        /// <summary>
        /// Write attribute of DOT object.
        /// </summary>
        /// <param name="writer">Text writer assigned to a DOT file.</param>
        /// <param name="name">Name of the attribute.</param>
        /// <param name="defaultValue">Default value of the attribute.</param>
        /// <param name="value">Value of the attribute. If it equals default value or null, the attribute isn't written.</param>
        /// <param name="converter">Converter that represent value of attribute as a string. If null, ToString() method of value is used.</param>
        public static void WriteAttribute(TextWriter writer, string name, object defaultValue, object value, Func<object, string> converter = null)
        {
            if (value == null || value.Equals(defaultValue))
            {
                return;
            }
            string stringValue = converter == null ? value.ToString() : converter(value);
            writer.Write($"{name}=\"{stringValue}\" ");
        }
    }



    /// <summary>
    /// Holds helping methods for writing HTML files.
    /// </summary>
    class HtmlWriteHelper : WriteHelper
    {
        private int indentLevel = 0;
        private StringWriter store = new StringWriter();
        private Stack<string> tagStack = new Stack<string>();

        /// <summary>
        /// Open a new tag.
        /// </summary>
        /// <param name="tagName">Opening tag.</param>
        /// <param name="addingInfo">Extra info for a tag.</param>
        /// <param name="close">If true, tag will closed automatically.</param>
        public void OpenTag(string tagName, string addingInfo = null, bool close = false)
        {
            bool clearIndent = (tagName == "pre" || tagName == "code");
            if (clearIndent) indentLevel = 0;
            if (addingInfo != null)
            {
                WriteIndent(store, $"<{tagName} {addingInfo}>", indentLevel);
            }
            else
            {
                WriteIndent(store, $"<{tagName}>", indentLevel);
            }

            if (close)
            {
                store.WriteLine($"</{tagName}>");
            }
            else
            {
                if (!clearIndent)
                {
                    store.WriteLine();
                    indentLevel++;
                }
                tagStack.Push(tagName);
            }
        }

        /// <summary>
        /// Append text to a store without indenting.
        /// </summary>
        /// <param name="text"></param>
        public void Append(string text)
        {
            store.Write(text);
        }

        /// <summary>
        /// Write a string with indenting.
        /// </summary>
        /// <param name="text"></param>
        public void Write(string text)
        {
            WriteIndent(store, text, indentLevel);
        }

        /// <summary>
        /// Write new line string using indenting.
        /// </summary>
        /// <param name="text"></param>
        public void WriteLine(string text = null)
        {
            if (text == null)
            {
                store.WriteLine();
            }
            else
            {
                WriteLineIndent(store, text, indentLevel);
            }
        }

        /// <summary>
        /// Close current tag.
        /// </summary>
        public void CloseTag()
        {
            var tag = tagStack.Pop();
            indentLevel--;
            WriteLineIndent(store, "</" + tag + ">", indentLevel);
        }

        /// <summary>
        /// Save generated HTML text to a file.
        /// </summary>
        /// <param name="path"></param>
        public void SaveToFile(string path)
        {
            using (var writer = new StreamWriter(path, false))
            {
                writer.Write(store.ToString());
            }
        }

        /// <summary>
        /// Get a string holding generated HTML text.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return store.ToString();
        }
    }
}

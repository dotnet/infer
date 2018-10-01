// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;

    using Xunit;

    using Microsoft.ML.Probabilistic.Learners.Runners;

    

    /// <summary>
    /// Tests for <see cref="CommandLineParser"/> class.
    /// </summary>
    public class CommandLineParserTests
    {
        /// <summary>
        /// Tests if it is impossible to register with an invalid description.
        /// </summary>
        [Fact]
        public void InvalidParameterRegistrationTest()
        {
            int p = 0;
            Action<int> handler = v => p = v;
            var parser = new CommandLineParser();
            
            // Null parameter name
            Assert.Throws<ArgumentException>(() => parser.RegisterParameterHandler(null, "P", "P", handler, CommandLineParameterType.Required));
            
            // Empty parameter name
            Assert.Throws<ArgumentException>(() => parser.RegisterParameterHandler(string.Empty, "P", "P", handler, CommandLineParameterType.Required));

            // Null parameter description
            Assert.Throws<ArgumentException>(() => parser.RegisterParameterHandler("-p", null, "P", handler, CommandLineParameterType.Required));

            // Empty parameter description
            Assert.Throws<ArgumentException>(() => parser.RegisterParameterHandler("-p", string.Empty, "P", handler, CommandLineParameterType.Required));

            // Null parameter value description
            Assert.Throws<ArgumentException>(() => parser.RegisterParameterHandler("-p", "P", null, handler, CommandLineParameterType.Required));

            // Empty parameter description
            Assert.Throws<ArgumentException>(() => parser.RegisterParameterHandler("-p", "P", string.Empty, handler, CommandLineParameterType.Required));

            // Null handler
            Assert.Throws<ArgumentNullException>(() => parser.RegisterParameterHandler("-p", "P", "P", (Action<int>)null, CommandLineParameterType.Required));

            // Should work
            parser.RegisterParameterHandler("-p", "P", "P", handler, CommandLineParameterType.Required);

            // Same parameter registered twice
            Assert.Throws<ArgumentException>(() => parser.RegisterParameterHandler("-p", "P", "P", handler, CommandLineParameterType.Required));
        }
        
        /// <summary>
        /// Tests command line parsing in different circumstances.
        /// </summary>
        [Fact]
        public void ParseParametersTest()
        {
            int p = 0;
            double q = 0, r = 3;
            string s = "abc";
            bool f = false;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("-p", "P", "P", (int v) => p = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("-q", "Q", "Q", (double v) => q = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("-r", "R", "R", (double v) => r = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("-s", "S", "S", (string v) => s = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("-f", "F", () => f = true);

            // Without optional parameters
            Assert.True(parser.TryParse(new[] { "-p", "1", "-q", "2.3" }, string.Empty));
            Assert.Equal(1, p);
            Assert.Equal(2.3, q);
            Assert.Equal(3.0, r);
            Assert.Equal("abc", s);
            Assert.False(f);

            // With optional parameters
            Assert.True(parser.TryParse(new[] { "-p", "1", "-q", "2.3", "-r", "-7.4", "-s", "def", "-f" }, string.Empty));
            Assert.Equal(1, p);
            Assert.Equal(2.3, q);
            Assert.Equal(-7.4, r);
            Assert.Equal("def", s);
            Assert.True(f);
        }

        /// <summary>
        /// Tests if invalid command lines are identified correctly.
        /// </summary>
        [Fact]
        public void ParseInvalidParametersTest()
        {
            int p = 0;
            double q = 0;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("-p", "P", "P", (int v) => p = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("-q", "Q", "Q", (double v) => q = v, CommandLineParameterType.Required);

            // Should work
            Assert.True(parser.TryParse(new[] { "-p", "-1", "-q", "1" }, string.Empty));
            Assert.Equal(-1, p);
            Assert.Equal(1, q);

            // Missing required parameter
            Assert.False(parser.TryParse(new[] { "-p", "-1" }, string.Empty));

            // Missing parameter value
            Assert.False(parser.TryParse(new[] { "-p", "-1", "-q" }, string.Empty));

            // Unknown parameter
            Assert.False(parser.TryParse(new[] { "-p", "-1", "-q", "1", "-r", "2" }, string.Empty));

            // Unknown parameter because of missing argument
            Assert.False(parser.TryParse(new[] { "-p", "-q", "1" }, string.Empty));

            // Integer parameter parse error
            Assert.False(parser.TryParse(new[] { "-p", "1.3", "-q", "1" }, string.Empty));

            // Double parameter parse error
            Assert.False(parser.TryParse(new[] { "-p", "1", "-q", "XX" }, string.Empty));
        }
    }
}

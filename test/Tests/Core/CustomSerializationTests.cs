// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.IO;
    using Xunit;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Serialization;

    using Assert = Xunit.Assert;

    /// <summary>
    /// Tests for custom binary serialization.
    /// </summary>
    public class CustomSerializationTests
    {
        /// <summary>
        /// Tests custom binary serialization and deserialization of a <see cref="Gaussian"/> distribution.
        /// </summary>
        [Fact]
        public void GaussianCustomSerializationTest()
        {
            using (var stream = new MemoryStream())
            {
                var writer = new WrappedBinaryWriter(new BinaryWriter(stream));
                writer.Write(Gaussian.FromNatural(-13.89, 436.12)); 
                writer.Write(Gaussian.PointMass(Math.PI)); 
                writer.Write(Gaussian.Uniform()); 

                stream.Seek(0, SeekOrigin.Begin);

                using (var reader = new WrappedBinaryReader(new BinaryReader(stream)))
                {
                    Gaussian natural = reader.ReadGaussian();
                    Assert.Equal(-13.89, natural.MeanTimesPrecision);
                    Assert.Equal(436.12, natural.Precision);

                    Gaussian pointMass = reader.ReadGaussian();
                    Assert.True(pointMass.IsPointMass);
                    Assert.Equal(Math.PI, pointMass.GetMean());

                    Gaussian uniform = reader.ReadGaussian();
                    Assert.True(uniform.IsUniform());
                }
            }
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of a <see cref="Gamma"/> distribution.
        /// </summary>
        [Fact]
        public void GammaCustomSerializationTest()
        {
            using (var stream = new MemoryStream())
            {
                var writer = new WrappedBinaryWriter(new BinaryWriter(stream));
                writer.Write(Gamma.FromShapeAndRate(1.0, 10.0)); 
                writer.Write(Gamma.PointMass(Math.PI));
                writer.Write(Gamma.Uniform()); 

                stream.Seek(0, SeekOrigin.Begin);

                using (var reader = new WrappedBinaryReader(new BinaryReader(stream)))
                {
                    Gamma shapeAndRate = reader.ReadGamma();
                    Assert.Equal(1.0, shapeAndRate.Shape);
                    Assert.Equal(10.0, shapeAndRate.Rate);

                    Gamma pointMass = reader.ReadGamma();
                    Assert.True(pointMass.IsPointMass);
                    Assert.Equal(Math.PI, pointMass.GetMean());

                    Gamma uniform = reader.ReadGamma();
                    Assert.True(uniform.IsUniform());
                }
            }
        }
    }
}

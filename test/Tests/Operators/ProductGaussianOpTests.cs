using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;

using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class ProductGaussianOpTests
    {
        internal void ProductGaussianGammaVmpOpTest()
        {
            double prec = 1;
            double a = 2;
            Gamma B = Gamma.FromShapeAndRate(1e-3, 1e-3);
            //Gamma B = Gamma.FromShapeAndRate(1, 1);
            for (int i = 0; i < 10; i++)
            {
                prec = (i + 1) * 4.0e-15;
                Gamma result = ProductGaussianGammaVmpOp.BAverageLogarithm(Gaussian.FromNatural(0.5, prec), Gaussian.PointMass(a), B, Gamma.Uniform());
                Gamma result2 = ProductGaussianGammaVmpOp.BAverageLogarithm(Gaussian.FromNatural(0.5, prec), Gaussian.PointMass(a), B, Gamma.Uniform());
                Console.WriteLine("{0} {1}", result, result2);
                //Console.WriteLine("rate = {0}", result.Rate);
                // rate = -0.5*A
            }
        }

        [Fact]
        public void ProductOpTest()
        {
            GaussianProductOp.ProductAverageConditional(Gaussian.FromNatural(-1765.4546871987516, 1338.3779930136373), Gaussian.FromNatural(47353.238656616719, 51908.220427509987), Gaussian.FromNatural(-110433.17966670298, 385287.49472649547));
            GaussianProductOp_Slow.AAverageConditional(Gaussian.FromNatural(1.2464052053935622E-100, 8.057496451574595E-100), Gaussian.FromNatural(367.30950323702245, 328.35007985731033), Gaussian.FromNatural(951.91707682214258, 6998.5956711899107));
            GaussianProductOp_Slow.AAverageConditional(Gaussian.FromNatural(-32.320457480359039, 43.195460703170248), Gaussian.FromNatural(-41.083697150214341, 22.757594736100287), Gaussian.FromNatural(31.117245365484337, 27.164520772286647));
            Assert.True(GaussianProductOp_Slow.ProductAverageConditional(
                Gaussian.FromNatural(0.0019528178431691338, 3.25704676859826E-06),
                Gaussian.FromNatural(-1.4311468676808659E-17, 5.4527745979495584E-21),
                Gaussian.FromNatural(2157531.2967657731, 1830.6558666566498)).Precision > 0); // Gaussian.FromNatural()
            Assert.True(GaussianProductOp_Slow.ProductAverageConditional(
                Gaussian.FromNatural(6.2927332361739073E-13, 0.00099999696614431447),
                Gaussian.FromNatural(-3.4418586572681724E-14, 4.2560312731693555E-12),
                Gaussian.FromNatural(-2.58594546750577, 0.1)).Precision > 0); // Gaussian.FromNatural(1.262909232222487E-15, 6.6429067108186857E-15)
            Assert.True(GaussianProductOp_Slow.ProductAverageConditional(
                Gaussian.FromNatural(1.0947079898711334E-15, 0.00099999696759297),
                Gaussian.FromNatural(-9.7432361496028023E-16, 2.630301623215118E-09),
                Gaussian.FromNatural(-2.5848205615986561, 0.1)).Precision > 0); // Gaussian.FromNatural(7.0300069947451969E-17, 4.2701144269824409E-12)

            var testCases = new[]
            {
                (4.94065645841247E-324, 0, -1e-5, 29),
                (4.94065645841247E-324, 1e-320, -1e-5, 29),
                (4.94065645841247E-324, 1, -1e-5, 29),
                (4.94065645841247E-324, 100, -1e-5, 29),
                (2.0, 0.0, 3.0, 5.0),
                (0.0, 0.0, 3.0, 5.0),
                (2, 4, 3, 5),
            };
            double tolerance = 1e-14;
            foreach (var testCase in testCases)
            {
                double AMean = testCase.Item1;
                double AVariance = testCase.Item2;
                double BMean = testCase.Item3;
                double BVariance = testCase.Item4;
                Gaussian expected = Gaussian.FromMeanAndVariance(AMean * BMean, AMean * AMean * BVariance + BMean * BMean * AVariance + AVariance * BVariance);
                if (AVariance == 0)
                {
                    double A = AMean;
                    expected = Gaussian.FromMeanAndVariance(A * BMean, A * A * BVariance);
                    Assert.True(GaussianProductVmpOp.ProductAverageLogarithm(
                        A,
                        new Gaussian(BMean, BVariance)).MaxDiff(expected) < tolerance);
                    Assert.True(GaussianProductOp.ProductAverageConditional(
                        A,
                        new Gaussian(BMean, BVariance)).MaxDiff(expected) < tolerance);
                    Assert.True(GaussianProductOp.ProductAverageConditional(new Gaussian(0, 1),
                      Gaussian.PointMass(A),
                      Gaussian.FromMeanAndVariance(BMean, BVariance)).MaxDiff(expected) < tolerance);
                }
                Assert.True(GaussianProductVmpOp.ProductAverageLogarithm(
                  Gaussian.FromMeanAndVariance(AMean, AVariance),
                  Gaussian.FromMeanAndVariance(BMean, BVariance)).MaxDiff(expected) < tolerance);
                Assert.True(GaussianProductOp.ProductAverageConditional(Gaussian.Uniform(),
                  Gaussian.FromMeanAndVariance(AMean, AVariance),
                  Gaussian.FromMeanAndVariance(BMean, BVariance)).MaxDiff(expected) < tolerance);
                Assert.True(GaussianProductOp.ProductAverageConditional(Gaussian.FromMeanAndVariance(0, 1e16),
                  Gaussian.FromMeanAndVariance(AMean, AVariance),
                  Gaussian.FromMeanAndVariance(BMean, BVariance)).MaxDiff(expected) < 1e-5);
            }

            Assert.True(GaussianProductOp.AAverageConditional(
                Gaussian.FromNatural(0.11690888200261176, 0.00021561758318567543),
                Gaussian.FromNatural(2.3189502343045755E-17, 0.00024804962073216578),
                Gaussian.FromNatural(3825912.925085804, 3815545.0940052439)).Precision > 0); // Gaussian.FromNatural(0.11722698913727095, 0.00021679261874209184)
            Assert.True(GaussianProductOp.AAverageConditional(6.0, 2.0)
              .MaxDiff(Gaussian.PointMass(6.0 / 2.0)) < tolerance);
            Assert.True(GaussianProductOp.AAverageConditional(6.0, new Gaussian(1, 3), Gaussian.PointMass(2.0))
              .MaxDiff(Gaussian.PointMass(6.0 / 2.0)) < tolerance);
            Assert.True(GaussianProductOp.AAverageConditional(0.0, 0.0).IsUniform());
            Assert.True(GaussianProductOp.AAverageConditional(Gaussian.Uniform(), 2.0).IsUniform());
            Assert.True(GaussianProductOp.AAverageConditional(Gaussian.Uniform(), new Gaussian(1, 3), Gaussian.PointMass(2.0)).IsUniform());
            Assert.True(GaussianProductOp.AAverageConditional(Gaussian.Uniform(), new Gaussian(1, 3), new Gaussian(2, 4)).IsUniform());

            Gaussian aPrior = Gaussian.FromMeanAndVariance(0.0, 1000.0);
            Assert.True((GaussianProductOp.AAverageConditional(
              Gaussian.FromMeanAndVariance(10.0, 1.0),
              aPrior,
              Gaussian.FromMeanAndVariance(5.0, 1.0)) * aPrior).MaxDiff(
              Gaussian.FromMeanAndVariance(2.208041421368822, 0.424566765678152)) < 1e-4);

            Gaussian g = new Gaussian(0, 1);
            Assert.True(GaussianProductOp.AAverageConditional(g, 0.0).IsUniform());
            Assert.True(GaussianProductOp.AAverageConditional(0.0, 0.0).IsUniform());
            Assert.True(GaussianProductVmpOp.AAverageLogarithm(g, 0.0).IsUniform());
            Assert.True(Gaussian.PointMass(3.0).MaxDiff(GaussianProductVmpOp.AAverageLogarithm(6.0, 2.0)) < tolerance);
            Assert.True(GaussianProductVmpOp.AAverageLogarithm(0.0, 0.0).IsUniform());
            try
            {
                Assert.True(GaussianProductVmpOp.AAverageLogarithm(6.0, g).IsUniform());
                Assert.True(false, "Did not throw NotSupportedException");
            }
            catch (NotSupportedException)
            {
            }
            try
            {
                g = GaussianProductOp.AAverageConditional(12.0, 0.0);
                Assert.True(false, "Did not throw AllZeroException");
            }
            catch (AllZeroException)
            {
            }
            try
            {
                g = GaussianProductVmpOp.AAverageLogarithm(12.0, 0.0);
                Assert.True(false, "Did not throw AllZeroException");
            }
            catch (AllZeroException)
            {
            }
        }

        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void GaussianProductOp_APointMassTest()
        {
            using (TestUtils.TemporarilyAllowGaussianImproperMessages)
            {
                Gaussian Product = Gaussian.FromMeanAndVariance(1.3, 0.1);
                Gaussian B = Gaussian.FromMeanAndVariance(1.24, 0.04);
                GaussianProductOp_APointMass(1, Product, B);

                Product = Gaussian.FromMeanAndVariance(10, 1);
                B = Gaussian.FromMeanAndVariance(5, 1);
                GaussianProductOp_APointMass(2, Product, B);

                Product = Gaussian.FromNatural(1, 0);
                GaussianProductOp_APointMass(2, Product, B);
            }
        }

        private void GaussianProductOp_APointMass(double aMean, Gaussian Product, Gaussian B)
        {
            bool isProper = Product.IsProper();
            Gaussian A = Gaussian.PointMass(aMean);
            Gaussian result = GaussianProductOp.AAverageConditional(Product, A, B);
            Console.WriteLine("{0}: {1}", A, result);
            Gaussian result2 = isProper ? GaussianProductOp_Slow.AAverageConditional(Product, A, B) : result;
            Console.WriteLine("{0}: {1}", A, result2);
            Assert.True(result.MaxDiff(result2) < 1e-6);
            var Amsg = InnerProductOp_PointB.BAverageConditional(Product, DenseVector.FromArray(B.GetMean()), new PositiveDefiniteMatrix(new double[,] { { B.GetVariance() } }), VectorGaussian.PointMass(aMean), VectorGaussian.Uniform(1));
            //Console.WriteLine("{0}: {1}", A, Amsg);
            Assert.True(result.MaxDiff(Amsg.GetMarginal(0)) < 1e-6);
            double prevDiff = double.PositiveInfinity;
            for (int i = 3; i < 40; i++)
            {
                double v = System.Math.Pow(0.1, i);
                A = Gaussian.FromMeanAndVariance(aMean, v);
                result2 = isProper ? GaussianProductOp.AAverageConditional(Product, A, B) : result;
                double diff = result.MaxDiff(result2);
                Console.WriteLine("{0}: {1} diff={2}", A, result2, diff.ToString("g4"));
                //Assert.True(diff <= prevDiff || diff < 1e-6);
                result2 = isProper ? GaussianProductOp_Slow.AAverageConditional(Product, A, B) : result;
                diff = result.MaxDiff(result2);
                Console.WriteLine("{0}: {1} diff={2}", A, result2, diff.ToString("g4"));
                Assert.True(diff <= prevDiff || diff < 1e-6);
                prevDiff = diff;
            }
        }

        [Fact]
        public void GaussianProductOp_ProductPointMassTest()
        {
            Gaussian A = new Gaussian(1, 2);
            Gaussian B = new Gaussian(3, 4);
            Gaussian pointMass = Gaussian.PointMass(4);
            Gaussian to_pointMass = GaussianProductOp.ProductAverageConditional(pointMass, A, B);
            //Console.WriteLine(to_pointMass);
            double prevDiff = double.PositiveInfinity;
            for (int i = 0; i < 100; i++)
            {
                Gaussian Product = Gaussian.FromMeanAndVariance(pointMass.Point, System.Math.Pow(10, -i));
                Gaussian to_product = GaussianProductOp.ProductAverageConditional(Product, A, B);
                //Gaussian to_product2 = GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
                //double evidence = GaussianProductOp.LogEvidenceRatio(Product, A, B, to_product);
                double diff = to_product.MaxDiff(to_pointMass);
                //Console.WriteLine($"{Product} {to_product} {evidence} {diff}");
                Assert.True(diff <= prevDiff || diff < 1e-6);
                prevDiff = diff;
            }
        }

        [Fact]
        public void ProductOpTest3()
        {
            Gaussian Product = new Gaussian(3.207, 2.222e-06);
            Gaussian A = new Gaussian(2.854e-06, 1.879e-05);
            Gaussian B = new Gaussian(0, 1);
            Gaussian result = GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
            Console.WriteLine(result);
            Assert.False(double.IsNaN(result.Precision));

            Product = Gaussian.FromNatural(2, 1);
            A = Gaussian.FromNatural(0, 3);
            B = Gaussian.FromNatural(0, 1);
            result = GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
            Console.WriteLine("{0}: {1}", Product, result);

            Product = Gaussian.FromNatural(129146.60457039363, 320623.20967711863);
            A = Gaussian.FromNatural(-0.900376203577801, 0.00000001);
            B = Gaussian.FromNatural(0, 1);
            result = GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
            Console.WriteLine("{0}: {1}", Product, result);

            Assert.True(GaussianProductOp_Slow.ProductAverageConditional(
              Gaussian.FromMeanAndVariance(0.0, 1000.0),
              Gaussian.FromMeanAndVariance(2.0, 3.0),
              Gaussian.FromMeanAndVariance(5.0, 1.0)).MaxDiff(
              Gaussian.FromMeanAndVariance(9.911, 79.2)
              // Gaussian.FromMeanAndVariance(12.110396063215639,3.191559311624262e+002)
              ) < 1e-4);

            A = new Gaussian(2, 3);
            B = new Gaussian(4, 5);
            Product = Gaussian.PointMass(2);
            result = GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
            Console.WriteLine("{0}: {1}", Product, result);
            double prevDiff = double.PositiveInfinity;
            for (int i = 3; i < 40; i++)
            {
                double v = System.Math.Pow(0.1, i);
                Product = Gaussian.FromMeanAndVariance(2, v);
                Gaussian result2 = GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
                double diff = result.MaxDiff(result2);
                Console.WriteLine("{0}: {1} diff={2}", Product, result2, diff.ToString("g4"));
                Assert.True(diff <= prevDiff || diff < 1e-6);
                prevDiff = diff;
            }

            Product = Gaussian.Uniform();
            result = GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
            Console.WriteLine("{0}: {1}", Product, result);
            prevDiff = double.PositiveInfinity;
            for (int i = 3; i < 40; i++)
            {
                double v = System.Math.Pow(10, i);
                Product = Gaussian.FromMeanAndVariance(2, v);
                Gaussian result2 = GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
                double diff = result.MaxDiff(result2);
                Console.WriteLine("{0}: {1} diff={2}", Product, result2, diff.ToString("g4"));
                Assert.True(diff <= prevDiff || diff < 1e-6);
                prevDiff = diff;
            }
        }

        [Fact]
        public void LogisticProposalDistribution()
        {
            double[] TrueCounts = { 2, 20, 200, 2000, 20000, 2, 20, 200, 2000, 20000 };
            double[] FalseCounts = { 2, 200, 20000, 200, 20, 200, 2000, 2, 2000, 20 };
            double[] Means = { .5, 1, 2, 4, 8, 16, 32, 0, -2, -20 };
            double[] Variances = { 1, 2, 4, 8, .1, .0001, .01, .000001, 0.000001, 0.001 };
            for (int i = 0; i < 10; i++)
            {
                Beta b = new Beta();
                b.TrueCount = TrueCounts[i];
                b.FalseCount = FalseCounts[i];
                Gaussian g = Gaussian.FromMeanAndVariance(Means[i], Variances[i]);
                Gaussian gProposal = GaussianBetaProductOp.LogisticProposalDistribution(b, g);
            }
        }
    }
}

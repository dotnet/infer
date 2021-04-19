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
    using Assert = AssertHelper;

    public class DiscreteFromDirichletOpTests
    {
        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void BernoulliFromBetaOpTest()
        {
            Assert.True(System.Math.Abs(BernoulliFromBetaOp.LogEvidenceRatio(new Bernoulli(3e-5), Beta.PointMass(1)) - 0) < 1e-10);

            using (TestUtils.TemporarilyAllowBetaImproperSums)
            {
                Beta probTrueDist = new Beta(3, 2);
                Bernoulli sampleDist = new Bernoulli();
                Assert.True(new Beta(1, 1).MaxDiff(BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist, probTrueDist)) < 1e-4);
                sampleDist = Bernoulli.PointMass(true);
                Assert.True(new Beta(2, 1).MaxDiff(BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist, probTrueDist)) < 1e-4);
                sampleDist = Bernoulli.PointMass(false);
                Assert.True(new Beta(1, 2).MaxDiff(BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist, probTrueDist)) < 1e-4);
                sampleDist = new Bernoulli(0.9);
                Assert.True(new Beta(1.724, 0.9598).MaxDiff(BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist, probTrueDist)) < 1e-3);
                Assert.Throws<ImproperMessageException>(() =>
                {
                    BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist, new Beta(1, -2));
                });
            }
        }

        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void BernoulliFromBetaOpTest2()
        {
            Bernoulli sampleDist = new Bernoulli(2.0 / 3);
            using (TestUtils.TemporarilyAllowBetaImproperSums)
            {
                for (int i = 10; i <= 10; i++)
                {
                    double s = System.Math.Exp(i);
                    Beta probTrueDist = new Beta(s, 2 * s);
                    Beta expected = BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist, probTrueDist);
                    if (i == 10)
                        probTrueDist.Point = probTrueDist.GetMean();
                    Beta actual = BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist, probTrueDist);
                    Console.WriteLine("{0} {1}", probTrueDist, actual);
                    Assert.True(expected.MaxDiff(actual) < 1e-4);
                }
            }
            for (int i = 10; i <= 10; i++)
            {
                Beta probTrueDist = new Beta(System.Math.Exp(i), 1);
                Beta expected = BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist, probTrueDist);
                if (i == 10)
                    probTrueDist.Point = 1;
                Beta actual = BernoulliFromBetaOp.ProbTrueAverageConditional(sampleDist, probTrueDist);
                Console.WriteLine("{0} {1}", probTrueDist, actual);
                Assert.True(expected.MaxDiff(actual) < 1e-4);
            }
        }

        [Trait("Category", "ModifiesGlobals")]
        internal void DiscreteFromDirichletOpTest2()
        {
            Discrete sample = new Discrete(1.0 / 3, 2.0 / 3);
            Vector sampleProbs = sample.GetProbs();
            Dirichlet result = Dirichlet.Uniform(2);
            using (TestUtils.TemporarilyAllowDirichletImproperSums)
            {
                for (int i = 1; i <= 10; i++)
                {
                    double s = System.Math.Exp(i);
                    Dirichlet probsDist = new Dirichlet(s, 2 * s);
                    if (i == 10)
                        probsDist.Point = probsDist.GetMean();
                    Console.WriteLine("{0} {1}", probsDist, DiscreteFromDirichletOp.ProbsAverageConditional(sample, probsDist, result));
                    Dirichlet post = probsDist * result;
                    Vector postMean = post.GetMean();
                    Vector priorMean = probsDist.GetMean();
                    Vector postMeanSquare = post.GetMeanSquare();
                    DenseVector delta2 = DenseVector.Zero(postMean.Count);
                    for (int j = 0; j < delta2.Count; j++)
                    {
                        delta2[j] = postMeanSquare[j] - (probsDist.PseudoCount[j] + 1) / (probsDist.TotalCount + 1) * postMean[j];
                    }
                    delta2.Scale(probsDist.TotalCount + 1);
                    Vector delta1 = (postMean - priorMean) * probsDist.TotalCount;
                    DenseVector delta1Approx = DenseVector.Zero(postMean.Count);
                    //delta.SetToFunction(sampleProbs, priorMean, (sampleProb, prob) => (
                    double Z = sampleProbs.Inner(priorMean);
                    delta1Approx[0] = priorMean[0] * priorMean[1] * (sampleProbs[0] - sampleProbs[1]) / Z;
                    DenseVector delta2Approx = DenseVector.Zero(postMean.Count);
                    delta2Approx[0] = priorMean[0] * priorMean[0] * priorMean[1] * (sampleProbs[0] - sampleProbs[1]) / Z;
                    //Console.WriteLine("delta = {0} {1} {2} {3}", delta1, delta1Approx, delta2, delta2Approx);
                }
            }
            for (int i = 1; i <= 10; i++)
            {
                double s = System.Math.Exp(i);
                Dirichlet probsDist = new Dirichlet(s * 4, s * 5);
                if (i == 10)
                    probsDist.Point = probsDist.GetMean();
                Console.WriteLine("{0} {1}", probsDist, DiscreteFromDirichletOp.ProbsAverageConditional(sample, probsDist, result));
            }
        }

        [Trait("Category", "ModifiesGlobals")]
        internal void DiscreteFromDirichletOpTest3()
        {
            Discrete sample = new Discrete(3.0 / 6, 2.0 / 6, 1.0 / 6);
            Vector sampleProbs = sample.GetProbs();
            Dirichlet result = Dirichlet.Uniform(3);
            using (TestUtils.TemporarilyAllowDirichletImproperSums)
            {
                for (int i = 1; i <= 10; i++)
                {
                    double s = System.Math.Exp(i);
                    Dirichlet probsDist = new Dirichlet(s * 2, s * 1, s * 3);
                    if (i == 10)
                        probsDist.Point = probsDist.GetMean();
                    Console.WriteLine("{0} {1}", probsDist, DiscreteFromDirichletOp.ProbsAverageConditional(sample, probsDist, result));
                    Dirichlet post = probsDist * result;
                    Vector postMean = post.GetMean();
                    Vector postMeanSquare = post.GetMeanSquare();
                    Vector priorMean = probsDist.GetMean();
                    Vector delta1 = (postMean - priorMean) * probsDist.TotalCount;
                    DenseVector delta2 = DenseVector.Zero(postMean.Count);
                    for (int j = 0; j < delta2.Count; j++)
                    {
                        delta2[j] = postMeanSquare[j] - (probsDist.PseudoCount[j] + 1) / (probsDist.TotalCount + 1) * postMean[j];
                    }
                    delta2.Scale(probsDist.TotalCount + 1);
                    double Z = sampleProbs.Inner(priorMean);
                    DenseVector delta1Approx = DenseVector.Zero(postMean.Count);
                    //delta.SetToFunction(sampleProbs, priorMean, (sampleProb, prob) => (
                    //delta1Approx[0] = priorMean[0] * priorMean[2] * (sampleProbs[0] - sampleProbs[2])/Z;
                    delta1Approx[0] = probsDist.PseudoCount[1] * postMean[0] - probsDist.PseudoCount[0] * postMean[1] + priorMean[0] * priorMean[2] * (sampleProbs[0] - sampleProbs[2]) / Z;
                    delta1Approx[0] = (probsDist.PseudoCount[1] + probsDist.PseudoCount[2]) / probsDist.PseudoCount[2] * priorMean[0] * priorMean[2] * (sampleProbs[0] - sampleProbs[2]) / Z
                        - probsDist.PseudoCount[0] / probsDist.PseudoCount[2] * priorMean[1] * priorMean[2] * (sampleProbs[1] - sampleProbs[2]) / Z;
                    delta1Approx[1] = probsDist.PseudoCount[0] * postMean[1] - probsDist.PseudoCount[1] * postMean[0] + priorMean[1] * priorMean[2] * (sampleProbs[1] - sampleProbs[2]) / Z;
                    delta1Approx[2] = -priorMean[0] * priorMean[2] * (sampleProbs[0] - sampleProbs[2]) / Z - priorMean[1] * priorMean[2] * (sampleProbs[1] - sampleProbs[2]) / Z;
                    DenseVector delta2Approx = DenseVector.Zero(postMean.Count);
                    delta2Approx[0] = priorMean[0] * priorMean[0] * priorMean[1] * (sampleProbs[0] - sampleProbs[1]) / Z;
                    //Console.WriteLine("delta = {0} {1} {2} {3}", delta1, delta1Approx, delta2, delta2Approx);
                }
            }
            for (int i = 1; i <= 10; i++)
            {
                double s = System.Math.Exp(i);
                Dirichlet probsDist = new Dirichlet(s * 4, s * 5, s * 6);
                if (i == 10)
                    probsDist.Point = probsDist.GetMean();
                Console.WriteLine("{0} {1}", probsDist, DiscreteFromDirichletOp.ProbsAverageConditional(sample, probsDist, result));
            }
        }

        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void DiscreteFromDirichletOpMomentMatchTest()
        {
            using (TestUtils.TemporarilyAllowDirichletImproperSums)
            {
                Dirichlet probsDist = new Dirichlet(1, 2, 3, 4);
                Discrete sampleDist = Discrete.Uniform(4);
                Assert.True(Dirichlet.Uniform(4).MaxDiff(DiscreteFromDirichletOp.ProbsAverageConditional(sampleDist, probsDist, Dirichlet.Uniform(4))) < 1e-4);
                sampleDist = Discrete.PointMass(1, 4);
                Assert.True(new Dirichlet(1, 2, 1, 1).MaxDiff(DiscreteFromDirichletOp.ProbsAverageConditional(sampleDist, probsDist, Dirichlet.Uniform(4))) < 1e-4);
                sampleDist = new Discrete(0, 1, 1, 0);
                Assert.True(new Dirichlet(0.9364, 1.247, 1.371, 0.7456).MaxDiff(DiscreteFromDirichletOp.ProbsAverageConditional(sampleDist, probsDist, Dirichlet.Uniform(4))) <
                              1e-3);
            }
        }

        [Fact]
        public void DiscreteFromDirichletOpTest()
        {
            Dirichlet probs4 = new Dirichlet(1.0, 2, 3, 4);
            Discrete sample4 = new Discrete(0.4, 0.6, 0, 0);
            Dirichlet result4 = DiscreteFromDirichletOp.ProbsAverageConditional(sample4, probs4, Dirichlet.Uniform(4));

            Dirichlet probs3 = new Dirichlet(1.0, 2, 7);
            Discrete sample3 = new Discrete(0.4, 0.6, 0);
            Dirichlet result3 = DiscreteFromDirichletOp.ProbsAverageConditional(sample3, probs3, Dirichlet.Uniform(3));
            for (int i = 0; i < 3; i++)
            {
                Assert.True(MMath.AbsDiff(result4.PseudoCount[i], result3.PseudoCount[i], 1e-6) < 1e-10);
            }

            Dirichlet probs2 = new Dirichlet(1.0, 2);
            Discrete sample2 = new Discrete(0.4, 0.6);
            Dirichlet result2 = DiscreteFromDirichletOp.ProbsAverageConditional(sample2, probs2, Dirichlet.Uniform(2));

            Beta beta = new Beta(1.0, 2);
            Bernoulli bernoulli = new Bernoulli(0.4);
            Beta result1 = BernoulliFromBetaOp.ProbTrueAverageConditional(bernoulli, beta);
            Assert.Equal(result2.PseudoCount[0], result1.TrueCount, 1e-10);
            Assert.Equal(result2.PseudoCount[1], result1.FalseCount, 1e-10);

            // test handling of small alphas
            Discrete sample = DiscreteFromDirichletOp.SampleAverageLogarithm(Dirichlet.Symmetric(2, 1e-8), Discrete.Uniform(2));
            Assert.True(sample.IsUniform());
        }
    }
}

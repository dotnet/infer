// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Factors;
using Xunit;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    
    public class DiscreteTests
    {
        [Fact]
        public void PlusIntTest()
        {
            double p = 0.25;
            Discrete d = new Discrete(p, 1-p);
            d = IntegerPlusOp.SumAverageConditional(d, d, d);
            double Z = p * p + 2 * p * (1 - p);
            Assert.Equal(new Discrete(p*p/Z, 2*p*(1-p)/Z), d);

            d = new Discrete(p, 1 - p);
            d = IntegerPlusOp.SumAverageConditional(1, d, d);
            Assert.Equal(new Discrete(0.0, 1.0), d);

            d = IntegerPlusOp.SumAverageConditional(1, d, d);
            Assert.True(d.IsZero());
        }

        internal void UniqueVisitorCountingInt()
        {
            //  http://en.wikipedia.org/wiki/Unique_visitors
            // Variables describing the population
            double noise = 0.2;
            int maxVisitors = 10;
            Range visitor = new Range(maxVisitors + 1).Named("visitor");
            Variable<int> numVisitors = Variable.DiscreteUniform(visitor).Named("numVisitors");
            VariableArray<int> browserOf = Variable.Array<int>(visitor).Named("browserOf");
            // http://en.wikipedia.org/wiki/Usage_share_of_web_browsers
            double[] browserProbs = {0.66, 0.25, 0.09};
            int numBrowsers = browserProbs.Length;
            Range browser = new Range(numBrowsers);
            browserOf[visitor] = Variable.Discrete(browser, browserProbs).ForEach(visitor);
            Vector[] browserNoiseProbs = new Vector[numBrowsers];
            for (int i = 0; i < browserNoiseProbs.Length; i++)
            {
                double[] noiseProbs = new double[numBrowsers];
                for (int j = 0; j < noiseProbs.Length; j++)
                {
                    noiseProbs[j] = (i == j) ? 1 - noise : noise/(numBrowsers - 1);
                }
                browserNoiseProbs[i] = Vector.FromArray(noiseProbs);
            }
            var browserNoise = Variable.Observed(browserNoiseProbs, browser).Named("browserNoise");
            Variable<int> numObserved = Variable.New<int>().Named("numObserved");
            Range observedVisitor = new Range(numObserved).Named("observedVisitor");
            VariableArray<int> observedAgent = Variable.Array<int>(observedVisitor).Named("observedAgent");
            using (Variable.ForEach(observedVisitor))
            {
                Variable<int> visitorIndex = Variable.DiscreteUniform(numVisitors).Named("visitorIndex");
                using (Variable.Switch(visitorIndex))
                {
                    observedAgent[observedVisitor] = Variable.Discrete(browserNoise[browserOf[visitorIndex]]);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            int numObservedInt = 6;
            int[] data = new int[numObservedInt];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 0;
            }
            numObserved.ObservedValue = data.Length;
            observedAgent.ObservedValue = data;
            Console.WriteLine("browserOf = {0}", engine.Infer(browserOf));
            Console.WriteLine("numVisitors = {0}", engine.Infer(numVisitors));
            //Console.WriteLine("      exact = {0}", BirdCountingExact(maxBirds, numObservedInt, numObservedMale, noise));
        }

        [Fact]
        public void BallCountingTest()
        {
            // Variables describing the population
            int maxBalls = 8;
            Range ball = new Range(maxBalls).Named("ball");
            Variable<int> numBalls = Variable.DiscreteUniform(maxBalls + 1).Named("numBalls");
            VariableArray<bool> isBlue = Variable.Array<bool>(ball).Named("isBlue");
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

            // Variables describing the observations
            Range draw = new Range(10).Named("draw");
            VariableArray<bool> observedBlue = Variable.Array<bool>(draw).Named("observedBlue");
            using (Variable.ForEach(draw))
            {
                // cannot use Variable.DiscreteUniform(numBalls) here since ballIndex will get the wrong range.
                Variable<int> ballIndex = Variable.DiscreteUniform(ball, numBalls).Named("ballIndex");
                using (Variable.Switch(ballIndex))
                {
                    //Variable.ConstrainTrue(isBlue[ballIndex]);
                    observedBlue[draw] = isBlue[ballIndex];
                    Variable.ConstrainTrue(observedBlue[draw]);
                }
            }

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            // 16 iters with good schedule
            // 120 iters with bad schedule
            engine.NumberOfIterations = 150;
            Discrete numBallsActual = engine.Infer<Discrete>(numBalls);
            Console.WriteLine("numBalls = {0}", numBallsActual);
            Discrete numBallsExpected = new Discrete(0, 0.5079, 0.3097, 0.09646, 0.03907, 0.02015, 0.01225, 0.008336, 0.006133);
            Assert.True(numBallsExpected.MaxDiff(numBallsActual) < 1e-4);
            numBalls.ObservedValue = 1;
            Console.WriteLine(engine.Infer(isBlue));
        }

        [Fact]
        public void BallCountingNoisy()
        {
            // Variables describing the population
            int maxBalls = 8;
            Range ball = new Range(maxBalls).Named("ball");
            Variable<int> numBalls = Variable.DiscreteUniform(maxBalls + 1).Named("numBalls");
            VariableArray<bool> isBlue = Variable.Array<bool>(ball).Named("isBlue");
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

            // Variables describing the observations
            Range draw = new Range(10).Named("draw");
            using (Variable.ForEach(draw))
            {
                Variable<int> ballIndex = Variable.DiscreteUniform(ball, numBalls).Named("ballIndex");
                using (Variable.Switch(ballIndex))
                {
                    Variable<bool> switchedColor = Variable.Bernoulli(0.2).Named("switchedColor");
                    Variable.ConstrainTrue(isBlue[ballIndex] != switchedColor);
                }
            }

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            Discrete numUsersActual = engine.Infer<Discrete>(numBalls);
            Console.WriteLine("numUsers = {0}", numUsersActual);
            Discrete numUsersExpected = new Discrete(0, 0.463, 0.2354, 0.1137, 0.06589, 0.04392, 0.0322, 0.02521, 0.02068);
            Assert.True(numUsersExpected.MaxDiff(numUsersActual) < 1e-4);

            // without noise:
            // numUsers = Discrete(0 0.5079 0.3097 0.09646 0.03907 0.02015 0.01225 0.008336 0.006133)
            // with 20% noise:
            // numUsers = Discrete(0 0.463 0.2354 0.1137 0.06589 0.04392 0.0322 0.02521 0.02068)
            // with 50% noise:
            // numUsers = Discrete(0 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125)
            // 0 requests makes it uniform over whole range    
        }

        [Fact]
        public void BallCountingNoisy2()
        {
            // Variables describing the population
            int maxBalls = 8;
            Range ball = new Range(maxBalls).Named("ball");
            Variable<int> numBalls = Variable.DiscreteUniform(maxBalls + 1).Named("numBalls");
            VariableArray<bool> isBlue = Variable.Array<bool>(ball).Named("isBlue");
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

            // Variables describing the observations
            Range draw = new Range(10).Named("draw");
            using (Variable.ForEach(draw))
            {
                Variable<int> ballIndex = Variable.DiscreteUniform(ball, numBalls).Named("ballIndex");
                if (false)
                {
                    using (Variable.Switch(ballIndex))
                    {
                        Variable<bool> switchedColor = Variable.Bernoulli(0.2).Named("switchedColor");
                        using (Variable.If(switchedColor))
                        {
                            Variable.ConstrainFalse(isBlue[ballIndex]);
                        }
                        using (Variable.IfNot(switchedColor))
                        {
                            Variable.ConstrainTrue(isBlue[ballIndex]);
                        }
                    }
                }
                else
                {
                    Variable<bool> switchedColor = Variable.Bernoulli(0.2).Named("switchedColor");
                    using (Variable.If(switchedColor))
                    {
                        using (Variable.Switch(ballIndex))
                            Variable.ConstrainFalse(isBlue[ballIndex]);
                    }
                    using (Variable.IfNot(switchedColor))
                    {
                        using (Variable.Switch(ballIndex))
                            Variable.ConstrainTrue(isBlue[ballIndex]);
                    }
                }
            }


            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            Discrete numBallsActual = engine.Infer<Discrete>(numBalls);
            Console.WriteLine("numBalls = {0}", numBallsActual);
            Discrete numBallsExpected = new Discrete(0, 0.463, 0.2354, 0.1137, 0.06589, 0.04392, 0.0322, 0.02521, 0.02068);
            Assert.True(numBallsExpected.MaxDiff(numBallsActual) < 1e-4);
        }

        [Fact]
        public void BallCountingNoisy3()
        {
            // Variables describing the population
            int maxBalls = 8;
            Range ball = new Range(maxBalls).Named("ball");
            Variable<int> numBalls = Variable.DiscreteUniform(maxBalls + 1).Named("numBalls");
            VariableArray<bool> isBlue = Variable.Array<bool>(ball).Named("isBlue");
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

            // Variables describing the observations
            Range draw = new Range(10).Named("draw");
            VariableArray<bool> observedBlue = Variable.Array<bool>(draw).Named("observedBlue");
            using (Variable.ForEach(draw))
            {
                Variable<int> ballIndex = Variable.DiscreteUniform(ball, numBalls).Named("ballIndex");
                using (Variable.Switch(ballIndex))
                {
                    Variable<bool> switchedColor = Variable.Bernoulli(0.2).Named("switchedColor");
                    using (Variable.If(switchedColor))
                    {
                        Variable.ConstrainEqual(observedBlue[draw], (!isBlue[ballIndex]).Named("!isBlue"));
                    }
                    using (Variable.IfNot(switchedColor))
                    {
                        Variable.ConstrainEqual(observedBlue[draw], isBlue[ballIndex]);
                    }
                }
            }
            bool[] data = {true, true, true, true, true, true, true, true, true, true};
            observedBlue.ObservedValue = data;

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            Discrete numBallsActual = engine.Infer<Discrete>(numBalls);
            Console.WriteLine("numBalls = {0}", numBallsActual);
            Discrete numBallsExpected = new Discrete(0, 0.463, 0.2354, 0.1137, 0.06589, 0.04392, 0.0322, 0.02521, 0.02068);
            Assert.True(numBallsExpected.MaxDiff(numBallsActual) < 1e-4);
        }

        [Fact]
        public void BallCountingNoisy4()
        {
            // Variables describing the population
            int maxBalls = 8;
            Range ball = new Range(maxBalls).Named("ball");
            Variable<int> numBalls = Variable.DiscreteUniform(maxBalls + 1).Named("numBalls");
            VariableArray<bool> isBlue = Variable.Array<bool>(ball).Named("isBlue");
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

            // Variables describing the observations
            Range draw = new Range(10).Named("draw");
            VariableArray<bool> observedBlue = Variable.Array<bool>(draw).Named("observedBlue");
            using (Variable.ForEach(draw))
            {
                Variable<int> ballIndex = Variable.DiscreteUniform(ball, numBalls).Named("ballIndex");
                if (true)
                {
                    using (Variable.Switch(ballIndex))
                    {
                        Variable<bool> switchedColor = Variable.Bernoulli(0.2).Named("switchedColor");
                        using (Variable.If(switchedColor))
                        {
                            observedBlue[draw] = !isBlue[ballIndex];
                        }
                        using (Variable.IfNot(switchedColor))
                        {
                            observedBlue[draw] = isBlue[ballIndex];
                        }
                    }
                }
                else
                {
                    // an equivalent model that gives the same results
                    Variable<bool> switchedColor = Variable.Bernoulli(0.2).Named("switchedColor");
                    using (Variable.If(switchedColor))
                    {
                        using (Variable.Switch(ballIndex))
                            observedBlue[draw] = !isBlue[ballIndex];
                    }
                    using (Variable.IfNot(switchedColor))
                    {
                        using (Variable.Switch(ballIndex))
                            observedBlue[draw] = isBlue[ballIndex];
                    }
                }
            }
            bool[] data = {true, true, true, true, true, true, true, true, true, true};
            observedBlue.ObservedValue = data;

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            Discrete numUsersActual = engine.Infer<Discrete>(numBalls);
            Console.WriteLine("numUsers = {0}", numUsersActual);
            Discrete numUsersExpected = new Discrete(0, 0.463, 0.2354, 0.1137, 0.06589, 0.04392, 0.0322, 0.02521, 0.02068);
            Assert.True(numUsersExpected.MaxDiff(numUsersActual) < 1e-4);
        }

        [Fact]
        public void BallCountingWithData()
        {
            // Variables describing the population
            int maxBalls = 8;
            Range ball = new Range(maxBalls).Named("ball");
            Variable<int> numBalls = Variable.DiscreteUniform(maxBalls + 1).Named("numBalls");
            VariableArray<bool> isBlue = Variable.Array<bool>(ball).Named("isBlue");
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

            // Variables describing the observations
            Range draw = new Range(10).Named("draw");
            VariableArray<bool> observedBlue = Variable.Array<bool>(draw).Named("observedBlue");
            using (Variable.ForEach(draw))
            {
                Variable<int> ballIndex = Variable.DiscreteUniform(ball, numBalls).Named("ballIndex");
                using (Variable.Switch(ballIndex))
                {
                    observedBlue[draw] = isBlue[ballIndex];
                }
            }

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 150;
            bool[] data = new bool[10];
            for (int i = 0; i < 10; i++)
            {
                data[i] = true;
            }
            observedBlue.ObservedValue = data;
            Discrete numBallsActual = engine.Infer<Discrete>(numBalls);
            Console.WriteLine("numBalls = {0}", numBallsActual);
            Discrete numBallsExpected = new Discrete(0, 0.5079, 0.3097, 0.09646, 0.03907, 0.02015, 0.01225, 0.008336, 0.006133);
            Console.WriteLine(" should be {0}", numBallsExpected);
            Console.WriteLine("   exact = {0}", BallCountingExact(maxBalls, data.Length, 10, 0.0));
            Assert.True(numBallsExpected.MaxDiff(numBallsActual) < 1e-4);
        }

        [Fact]
        public void BallCountingNoisyWithData()
        {
            // Variables describing the population
            int maxBalls = 8;
            Range ball = new Range(maxBalls).Named("ball");
            Variable<int> numBalls = Variable.DiscreteUniform(maxBalls + 1).Named("numBalls");
            VariableArray<bool> isBlue = Variable.Array<bool>(ball).Named("isBlue");
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

            // Variables describing the observations
            Range draw = new Range(10).Named("draw");
            VariableArray<bool> observedBlue = Variable.Array<bool>(draw).Named("observedBlue");
            using (Variable.ForEach(draw))
            {
                Variable<int> ballIndex = Variable.DiscreteUniform(ball, numBalls).Named("ballIndex");
                using (Variable.Switch(ballIndex))
                {
                    Variable<bool> switchedBrowser = Variable.Bernoulli(0.2);
                    observedBlue[draw] = (isBlue[ballIndex] != switchedBrowser);
                }
            }

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            bool[] data = new bool[10];
            for (int i = 0; i < 5; i++)
            {
                data[i] = true;
            }
            observedBlue.ObservedValue = data;
            Discrete numBallsActual = engine.Infer<Discrete>(numBalls);
            Console.WriteLine("numBalls = {0}", numBallsActual);
            Discrete numBallsExpected = new Discrete(0, 0.08198, 0.09729, 0.1102, 0.1217, 0.1324, 0.1425, 0.1523, 0.1617);
            Assert.True(numBallsExpected.MaxDiff(numBallsActual) < 1e-4);
            Console.WriteLine("   exact = {0}", BallCountingExact(maxBalls, data.Length, 5, 0.2));
        }

        internal void BallCounting()
        {
            /* The colored ball example, example 1.1 from

 http://people.csail.mit.edu/milch/papers/blog-chapter.pdf

``An urn contains an unknown number of balls--say, a number chosen from
a Poisson or a uniform distributions. Balls are equally likely to be blue or
green. We draw some balls from the urn, observing the color of each
and replacing it. We cannot tell two identically colored balls apart;
furthermore, observed colors are wrong with probability 0.2.  How many
balls are in the urn? Was the same ball drawn twice?''
             * 
             * http://okmij.org/ftp/kakuritu/blip/colored_balls.ml
             */
            double[] exactProbs = new double[]
                {
                    0,
                    0.416368552153129956,
                    0.211354336837841883,
                    0.123751456856916756,
                    0.0777907061915841,
                    0.055953476873295431,
                    0.0445344709122082669,
                    0.0384083436381660176,
                    0.0318386565368576138,
                };

            double noise = 0.2;
            int maxBalls = 8;
            Range ball = new Range(maxBalls).Named("ball");
            Variable<int> numBalls = Variable.DiscreteUniform(maxBalls + 1).Named("numBalls");
            VariableArray<bool> isBlue = Variable.Array<bool>(ball).Named("isBlue");
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);
            Variable<int> numObserved = Variable.New<int>().Named("numObserved");
            Range observedBall = new Range(numObserved).Named("observedBall");
            VariableArray<bool> observedBlue = Variable.Array<bool>(observedBall).Named("observedBlue");
            using (Variable.ForEach(observedBall))
            {
                Variable<int> ballIndex = Variable.DiscreteUniform(ball, numBalls).Named("ballIndex");
                using (Variable.Switch(ballIndex))
                {
                    observedBlue[observedBall] = (isBlue[ballIndex] == Variable.Bernoulli(1 - noise));
                }
            }

            InferenceEngine engine = new InferenceEngine();
            int numObservedInt = 10;
            // weird behavior with 1 different out of >=9 obs, no obs noise
            bool[] data = new bool[numObservedInt];
            int numObservedBlue = numObservedInt;
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (i < numObservedBlue);
            }
            numObserved.ObservedValue = data.Length;
            observedBlue.ObservedValue = data;
            Console.WriteLine("numBalls = {0}", engine.Infer(numBalls));
            Console.WriteLine("   exact = {0}", BallCountingExact(maxBalls, numObservedInt, numObservedBlue, noise));
        }

        internal void BirdCounting3()
        {
            double noise = 0.2;
            int maxBirds = 8;
            Range numBirdRange = new Range(maxBirds + 1).Named("numBirdRange");
            Variable<int> numBirds = Variable.DiscreteUniform(numBirdRange).Named("numBirds");
            SwitchBlock block = Variable.Switch(numBirds);
            Range bird = new Range(maxBirds).Named("bird");
            VariableArray<bool> isMale = Variable.Array<bool>(bird).Named("isMale");
            isMale[bird] = Variable.Bernoulli(0.5).ForEach(bird);
            Variable<int> numObserved = Variable.New<int>().Named("numObserved");
            Range observedBird = new Range(numObserved).Named("observedBird");
            VariableArray<bool> observedMale = Variable.Array<bool>(observedBird).Named("observedMale");
            //VariableArray<int> birdIndices = Variable.Array<int>(observedBird).Named("birdIndices");
            using (Variable.ForEach(observedBird))
            {
                //birdIndices[observedBird] = Variable.DiscreteUniform(numBirds);
                //Variable<int> birdIndex = birdIndices[observedBird];
                Variable<int> birdIndex = Variable.DiscreteUniform(bird, numBirds).Named("birdIndex");
                using (Variable.Switch(birdIndex))
                {
#if true
                    //Variable.ConstrainEqual(observedMale[observedBird], isMale[birdIndex]);
                    observedMale[observedBird] = (isMale[birdIndex] == Variable.Bernoulli(1 - noise));
#else
                    using (Variable.If(isMale[birdIndex])) {
                        observedMale[observedBird] = Variable.Bernoulli(0.8);
                    }
                    using (Variable.IfNot(isMale[birdIndex])) {
                        observedMale[observedBird] = Variable.Bernoulli(0.2);
                    }
#endif
                }
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int numObservedInt = 6; numObservedInt <= 10; numObservedInt++)
            {
                Console.WriteLine("numObserved = {0}", numObservedInt);
                // weird behavior with 1 different out of >=9 obs, no obs noise
                bool[] data = new bool[numObservedInt];
                int numObservedMale = 0;
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = (i < numObservedMale);
                }
                numObserved.ObservedValue = data.Length;
                observedMale.ObservedValue = data;
                //Console.WriteLine("birdIndices = {0}", engine.Infer(birdIndices));
                Console.WriteLine("isMale = {0}", engine.Infer(isMale));
                Console.WriteLine("numBirds = {0}", engine.Infer(numBirds));
                Console.WriteLine("   exact = {0}", BallCountingExact(maxBirds, numObservedInt, numObservedMale, noise));
            }
        }

        [Fact]
        public void BallCounting2()
        {
            double noise = 0.2;
            int maxBalls = 8;
            Range ball = new Range(maxBalls).Named("ball");
            Variable<int> numBalls = Variable.DiscreteUniform(maxBalls + 1).Named("numBalls");
            numBalls.SetValueRange(new Range(maxBalls + 1));
            SwitchBlock block = Variable.Switch(numBalls);
            Variable<int> numTrue = Variable.Binomial(numBalls, 0.5).Named("numTrue");
            Variable<int> numObserved = Variable.New<int>().Named("numObserved");
            Range observedBall = new Range(numObserved).Named("observedBall");
            VariableArray<bool> observedTrue = Variable.Array<bool>(observedBall).Named("observedTrue");
            using (Variable.ForEach(observedBall))
            {
                Variable<int> ballIndex = Variable.DiscreteUniform(ball, numBalls).Named("ballIndex");
                observedTrue[observedBall] = ((ballIndex < numTrue) == Variable.Bernoulli(1 - noise));
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            for (int numObservedInt = 10; numObservedInt <= 10; numObservedInt++)
            {
                Console.WriteLine("numObserved = {0}", numObservedInt);
                // weird behavior with 1 different out of >=9 obs, no obs noise
                bool[] data = new bool[numObservedInt];
                int numObservedTrue = 0;
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = (i < numObservedTrue);
                }
                numObserved.ObservedValue = data.Length;
                observedTrue.ObservedValue = data;
                //Console.WriteLine("numTrue = {0}", engine.Infer(numTrue));
                Discrete numBallsActual = engine.Infer<Discrete>(numBalls);
                Console.WriteLine("numBalls = {0}", numBallsActual);
                Discrete numBallsExpected = BallCountingExact(maxBalls, numObservedInt, numObservedTrue, noise);
                Console.WriteLine("   exact = {0}", numBallsExpected);
                Assert.True(numBallsExpected.MaxDiff(numBallsActual) < 1e-10);
            }
        }

        /// <summary>
        /// Returns the exact distribution over the number of balls in an urn, given observations with replacement.
        /// </summary>
        /// <param name="maxBalls"></param>
        /// <param name="numObserved">Number of observations</param>
        /// <param name="numObservedTrue">Number of observations which were "true"</param>
        /// <param name="noise">Probability of the observation being flipped</param>
        /// <returns>The exact distribution over the number of balls (0,...,maxBalls)</returns>
        public Discrete BallCountingExact(int maxBalls, int numObserved, int numObservedTrue, double noise)
        {
            int numObservedFalse = numObserved - numObservedTrue;
            double[] probSize = new double[maxBalls + 1];
            for (int size = 0; size <= maxBalls; size++)
            {
                double prob = 0.0;
                for (int numTrue = 0; numTrue <= size; numTrue++)
                {
                    if (size == 0) continue;
                    // numTrue ~ Binomial(size, 0.5)
                    double weight = MMath.ChooseLn(size, numTrue) + size* System.Math.Log(0.5);
                    int numFalse = size - numTrue;
                    double probTrue = (double) numTrue/size;
                    probTrue = (1 - noise)*probTrue + noise*(1 - probTrue);
                    if ((numObservedTrue > 0 && probTrue == 0.0) || (numObservedFalse > 0 && probTrue == 1.0)) continue;
                    // numObservedTrue ~ Binomial(numObserved, probTrue)
                    prob += System.Math.Exp(weight //+ MMath.ChooseLn(numObserved, numObservedTrue) 
                                     + numObservedTrue*(numObservedTrue == 0 ? 0.0 : System.Math.Log(probTrue))
                                     + numObservedFalse*(numObservedFalse == 0 ? 0.0 : System.Math.Log(1 - probTrue)));
                }
                probSize[size] = prob;
            }
            return new Discrete(probSize);
        }

        [Fact]
        public void EulerProject205()
        {
            // Peter has nine four-sided (pyramidal) dice, each with faces numbered 1, 2, 3, 4.
            // Colin has six six-sided (cubic) dice, each with faces numbered 1, 2, 3, 4, 5, 6.
            //
            // Peter and Colin roll their dice and compare totals: the highest total wins. 
            // The result is a draw if the totals are equal.
            //
            // What is the probability that Pyramidal Pete beats Cubic Colin? 
            // http://projecteuler.net/index.php?section=problems&id=205

            // We encode this problem by using variables sum[0] for Peter and sum[1] for Colin.
            // Ideally, we want to add up dice variables that range from 1-4 and 1-6.
            // However, integers in Infer.NET must start from 0.  
            // One approach is to use dice that range from 0-4 and 0-6, with zero probability on the value 0.
            // Another approach is to use dice that range 0-3 and 0-5, and add an offset at the end.
            // We use the second approach here.

            int[] numSides = new int[2] {4, 6};
            Vector[] probs = new Vector[2];
            for (int i = 0; i < 2; i++)
                probs[i] = Vector.Constant(numSides[i], 1.0/numSides[i]);
            int[] numDice = new int[] {9, 6};
            Variable<int>[] sum = new Variable<int>[2];
            for (int i = 0; i < 2; i++)
            {
                sum[i] = Variable.Discrete(probs[i]).Named("die" + i + "0");
                for (int j = 1; j < numDice[i]; j++)
                {
                    sum[i] += Variable.Discrete(probs[i]).Named("die" + i + j);
                }
            }
            for (int i = 0; i < 2; i++)
            {
                // add an offset due to zero-based integers
                sum[i] += numDice[i];
            }
            sum[0].Name = "sum0";
            sum[1].Name = "sum1";
            Variable<bool> win = sum[0] > sum[1];
            win.Name = "win";
            InferenceEngine engine = new InferenceEngine();
            engine.NumberOfIterations = 1; // Only one iteration needed
            Bernoulli winMarginal = engine.Infer<Bernoulli>(win);
            Console.WriteLine("{0:0.0000000}", winMarginal.GetProbTrue());
            Bernoulli winExpected = new Bernoulli(0.5731441);
            Assert.True(winExpected.MaxDiff(winMarginal) < 1e-7);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TestBossPredictor()
        {
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            //compiler.BrowserMode = ModelCompiler.BrowserModeOptions.ALWAYS;
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(BossPredictorModel, true);
            ca.Execute(10);
            Console.WriteLine("Probability of approving trip (raining)=" + ca.Marginal("approvesTrip"));
            Bernoulli expected = new Bernoulli(0.71);
            Bernoulli actual = ca.Marginal<Bernoulli>("approvesTrip");
            Assert.True(expected.MaxDiff(actual) < 1e-10);
            ca.SetObservedValue("isRaining", false);
            ca.Execute(10);
            Console.WriteLine("Probability of approving trip (not raining)=" + ca.Marginal("approvesTrip"));
            expected = new Bernoulli(0.85);
            actual = ca.Marginal<Bernoulli>("approvesTrip");
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }


        private void BossPredictorModel(bool isRaining)
        {
            bool raining = Factor.Random(new Bernoulli(0.8));
            bool coffee = Factor.Random(new Bernoulli(0.6));
            bool notRaining = Factor.Not(raining);
            bool temp = Factor.Or(coffee, notRaining);

            bool goodMood;
            bool approvesTrip;
            if (temp)
                goodMood = Factor.Bernoulli(0.9);
            else
                goodMood = Factor.Bernoulli(0.2);

            if (goodMood)
                approvesTrip = Factor.Bernoulli(0.9);
            else
                approvesTrip = Factor.Bernoulli(0.4);

            Constrain.Equal(raining, isRaining);
            InferNet.Infer(approvesTrip, nameof(approvesTrip));
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
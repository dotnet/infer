using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Core.Maths
{
    /// <summary>
    /// Represents a truncated power series (effectively, a polynomial) of the form
    /// c0 + c1*x + c2*x^2 + c3*x^3 + ... + cN*x^N
    /// </summary>
    struct TruncatedPowerSeries
    {
        /// <summary>
        /// Coefficients of the truncated power series.
        /// Coefficients[k] is the coefficient near x^k, so Coefficients[0] is the constant term.
        /// </summary>
        public double[] Coefficients { get; }

        /// <summary>
        /// Constructs a truncated power series.
        /// </summary>
        /// <param name="coefficients">
        /// Coefficients of the truncated power series.
        /// <paramref name="coefficients"/>[k] is the coefficient near x^k, so <paramref name="coefficients"/>[0] is the constant term.
        /// </param>
        public TruncatedPowerSeries(double[] coefficients)
        {
            this.Coefficients = coefficients;
        }

        /// <summary>
        /// Factory function to generate a truncated power series.
        /// All coefficients of the generated series are computed, when this function is called.
        /// </summary>
        /// <param name="length">
        /// Number of coefficients including the constant term to generate.
        /// The highest power of x in the resulting series will be (<paramref name="length"/> - 1)
        /// </param>
        /// <param name="coefGenerator">
        /// Provides the coefficients in the series, where the constant term is index zero.
        /// Coefficients[k] = <paramref name="coefGenerator"/>(k)
        /// </param>
        /// <returns>Generated truncated power series.</returns>
        public static TruncatedPowerSeries Generate(int length, Func<int, double> coefGenerator)
        {
            double[] coefficients = new double[length];
            for (int i = 0; i < length; ++i)
                coefficients[i] = coefGenerator(i);

            return new TruncatedPowerSeries(coefficients);
        }

        /// <summary>
        /// Factory function to generate a truncated power series.
        /// All coefficients of the generated series are computed, when this function is called.
        /// Coefficients with smaller indexes are used to compute coefficients with bigger indexes.
        /// </summary>
        /// <param name="length">
        /// Number of coefficients including the constant term to generate.
        /// The highest power of x in the resulting series will be (<paramref name="length"/> - 1)
        /// </param>
        /// <param name="coefGenerator">
        /// Provides the coefficients in the series, where the constant term is index zero.
        /// The second argument is the coefficients with indexes smaller than that of the coefficient being computed.
        /// Coefficients[k] = <paramref name="coefGenerator"/>(k, Coefficients[0..k-1])
        /// </param>
        /// <returns>Generated truncated power series.</returns>
        public static TruncatedPowerSeries Generate(int length, Func<int, IList<double>, double> coefGenerator)
        {
            List<double> coefficients = new List<double>(length);
            for (int i = 0; i < length; ++i)
                coefficients.Add(coefGenerator(i, coefficients));

            return new TruncatedPowerSeries(coefficients.ToArray());
        }

        public double Evaluate(double x)
        {
            double sum = 0.0;
            for (int k = Coefficients.Length - 1; k > 0; k--)
            {
                sum = x * (Coefficients[k] + sum);
            }
            return sum + Coefficients[0];
        }
    }

    /// <summary>
    /// Represents a mathematical power series of the form c0 + c1*x + c2*x^2 + c3*x^3 + ...
    /// </summary>
    struct PowerSeries
    {
        readonly Func<int, double> coefGenerator;

        /// <summary>
        /// Constructs a power series.
        /// </summary>
        /// <param name="coefGenerator">Provides the coefficients in the series, where the constant term is index zero. 
        /// Every zero coefficient must eventually be followed by a nonzero coefficient.</param>
        public PowerSeries(Func<int, double> coefGenerator)
        {
            this.coefGenerator = coefGenerator;
        }

        public double Evaluate(double x)
        {
            double sum = coefGenerator(0);
            double term = 1;
            for (int i = 1; ; i++)
            {
                double oldSum = sum;
                term *= x;
                double coefficient = coefGenerator(i);
                // To ensure termination, every zero coefficient must eventually be followed by a nonzero coefficient.
                if (coefficient != 0)
                {
                    sum += term * coefficient;
                    if (MMath.AreEqual(sum, oldSum)) break;
                }
            }
            return sum;
        }
    }

    /// <summary>
    /// Represents the state of a series coefficient generation attempt.
    /// Supposed to be passed by reference to coefficient generator.
    /// </summary>
    struct CoefficientGenerationState<InnerState>
    {
        /// <summary>
        /// Index of the next coefficient to be computed.
        /// Set by the caller (series evaluator).
        /// </summary>
        public int nextIndex;
        /// <summary>
        /// true, if the coefficient was successfully generated,
        /// false, if the series was finite and ended somewhere before the requested coefficient.
        /// Set by the callee (coefficient generator).
        /// </summary>
        public bool success;
        /// <summary>
        /// value of the computed coefficient, if success == true,
        /// undefined otherwise
        /// Set by the callee (coefficient generator).
        /// </summary>
        public double coefficient;
        /// <summary>
        /// state that can be used to generate the next coefficient, if success == true,
        /// undefined otherwise
        /// Initialized to default(InnerState) by the caller (series evaluator),
        /// afterwards set by the callee (coefficient generator).
        /// </summary>
        public InnerState innerState;
    }

    /// <summary>
    /// Provides the coefficients in the series, where the constant term is index zero.
    /// A state is expected to be passed between the computations of consequitive coefficients
    /// by the caller.
    /// When the constant term (index zero) is computed, default(State) is passed to the delegate,
    /// for a coefficient at index = n, a state obtained when computing the coefficient
    /// at index = n - 1 is passed.
    /// </summary>
    /// <typeparam name="State">A type that represents the state passed between
    /// the computations of consequitive coefficients</typeparam>
    /// <param name="state">State passed between
    /// the computations of consequitive coefficients</param>
    /// <remarks>
    /// For small coefficient generation functions, like Series.ExpMinus1Coefficient,
    /// manually calling this delegate is about 15-30% faster that using IEnumerable.
    /// </remarks>
    delegate void CoefficientGenerator<State>(ref CoefficientGenerationState<State> state);

    /// <summary>
    /// Represents a mathematical power series of the form c0 + c1*x + c2*x^2 + c3*x^3 + ...
    /// Allows stateful computation of the coefficients.
    /// </summary>
    struct PowerSeries<State>
    {
        readonly CoefficientGenerator<State> coefGenerator;

        /// <summary>
        /// Constructs a power series.
        /// </summary>
        /// <param name="coefGenerator">
        /// Provides the coefficients in the series, where the constant term is index zero.
        /// A state is expected to be passed between the computations of consequitive coefficients
        /// by the caller.
        /// </param>
        public PowerSeries(CoefficientGenerator<State> coefGenerator)
        {
            this.coefGenerator = coefGenerator;
        }

        public double Evaluate(double x)
        {
            CoefficientGenerationState<State> state;
            state.nextIndex = 0;
            state.success = true;
            state.coefficient = 0.0;
            state.innerState = default(State);
            double sum = 0.0;
            double term = 1.0;
            double oldSum = double.NaN;
            coefGenerator(ref state);
            while (state.success)
            {
                oldSum = sum;
                if (state.coefficient != 0)
                {
                    sum += term * state.coefficient;
                    if (MMath.AreEqual(sum, oldSum)) break;
                }
                term *= x;
                ++state.nextIndex;
                coefGenerator(ref state);
            }
            return sum;
        }
    }

    class Series
    {
        public TruncatedPowerSeries GammaAt2 { get; }
        public TruncatedPowerSeries DigammaAt2 { get; }
        /// <summary>
        /// Coefficients of de Moivre's expansion for the digamma function.
        /// </summary>
        public TruncatedPowerSeries DigammaAsymptotic { get; }
        public TruncatedPowerSeries TrigammaAt1 { get; }
        /// <summary>
        /// Coefficients of de Moivre's expansion for the trigamma function.
        /// </summary>
        public TruncatedPowerSeries TrigammaAsymptotic { get; }
        public TruncatedPowerSeries TetragammaAt1 { get; }
        /// <summary>
        /// Coefficients of de Moivre's expansion for the tetragamma function.
        /// </summary>
        public TruncatedPowerSeries TetragammaAsymptotic { get; }
        public TruncatedPowerSeries GammalnAsymptotic { get; }
        public PowerSeries Log1Plus { get; }
        public PowerSeries Log1Minus { get; }
        public PowerSeries<double> ExpMinus1 { get; }
        public PowerSeries<double> ExpMinus1RatioMinus1RatioMinusHalf { get; }
        public TruncatedPowerSeries LogExpMinus1RatioAt0 { get; }
        /// <summary>
        /// Asymptotic expansion of NormalCdfLn
        /// </summary>
        public TruncatedPowerSeries NormcdflnAsymptotic { get; }

        public Series(uint precisionBits)
        {
            uint doublePrecisionCutOff = 53;
            if (precisionBits <= doublePrecisionCutOff)
            {
                GammaAt2 = new TruncatedPowerSeries(gammaAt2Coefficients.Take(26).ToArray());
                DigammaAsymptotic = new TruncatedPowerSeries(digammaAsymptoticCoefficients.Take(8).ToArray());
                TrigammaAt1 = new TruncatedPowerSeries(trigammaAt1Coefficients.Take(2).ToArray());
                TrigammaAsymptotic = new TruncatedPowerSeries(trigammaAsymptoticCoefficients.Take(9).ToArray());
                TetragammaAt1 = new TruncatedPowerSeries(tetragammaAt1Coefficients.Take(2).ToArray());
                TetragammaAsymptotic = new TruncatedPowerSeries(tetragammaAsymptoticCoefficients.Take(9).ToArray());
                GammalnAsymptotic = new TruncatedPowerSeries(gammalnAsymptoticCoefficients.Take(7).ToArray());
                LogExpMinus1RatioAt0 = new TruncatedPowerSeries(logExpMinus1RatioAt0Coefficients.Take(5).ToArray());
                NormcdflnAsymptotic = new TruncatedPowerSeries(normcdflnAsymptoticCoefficients.Take(8).ToArray());
            }
            else
            {
                GammaAt2 = new TruncatedPowerSeries(gammaAt2Coefficients.ToArray());
                DigammaAsymptotic = new TruncatedPowerSeries(digammaAsymptoticCoefficients.ToArray());
                TrigammaAt1 = new TruncatedPowerSeries(trigammaAt1Coefficients.ToArray());
                TrigammaAsymptotic = new TruncatedPowerSeries(trigammaAsymptoticCoefficients.ToArray());
                TetragammaAt1 = new TruncatedPowerSeries(tetragammaAt1Coefficients.ToArray());
                TetragammaAsymptotic = new TruncatedPowerSeries(tetragammaAsymptoticCoefficients.ToArray());
                GammalnAsymptotic = new TruncatedPowerSeries(gammalnAsymptoticCoefficients.ToArray());
                LogExpMinus1RatioAt0 = new TruncatedPowerSeries(logExpMinus1RatioAt0Coefficients.ToArray());
                NormcdflnAsymptotic = new TruncatedPowerSeries(normcdflnAsymptoticCoefficients.ToArray());
            }
            DigammaAt2 = TruncatedPowerSeries.Generate(GammaAt2.Coefficients.Length, i => GammaAt2.Coefficients[i] * (i + 1));
            Log1Plus = new PowerSeries(Log1PlusCoefficient);
            Log1Minus = new PowerSeries(Log1MinusCoefficient);
            ExpMinus1 = new PowerSeries<double>(ExpMinus1Coefficient);
            ExpMinus1RatioMinus1RatioMinusHalf = new PowerSeries<double>(ExpMinus1RatioMinus1RatioMinusHalfCoefficient);
        }

        #region Coefficient generators

        private static double Log1PlusCoefficient(int n) => n == 0 ? 0.0 : (n % 2 == 0 ? -1.0 : 1.0) / n;
        private static double Log1MinusCoefficient(int n) => n == 0 ? 0.0 : -1.0 / n;
        public static void ExpMinus1Coefficient(ref CoefficientGenerationState<double> state)
        {
            if (state.nextIndex == 0)
            {
                state.success = true;
                state.coefficient = 0.0;
                state.innerState = 1.0;
            }
            else
            {
                state.coefficient = state.innerState / state.nextIndex;
                state.innerState = state.coefficient;
                state.success = state.coefficient != 0.0;
            }
        }
        public static void ExpMinus1RatioMinus1RatioMinusHalfCoefficient(ref CoefficientGenerationState<double> state)
        {
            if (state.nextIndex == 0)
            {
                state.success = true;
                state.coefficient = 0.0;
                state.innerState = 0.5;
            }
            else
            {
                state.coefficient = state.innerState / (state.nextIndex + 2);
                state.innerState = state.coefficient;
                state.success = state.coefficient != 0.0;
            }
        }

        #endregion

        #region Precomputed coefficients

        /* Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
print("            0.0,")
for k in range(2,110):
		print("            %0.34g," % ((-1)**k*(zeta(k)-1)/k).evalf(34))
         */
        private static readonly double[] gammaAt2Coefficients = new[]
        {
            0,
            0.3224670334241132030328458313306328,
            -0.06735230105319810200992236559613957,
            0.02058080842778454641606167285772244,
            -0.00738555102867398567678680620929299,
            0.002890510330741523359332489917505882,
            -0.001192753911703261018861788045342109,
            0.000509669524743042450139196564151689,
            -0.0002231547584535793857934971029521876,
            9.945751278180853098033475934158787e-05,
            -4.492623673813314204598598489148742e-05,
            2.050721277567069106674904621634425e-05,
            -9.439488275268396715186013101739348e-06,
            4.37486678990748817440030460357292e-06,
            -2.039215753801366189692763700169742e-06,
            9.551412130407419353005918014054565e-07,
            -4.492469198764566185487696111516831e-07,
            2.120718480555466464491199763200968e-07,
            -1.004322482396809908404802422730862e-07,
            4.769810169363980398286696825264908e-08,
            -2.271109460894316350425416834932224e-08,
            1.083865921489695459289913246009665e-08,
            -5.183475041970046644229946111422438e-09,
            2.483674543802478475235497043321041e-09,
            -1.192140140586091154743934012838306e-09,
            5.731367241678862251438860640198256e-10,
            -2.759522885124233355903938000146247e-10,
            1.330476437424448882003567968775656e-10,
            -6.422964563838099598864779601491062e-11,
            3.104424774732227563392725117909481e-11,
            -1.502138408075414166458594899693007e-11,
            7.275974480239079174553105387739685e-12,
            -3.527742476575915065189557262694141e-12,
            1.711991790559617978544962723971016e-12,
            -8.315385841420284979314085891293852e-13,
            4.042200525289440192260810475387441e-13,
            -1.966475631096616531130082285066565e-13,
            9.573630387838555566047107836916075e-14,
            -4.664076026428374443204720937282992e-14,
            2.273736960065972417354758289435552e-14,
            -1.109139947083452217591392010101267e-14,
            5.413659156725363290762788546531038e-15,
            -2.643880017860994856070349482131646e-15,
            1.291895906278996649797589034450644e-15,
            -6.315935504198448152815144632969046e-16,
            3.089316266963393006209215246728354e-16,
            -1.511793062810819824159605773901573e-16,
            7.401486856952319873829631842748242e-17,
            -3.625218048120653818830185998027823e-17,
            1.776356842186163324845409513060544e-17,
            -8.707631574791790545720149092818225e-18,
            4.270088559227003785123993337350986e-18,
            -2.094760424794464294026626618224464e-18,
            1.027984282378792796690424930004006e-18,
            -5.046468294792953328756217103439225e-19,
            2.478176394593791693371463023660605e-19,
            -1.217349807814763706484975874824611e-19,
            5.981805089941246261735805969233367e-20,
            -2.940209281436570304360268231936827e-20,
            1.445602896686655580463396514538343e-20,
            -7.109522442656804701132425037724369e-21,
            3.497426362898741485992060556339303e-21,
            -1.72095582935593863487067784712846e-21,
            8.470329472588509127992257400804905e-22,
            -4.170008355728413714644388723288791e-22,
            2.053413205469873327785584183708884e-22,
            -1.011382623588834223218438654968751e-22,
            4.982546748559995181315757428368296e-23,
            -2.455167963057679834508841583052901e-23,
            1.210047067506714006249752230839652e-23,
            -5.965020755313850001484134344489508e-24,
            2.94108662241138167013854591546344e-24,
            -1.450398882284963546831250522314545e-24,
            7.153994486945770419577690483337377e-25,
            -3.529303946893137333637237751583067e-25,
            1.741432868532761836030085313783932e-25,
            -8.594084286265459638986861745574907e-26,
            4.241951859246373681570791434569361e-26,
            -2.094128133045665479257451510604071e-26,
            1.033975765691293117071248028646776e-26,
            -5.106053163907605937631813496430958e-27,
            2.521892111442166601145541317372597e-27,
            -1.245753934567815698050968063924673e-27,
            6.154617652924322580570211265420193e-28,
            -3.041105193209663666084926529947082e-28,
            1.502871752458263590864301387762562e-28,
            -7.42798682249486384616203701175441e-29,
            3.671788940665074353595069096429321e-29,
            -1.815266442575991720790924205550424e-29,
            8.975484077181291618744867280323167e-30,
            -4.438426192012726108791433766298579e-30,
            2.195091214528033139239927553086061e-30,
            -1.085744041594510958603950819611772e-30,
            5.370967865334548331937709775945041e-31,
            -2.657215680744460758305032942851813e-31,
            1.314768175368352936035252591887088e-31,
            -6.506069321410406518422414169431261e-32,
            3.2198404294735178635568008439894e-32,
            -1.593658394385882570638484064799577e-32,
            7.888609052210118218331949440892363e-33,
            -3.905252006044612945499590526006017e-33,
            1.93348261083581322172569684496361e-33,
            -9.57355467501227836151640760828371e-34,
            4.740750632337811593763855621594469e-34,
            -2.347800313157773442176833948742271e-34,
            1.162825626799840470674685835501613e-34,
            -5.759790487887060065479999327489828e-35,
            2.853229547240349253548725242810521e-35,
            -1.413526564687879603645186309191011e-35,
        };

        /* Coefficients of de Moivre's expansion for the digamma function.
         Each coefficient is B_{2j}/(2j) where B_{2j} are the Bernoulli numbers, starting from j=1
         Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
print("            0,")
for k in range(1,22):
		print("            " + str(bernoulli(2 * k) / (2 * k)).replace("/", ".0 / ") + ".0,")
        */
        private static readonly double[] digammaAsymptoticCoefficients = new[]
        {
            0.0,
            1.0 / 12.0,
            -1.0 / 120.0,
            1.0 / 252.0,
            -1.0 / 240.0,
            1.0 / 132.0,
            -691.0 / 32760.0,
            1.0 / 12.0,
            -3617.0 / 8160.0,
            43867.0 / 14364.0,
            -174611.0 / 6600.0,
            77683.0 / 276.0,
            //-236364091.0 / 65520.0,
            //657931.0 / 12.0,
            //-3392780147.0 / 3480.0,
            //1723168255201.0 / 85932.0,
            //-7709321041217.0 / 16320.0,
            //151628697551.0 / 12.0,
            //-26315271553053477373.0 / 69090840.0,
            //154210205991661.0 / 12.0,
            //-261082718496449122051.0 / 541200.0,
            //1520097643918070802691.0 / 75852.0,
        };

        /* Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
for k in range(0,20):
		print("            %0.34g," % ((-1)**k * (k + 1) * zeta(k + 2)).evalf(34))
         */
        private static readonly double[] trigammaAt1Coefficients = new[]
        {
            1.644934066848226406065691662661266,
            -2.40411380631918847328165611543227,
            3.246969701133414432092649803962559,
            -4.147711020573479956397022760938853,
            5.086715309922245964457943045999855,
            -6.050095664291537111978414031909779,
            7.028541493385610294808429898694158,
            -8.016067142608656936886291077826172,
            9.0089511761503633380243627470918,
            -10.00494188604119472074671648442745,
            11.00270695208638827011782268527895,
            -12.00147256017094221647312224376947,
            13.00079622575576365761662600561976,
            -14.00042823530829849687506793998182,
            15.00022923389112960990132705774158,
            -16.00012219516220568493736209347844,
            17.00006489398550613145744137000293,
            -18.00003434782889755183532543014735,
            19.00001812527864331059390679001808,
            -20.00000953865973585266146983485669,
        };

        /* Coefficients of de Moivre's expansion for the digamma function.
         Each coefficient is B_{2j} where B_{2j} are the Bernoulli numbers, starting from j=1
         Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
print("            0,")
for k in range(1,22):
		print("            " + str(bernoulli(2 * k)).replace("/", ".0 / ") + ".0,")
        */
        private static readonly double[] trigammaAsymptoticCoefficients = new[]
        {
            0.0,
            1.0 / 6.0,
            -1.0 / 30.0,
            1.0 / 42.0,
            -1.0 / 30.0,
            5.0 / 66.0,
            -691.0 / 2730.0,
            7.0 / 6.0,
            -3617.0 / 510.0,
            43867.0 / 798.0,
            //-174611.0 / 330.0,
            //854513.0 / 138.0,
            //-236364091.0 / 2730.0,
            //8553103.0 / 6.0,
            //-23749461029.0 / 870.0,
            //8615841276005.0 / 14322.0,
            //-7709321041217.0 / 510.0,
            //2577687858367.0 / 6.0,
            //-26315271553053477373.0 / 1919190.0,
            //2929993913841559.0 / 6.0,
            //-261082718496449122051.0 / 13530.0,
            //1520097643918070802691.0 / 1806.0,
        };

        /* Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
for k in range(0,20):
		print("            %0.34g," % ((-1)**(k + 1) * (k + 1) * (k + 2) * zeta(k + 3)).evalf(34))
         */
        private static readonly double[] tetragammaAt1Coefficients = new[]
        {
            -2.40411380631918847328165611543227,
            6.493939402266828864185299607925117,
            -12.44313306172043986919106828281656,
            20.34686123968898385783177218399942,
            -30.25047832145768467171365045942366,
            42.17124896031366176885057939216495,
            -56.11246999826060743998823454603553,
            72.0716094092029067041949019767344,
            -90.04447697437075248672044835984707,
            110.0270695208638898066055844537914,
            -132.0161981618803679339180234819651,
            156.0095547090691638913995120674372,
            -182.0055670590078875648032408207655,
            210.0032092744758074331912212073803,
            -240.0018329274330994849151466041803,
            272.0010383037680981033190619200468,
            -306.0005839130912477230594959110022,
            342.0003262550155795906903222203255,
            -380.0001812345349776478542480617762,
            420.0001001492111640800430905073881,
        };

        /* Coefficients of de Moivre's expansion for the digamma function.
         Each coefficient is -(2j+1) B_{2j} where B_{2j} are the Bernoulli numbers, starting from j=0
         Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
print("            0.0,")
for k in range(0,21):
		print("            " + str(-(2 * k + 1) * bernoulli(2 * k)).replace("/", ".0 / ") + ".0,")
        */
        private static readonly double[] tetragammaAsymptoticCoefficients = new[]
        {
            0.0,
            -1.0,
            -1.0 / 2.0,
            1.0 / 6.0,
            -1.0 / 6.0,
            3.0 / 10.0,
            -5.0 / 6.0,
            691.0 / 210.0,
            -35.0 / 2.0,
            //3617.0 / 30.0,
            //-43867.0 / 42.0,
            //1222277.0 / 110.0,
            //-854513.0 / 6.0,
            //1181820455.0 / 546.0,
            //-76977927.0 / 2.0,
            //23749461029.0 / 30.0,
            //-8615841276005.0 / 462.0,
            //84802531453387.0 / 170.0,
            //-90219075042845.0 / 6.0,
            //26315271553053477373.0 / 51870.0,
            //-38089920879940267.0 / 2.0,
            //261082718496449122051.0 / 330.0,
        };

        /* Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
for k in range(1,21):
		print("            " + str(bernoulli(2 * k) / (2 * k * (2 * k - 1))).replace("/", ".0 / ") + ".0,")
        */
        private static readonly double[] gammalnAsymptoticCoefficients = new[]
        {
            1.0 / 12.0,
            -1.0 / 360.0,
            1.0 / 1260.0,
            -1.0 / 1680.0,
            1.0 / 1188.0,
            -691.0 / 360360.0,
            1.0 / 156.0,
            -3617.0 / 122400.0,
            43867.0 / 244188.0,
            -174611.0 / 125400.0,
            77683.0 / 5796.0,
            //-236364091.0 / 1506960.0,
            //657931.0 / 300.0,
            //-3392780147.0 / 93960.0,
            //1723168255201.0 / 2492028.0,
            //-7709321041217.0 / 505920.0,
            //151628697551.0 / 396.0,
            //-26315271553053477373.0 / 2418179400.0,
            //154210205991661.0 / 444.0,
            //-261082718496449122051.0 / 21106800.0,
        };

        /* Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
x = symbols('x')
for c in reversed(Poly(log((exp(x) - 1) / x).series(x, 0, 22).removeO()).all_coeffs()):
	print("            " + str(c).replace("/", ".0 / ") + ".0,")
        */
        private static readonly double[] logExpMinus1RatioAt0Coefficients = new[]
        {
            0.0,
            1.0 / 2.0,
            1.0 / 24.0,
            0.0,
            -1.0 / 2880.0,
            0.0,
            1.0 / 181440.0,
            0.0,
            -1.0 / 9676800.0,
            0.0,
            1.0 / 479001600.0,
            0.0,
            -691.0 / 15692092416000.0,
            0.0,
            1.0 / 1046139494400.0,
            0.0,
            -3617.0 / 170729965486080000.0,
            0.0,
            43867.0 / 91963695909076992000.0,
            0.0,
            -174611.0 / 16057153253965824000000.0,
        };

        /* Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
def next(v):
    idx = 1
    while idx < len(v) and v[idx] == 0:
        idx = idx + 1
    if idx == len(v): return False
    v0 = v[0]
    v[0] = 0
    v[idx] = v[idx] - 1
    v[idx - 1] = v0 + 1
    return True

# Formula for mth coefficient of the normcdfln asymptotic:
# \sum_{n=1}^m (-1)^{n+m+1} / n * \sum_{l1, l2, ..., ln \in N, l1 + l2 + ... + ln = m} (2 * l1 - 1)!! * (2 * l2 - 1)!! * ... * (2 * ln - 1)!!
# Can be obtained by composing the Taylor expansion for log(1 + x) and asymtotic expansion for erfc
def A(m):
    result = S((-1)**(2 + m)) * factorial2(2 * m - 1)
    for n in range(2,m+1):
        coef = S((-1)**(n + 1 + m)) / n
        deltas = []
        for k in range(0, n):
            deltas.append(0)
        deltas[-1] = m - n
        accSum = S(0)
        while True:
            accProd = S(1)
            for delta in deltas:
                accProd = accProd * factorial2(2 * (delta + 1) - 1)
            accSum = accSum + accProd
            if not next(deltas):
                break
        result = result + coef * accSum
    return result

print("            0.0,")
for k in range(1,21):
    print("            " + str(A(k)).replace("/", ".0 / ") + ".0,")
        */
        private static readonly double[] normcdflnAsymptoticCoefficients = new[]
        {
            0.0,
            -1.0,
            5.0 / 2.0,
            -37.0 / 3.0,
            353.0 / 4.0,
            -4081.0 / 5.0,
            55205.0 / 6.0,
            -854197.0 / 7.0,
            14876033.0 / 8.0,
            -288018721.0 / 9.0,
            1227782785.0 / 2.0,
            -142882295557.0 / 11.0,
            3606682364513.0 / 12.0,
            -98158402127761.0 / 13.0,
            //2865624738913445.0 / 14.0,
            //-89338394736560917.0 / 15.0,
            //2962542872271918593.0 / 16.0,
            //-104128401379446177601.0 / 17.0,
            //3867079042971339087365.0 / 18.0,
            //-151312533647578564021477.0 / 19.0,
            //6222025717549801744754273.0 / 20.0,
        };

        #endregion
    }
}

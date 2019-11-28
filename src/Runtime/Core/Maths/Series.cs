using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Core.Maths
{
    struct TruncatedPowerSeries
    {
        public double[] Coefficients { get; }

        public TruncatedPowerSeries(double[] coefficients)
        {
            this.Coefficients = coefficients;
        }

        public static TruncatedPowerSeries Generate(int length, Func<int, double> coefGenerator)
        {
            double[] coefficients = new double[length];
            for (int i = 0; i < length; ++i)
                coefficients[i] = coefGenerator(i);

            return new TruncatedPowerSeries(coefficients);
        }

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

    class PrecomputedSeriesCollection
    {
        public TruncatedPowerSeries GammaAt2 { get; }
        public TruncatedPowerSeries DigammaAt2 { get; }
        /// <summary>
        /// Coefficients of de Moivre's expansion for the digamma function.
        /// </summary>
        public TruncatedPowerSeries CDigamma { get; }
        public TruncatedPowerSeries TrigammaAt1 { get; }
        /// <summary>
        /// Coefficients of de Moivre's expansion for the trigamma function.
        /// </summary>
        public TruncatedPowerSeries CTrigamma { get; }
        public TruncatedPowerSeries TetragammaAt1 { get; }
        /// <summary>
        /// Coefficients of de Moivre's expansion for the tetragamma function.
        /// </summary>
        public TruncatedPowerSeries CTetragamma { get; }
        public TruncatedPowerSeries CGammaln { get; }
        public TruncatedPowerSeries Log1Plus { get; }
        public TruncatedPowerSeries Log1Minus { get; }

        public PrecomputedSeriesCollection(uint precisionBits)
        {
            uint doublePrecisionCutOff = 53;
            if (precisionBits <= doublePrecisionCutOff)
            {
                GammaAt2 = new TruncatedPowerSeries(gammaAt2Coefficients.Take(26).ToArray());
                CDigamma = new TruncatedPowerSeries(cDigammaCoefficients.Take(8).ToArray());
                TrigammaAt1 = new TruncatedPowerSeries(trigammaAt1Coefficients.Take(2).ToArray());
                CTrigamma = new TruncatedPowerSeries(cTrigammaCoefficients.Take(9).ToArray());
                TetragammaAt1 = new TruncatedPowerSeries(tetragammaAt1Coefficients.Take(2).ToArray());
                CTetragamma = new TruncatedPowerSeries(cTetragammaCoefficients.Take(9).ToArray());
                CGammaln = new TruncatedPowerSeries(cGammalnCoefficients.Take(7).ToArray());
                Log1Plus = TruncatedPowerSeries.Generate(6, Log1PlusCoefficient);
                Log1Minus = TruncatedPowerSeries.Generate(5, Log1MinusCoefficient);
            }
            else
            {
                GammaAt2 = new TruncatedPowerSeries(gammaAt2Coefficients.ToArray());
                CDigamma = new TruncatedPowerSeries(cDigammaCoefficients.ToArray());
                TrigammaAt1 = new TruncatedPowerSeries(trigammaAt1Coefficients.ToArray());
                CTrigamma = new TruncatedPowerSeries(cTrigammaCoefficients.ToArray());
                TetragammaAt1 = new TruncatedPowerSeries(tetragammaAt1Coefficients.ToArray());
                CTetragamma = new TruncatedPowerSeries(cTetragammaCoefficients.ToArray());
                CGammaln = new TruncatedPowerSeries(cGammalnCoefficients.ToArray());
                Log1Plus = TruncatedPowerSeries.Generate(16, Log1PlusCoefficient);
                Log1Minus = TruncatedPowerSeries.Generate(16, Log1MinusCoefficient);
            }
            DigammaAt2 = TruncatedPowerSeries.Generate(GammaAt2.Coefficients.Length, i => GammaAt2.Coefficients[i] * (i + 1));
        }

        #region Coefficient generators

        private static double Log1PlusCoefficient(int n) => n == 0 ? 0.0 : (n % 2 == 0 ? -1.0 : 1.0) / n;
        private static double Log1MinusCoefficient(int n) => n == 0 ? 0.0 : -1.0 / n;

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
		print("            %0.34g," % (bernoulli(2 * k) / (2 * k)).evalf(34))
        */
        private static readonly double[] cDigammaCoefficients = new[]
        {
            0.0,
            0.08333333333333332870740406406184775,
            -0.008333333333333333217685101601546194,
            0.00396825396825396803368590781246894,
            -0.004166666666666666608842550800773097,
            0.007575757575757575967845269815370557,
            -0.02109279609279609418726053604586923,
            0.08333333333333332870740406406184775,
            -0.4432598039215686069880462127912324,
            3.053954330270119754686675150878727,
            -26.4562121212121219571145047666505,
            281.4601449275362483604112640023232,
            //-3607.510546398046244576107710599899,
            //54827.58333333333575865253806114197,
            //-974936.823850574670359492301940918,
            //20052695.796688079833984375,
            //-472384867.721629917621612548828125,
            //12635724795.9166660308837890625,
            //-380879311252.45367431640625,
            //12850850499305.083984375,
            //-482414483548501.6875,
            //20040310656516252,
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
		print("            %0.34g," % (bernoulli(2 * k)).evalf(34))
        */
        private static readonly double[] cTrigammaCoefficients = new[]
        {
            0.0,
            0.1666666666666666574148081281236955,
            -0.03333333333333333287074040640618477,
            0.02380952380952380820211544687481364,
            -0.03333333333333333287074040640618477,
            0.07575757575757575967845269815370557,
            -0.2531135531135531024915508169215173,
            1.166666666666666740681534975010436,
            -7.092156862745097711808739404659718,
            54.97117794486215558436015271581709,
            //-529.1242424242424249314353801310062,
            //6192.12318840579700918169692158699,
            //-86580.25311355311714578419923782349,
            //1425517.166666666744276881217956543,
            //-27298231.06781609356403350830078125,
            //601580873.90064239501953125,
            //-15116315767.0921573638916015625,
            //429614643061.16668701171875,
            //-13711655205088.33203125,
            //488332318973593.1875,
            //-19296579341940068,
            //841693047573682560,
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
		print("            %0.34g," % (-(2 * k + 1) * bernoulli(2 * k)).evalf(34))
        */
        private static readonly double[] cTetragammaCoefficients = new[]
        {
            0.0,
            -1,
            -0.5,
            0.1666666666666666574148081281236955,
            -0.1666666666666666574148081281236955,
            0.2999999999999999888977697537484346,
            -0.833333333333333370340767487505218,
            3.290476190476190332390160619979724,
            -17.5,
            //120.5666666666666628771054092794657,
            //-1044.452380952380963208270259201527,
            //11111.6090909090908098733052611351,
            //-142418.8333333333430346101522445679,
            //2164506.327838827855885028839111328,
            //-38488963.5,
            //791648700.966666698455810546875,
            //-18649007090.91991424560546875,
            //498838420314.04119873046875,
            //-15036512507140.833984375,
            //507331242588268.3125,
            //-19044960439970132,
            //791159753019542784,
        };

        /* Python code to generate this table (must not be indented):
from __future__ import division
from sympy import *
for k in range(1,21):
		print("            %0.34g," % (bernoulli(2 * k) / (2 * k * (2 * k - 1))).evalf(34))
        */
        private static readonly double[] cGammalnCoefficients = new[]
        {
            0.08333333333333332870740406406184775,
            -0.002777777777777777883788656865249322,
            0.0007936507936507936501052684619139654,
            -0.0005952380952380952917890599707106958,
            0.0008417508417508417139715759525131489,
            -0.001917526917526917633674554686251668,
            0.006410256410256410034009810772204219,
            -0.02955065359477124231624145522800973,
            0.179644372368830573805098538286984,
            -1.392432216905901132264489206136204,
            13.40286404416839260989036120008677,
            //-156.8482846260020266981882741674781,
            //2193.103333333333466725889593362808,
            //-36108.77125372498994693160057067871,
            //691472.2688513130415230989456176758,
            //-15238221.53940741531550884246826172,
            //382900751.391414165496826171875,
            //-10882266035.7843914031982421875,
            //347320283765.00225830078125,
            //-12369602142269.275390625,
        };

        #endregion
    }
}

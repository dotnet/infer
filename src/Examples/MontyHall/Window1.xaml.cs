// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
// Monty hall problem. Example written by Bryant Tan, July 2009
using System;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Animation;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic;

namespace MontyHall
{
    public partial class Window1 : Window
    {
        #region state
        InferenceEngine ie = new InferenceEngine();
        Variable<int> c, h, p;
        Variable<bool> hostIsObserved;
        Microsoft.ML.Probabilistic.Math.Vector cProbs = Microsoft.ML.Probabilistic.Math.Vector.Zero(3);
        int totalCount = 0;
        int trueCount = 0;
        int state = 0;
        Random rnd = new Random();
        int actualCarPosition;
        #endregion

        #region initialisation
        public Window1()
        {
            InitializeComponent();

            //------------------------
            // The model
            //------------------------
            // c represents the position of the car
            c = Variable.DiscreteUniform(3).Named("Car");

            // p represents the pick. This will be observed
            p = Variable.DiscreteUniform(3).Named("Pick");

            // h represents the host pick.
            h = Variable.New<int>().Named("Host");

            // Whether the host is observed
            hostIsObserved = Variable.Observed<bool>(false);

            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    double[] probs = { 1, 1, 1 };

                    for (int ps = 0; ps < 3; ps++)
                    {
                        if (ps == a || ps == b)
                        {
                            probs[ps] = 0;
                        }
                    }

                    using (Variable.Case(p, a))
                    {
                        using (Variable.Case(c, b))
                        {
                            h.SetTo(Variable.Discrete(probs));
                        }
                    }
                }
            }

            using (Variable.If(hostIsObserved))
            {
                Variable.ConstrainFalse(h == c);
            }

            // Compile the model
            getProbs(h);

            OnReset();
        }
        #endregion

        #region core functions
        private void OnReset()
        {
            for (int q = 0; q < 3; q++)
            {
                cProbs[q] = 1.0 / 3.0;
            }

            string fmtStr = "{0:0.0000}";
            Lbl1.Content = String.Format(fmtStr, cProbs[0]);
            Lbl2.Content = String.Format(fmtStr, cProbs[1]);
            Lbl3.Content = String.Format(fmtStr, cProbs[2]);

            label1.Content = "P ( Car Inside )\nP ( Host Chooses )";

            if (Door1.Width == 20)
            {
                doAnimation("closedoor1");
            }

            if (Door2.Width == 20)
            {
                doAnimation("closedoor2");
            }

            if (Door3.Width == 20)
            {
                doAnimation("closedoor3");
            }

            if (carg1.Opacity == 1)
            {
                doAnimation("hidecarg1");
            }

            if (carg2.Opacity == 1)
            {
                doAnimation("hidecarg2");
            }

            if (carg3.Opacity == 1)
            {
                doAnimation("hidecarg3");
            }

            if (goatg1.Opacity == 1)
            {
                doAnimation("hidegoatg1");
            }

            if (goatg2.Opacity == 1)
            {
                doAnimation("hidegoatg2");
            }

            if (goatg3.Opacity == 1)
            {
                doAnimation("hidegoatg3");
            }

            if (q1.Opacity != 0)
            {
                doAnimation("hideq1");
            }

            if (q2.Opacity != 0)
            {
                doAnimation("hideq2");
            }

            if (q3.Opacity != 0)
            {
                doAnimation("hideq3");
            }

            button2.Visibility = Visibility.Hidden;

            // The actual position of the car
            actualCarPosition = rnd.Next(3);
            state = 0;
        }

        private void OnResetTally()
        {
            totalCount = 0;
            trueCount = 0;
            percentlbl.Content = "";
            button3.Visibility = Visibility.Hidden;
        }

        // Called when a door is picked by the user, at any state
        // state 0: Before any pick
        // state 1: Door has been picked
        // state 2: Host has picked
        // of a new game, or after the host has picked
        private void OnPick(int index)
        {
            if (state != 2)
            {
                // Start a new game
                OnReset();

                // Close all the doors
                if (q1.Opacity == 1)
                {
                    doAnimation("hideq1");
                }

                if (q2.Opacity == 1)
                {
                    doAnimation("hideq2");
                }

                if (q3.Opacity == 1)
                {
                    doAnimation("hideq3");
                }

                // Set the observation
                p.ObservedValue = index;

                // Show the pick with a question mark
                doAnimation("showq" + (index + 1));

                // We don't have a host pick observation yet
                BeforeHostPick();

                // Activate the host pick button
                button2.Visibility = Visibility.Visible;
            }
            else
            {
                // Update the user pick
                p.ObservedValue = index;
                for (int q = 0; q < 3; q++)
                {
                    if (q == actualCarPosition)
                    {
                        cProbs[q] = 1;
                    }
                    else
                    {
                        cProbs[q] = 0;
                    }
                }

                UpdateDoors(cProbs);
                totalCount++;
                if (index == actualCarPosition)
                {
                    trueCount++;
                }
                else
                {
                    doAnimation("showgoatg" + (index + 1));
                    doAnimation("opendoor" + (index + 1));
                }

                percentlbl.Content = String.Format(
                    "Success: {0} out of {1}\n({2}% correct)",
                    trueCount,
                    totalCount,
                    Math.Round(100.0 * ((double)trueCount / (double)totalCount), 0));
                button3.Visibility = Visibility.Visible;
                state = 3;
            }
        }

        private void BeforeHostPick()
        {
            h.ClearObservedValue();
            hostIsObserved.ObservedValue = false;
            Microsoft.ML.Probabilistic.Math.Vector cProbs = getProbs(c);
            Microsoft.ML.Probabilistic.Math.Vector hProbs = getProbs(h);
            string fmtStr = "{0:0.0000}\n{1:0.0000}";

            Lbl1.Content = String.Format(fmtStr, cProbs[0], hProbs[0]);
            Lbl2.Content = String.Format(fmtStr, cProbs[1], hProbs[1]);
            Lbl3.Content = String.Format(fmtStr, cProbs[2], hProbs[2]);
            UpdateDoors(cProbs);
            state = 1;
        }

        private void OnHostPick()
        {
            if (state != 1)
            {
                return;
            }

            // Host cannot pick the user pick, or the car position
            int hc = rnd.Next(3);
            while (hc == p.ObservedValue || hc == actualCarPosition)
            {
                hc = rnd.Next(3);
            }

            doAnimation("opendoor" + (hc + 1));
            doAnimation("showgoatg" + (hc + 1));

            h.ObservedValue = hc;
            hostIsObserved.ObservedValue = true;
            cProbs = getProbs(c);

            string fmtStr = "{0:0.0000}";
            Lbl1.Content = String.Format(fmtStr, cProbs[0]);
            Lbl2.Content = String.Format(fmtStr, cProbs[1]);
            Lbl3.Content = String.Format(fmtStr, cProbs[2]);
            UpdateDoors(cProbs);
            label1.Content = "P ( Car Inside )";
            button2.Visibility = Visibility.Hidden;
            state = 2;
        }
        #endregion

        #region auxiliary functions
        private void doAnimation(string name)
        {
            ((Storyboard)FindResource(name)).Begin(this);
        }

        private void opendoor(int index)
        {
            doAnimation("opendoor" + (index + 1));
        }

        Microsoft.ML.Probabilistic.Math.Vector getProbs(Variable v)
        {
            return ie.Infer<Discrete>(v).GetProbs();
        }

        private void UpdateDoors(Microsoft.ML.Probabilistic.Math.Vector cProbs)
        {
            for (int a = 0; a < 3; a++)
            {
                if (cProbs[a] == 1)
                {
                    doAnimation("showcarg" + (a + 1));
                    doAnimation("opendoor" + (a + 1));
                }
            }
        }
        #endregion

        #region event handlers

        private void Door1_MouseDown(object sender, MouseEventArgs e)
        {
            OnPick(0);
        }

        private void Door2_MouseDown(object sender, MouseEventArgs e)
        {
            OnPick(1);
        }

        private void Door3_MouseDown(object sender, MouseEventArgs e)
        {
            OnPick(2);
        }

        private void Door1_Enter(object sender, MouseEventArgs e)
        {
            this.Cursor = Cursors.Hand;
        }

        private void Door2_Enter(object sender, MouseEventArgs e)
        {
            this.Cursor = Cursors.Hand;
        }

        private void Door3_Enter(object sender, MouseEventArgs e)
        {
            this.Cursor = Cursors.Hand;
        }

        private void Door1_Leave(object sender, MouseEventArgs e)
        {
            this.Cursor = Cursors.Hand;
        }

        private void Door2_Leave(object sender, MouseEventArgs e)
        {
            this.Cursor = Cursors.Hand;
        }

        private void Door3_Leave(object sender, MouseEventArgs e)
        {
            this.Cursor = Cursors.Hand;
        }

        private void button1_Click(object sender, RoutedEventArgs e)
        {
            OnReset();
        }

        private void button3_Click(object sender, RoutedEventArgs e)
        {
            OnResetTally();
        }

        private void button2_Click(object sender, RoutedEventArgs e)
        {
            OnHostPick();
        }
        #endregion
    }
}

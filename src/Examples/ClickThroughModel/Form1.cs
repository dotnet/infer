// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.ComponentModel;
using System.Drawing;
using System.Globalization;
using System.Windows.Forms;

namespace ClickThroughModel
{
  public partial class Form1 : Form
  {
    private int nRanks = 2;
    Inference infer;
    UserData userInput;
    private int dataPictureBoxLocationY;
    private int dataPictureBoxHeight;
    private int dataTrackBarRange;
    bool bgwMustRestart;
    TrackBar[] userClickTrackBars = new TrackBar[4];
    PictureBox[] userClickPictureBoxes = new PictureBox[4];
    Label[] userClickLabels = new Label[4];

    TrackBar[] userParamsTrackBars = new TrackBar[3];
    Label[] userParamsLabels = new Label[3];

    public Form1()
    {
      InitializeComponent();

      makeArrays();

           dataPictureBoxHeight = pictureBoxFF.Size.Height;
      dataPictureBoxLocationY = pictureBoxFF.Location.Y;
      dataTrackBarRange = trackBarFF.Maximum - trackBarFF.Minimum;
    
      infer = new Inference(nRanks);
      userInput = new UserData();
      initializeUserData();
      intializeDependentComponents();

      createClicks(userInput);
      showResults(infer.performInference(userInput));
    }

    private void makeArrays()
    {
      trackBarTT.Tag = clickType.TT;
      trackBarTF.Tag = clickType.TF;
      trackBarFT.Tag = clickType.FT;
      trackBarFF.Tag = clickType.FF;

      userClickTrackBars[0] = trackBarTT;
      userClickPictureBoxes[0] = pictureBoxTT;
      userClickLabels[0] = labelTT;

      userClickTrackBars[1] = trackBarTF;
      userClickPictureBoxes[1] = pictureBoxTF;
      userClickLabels[1] = labelTF;

      userClickTrackBars[2] = trackBarFT;
      userClickPictureBoxes[2] = pictureBoxFT;
      userClickLabels[2] = labelFT;

      userClickTrackBars[3] = trackBarFT;
      userClickPictureBoxes[3] = pictureBoxFF;
      userClickLabels[3] = labelFF;

      trackBarClickNotRel.Tag = probType.ClickNotRel;
      trackBarClickRel.Tag = probType.ClickRel;
      trackBarNoClick.Tag = probType.NoClick;

      userParamsTrackBars[0] = trackBarNoClick;
      userParamsTrackBars[1] = trackBarClickNotRel;
      userParamsTrackBars[2] = trackBarClickRel;
      userParamsLabels[0] = labelNoClick;
      userParamsLabels[1] = labelClickNotRel;
      userParamsLabels[2] = labelClickRel;
    }

    private void initializeUserData()
    {
      userInput.nIters = 10; // int.Parse(textBoxNoIterations.Text);

      textBoxNoOfIters.Text = userInput.nIters.ToString(CultureInfo.InvariantCulture);

      for (int i = 0; i < userParamsTrackBars.Length; i++)
      {
        userInput.probExamine[i] = ((double)userParamsTrackBars[i].Value) / 100;
      }

      userInput.nUsers = 0;
      for (int i = 0; i < userClickTrackBars.Length; i++)
      {
        userInput.nClicks[i] = userClickTrackBars[i].Value;
        userInput.nUsers += userInput.nClicks[i];
      }
    }

    private void intializeDependentComponents()
    {
      for (int i = 0; i < userParamsLabels.Length; i++)
      {
          userParamsLabels[i].Text = ((double)userParamsTrackBars[i].Value / 100.0).ToString(CultureInfo.InvariantCulture);
      }

      for (int i = 0; i < userClickTrackBars.Length; i++)
      {
        updatePictureBox(userClickTrackBars[i], userClickPictureBoxes[i]);
      }
    }

    private void Form1_Load(object sender, EventArgs e)
    {
    }

    private void trackBarParameters_ValueChanged(object sender, EventArgs e)
    {
        TrackBar tbar = (TrackBar)sender;
        int index = (int)tbar.Tag;
        userInput.probExamine[index] = (double)tbar.Value / 100;
        userParamsLabels[index].Text = userInput.probExamine[index].ToString(CultureInfo.InvariantCulture);
        startBackgroundThread();
    }

    private void startBackgroundThread()
    {
        if (bgw.IsBusy)
        {
        bgw.CancelAsync();
        bgwMustRestart = true;
        }
        else
        {
        bgwMustRestart = false;

        // Start the asynchronous operation.
        bgw.RunWorkerAsync();
        }
    }

    private void updatePictureBox(TrackBar curTrackBar, PictureBox curPictureBox)
    {
      curPictureBox.ClientSize = new Size(curPictureBox.Size.Width, (int)Math.Round(curTrackBar.Value * dataPictureBoxHeight / (double)dataTrackBarRange));
      curPictureBox.Location = new Point(curPictureBox.Location.X, dataPictureBoxLocationY + (int)Math.Round((curTrackBar.Maximum - curTrackBar.Value) * dataPictureBoxHeight / (double)dataTrackBarRange));
      curPictureBox.Visible = true;
    }

    private void bgw_DoWork(object sender, DoWorkEventArgs e)
    {
      // Get the BackgroundWorker that raised this event.
      BackgroundWorker worker = sender as BackgroundWorker;

      if (worker.CancellationPending) return;
      UserData userData = new UserData();
      lock (userInput)
      {
        userInput.nClicks.CopyTo(userData.nClicks, 0);
        userData.nIters = userInput.nIters;
        userData.nUsers = userInput.nUsers;
        userInput.probExamine.CopyTo(userData.probExamine, 0);
      }

      createClicks(userData);
      if (worker.CancellationPending) return;
      e.Result = infer.performInference(userData); 
    }

    private void bgw_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
    {
      if (bgwMustRestart) startBackgroundThread();
      else showResults((DocumentStatistics[])e.Result);
    }

    private void bgw_ProgressChanged(object sender, ProgressChangedEventArgs e)
    {
    }

    private void showResults(DocumentStatistics[] docStats)
    {
      if (docStats == null)
      {
        colorInvalid(pictureBoxSummaryDoc1, labelSummaryDoc1.Size);
        colorInvalid(pictureBoxSummaryDoc2, labelSummaryDoc2.Size);
        colorInvalid(pictureBoxRelDoc1, labelRelDoc1.Size);
        colorInvalid(pictureBoxRelDoc2, labelRelDoc2.Size);

        return;
      }

      double summaryDoc1 = docStats[0].inferredAppeal.GetMean();
      double summaryDoc2 = docStats[1].inferredAppeal.GetMean();

      double relDoc1 = docStats[0].inferredRelevance.GetMean();
      double relDoc2 = docStats[1].inferredRelevance.GetMean();

      Size curSize = new Size();
      curSize.Height = labelSummaryDoc1.Size.Height;

      curSize.Width = (int)(summaryDoc1 * labelSummaryDoc1.Size.Width);
      colorResults(pictureBoxSummaryDoc1, curSize, summaryDoc1, true);
      
      curSize.Width = (int)(summaryDoc2 * labelSummaryDoc2.Size.Width);
      colorResults(pictureBoxSummaryDoc2, curSize, summaryDoc2, true);

      curSize.Width = (int)(relDoc1 * labelRelDoc1.Size.Width);
      colorResults(pictureBoxRelDoc1, curSize, relDoc1, false);

      curSize.Width = (int)(relDoc2 * labelRelDoc2.Size.Width);
      colorResults(pictureBoxRelDoc2, curSize, relDoc2, false);
    }

    private void colorResults(PictureBox curPictureBox,  Size curSize, double curValue, bool isSummary)
    {
      if (isSummary)
      {
        curPictureBox.BackColor = Color.Purple;
      }
      else
      {
        curPictureBox.BackColor = Color.Green;
      }

      curPictureBox.ClientSize = curSize;
      curPictureBox.Visible = true;
    }

    private void colorInvalid(PictureBox curPictureBox, Size curSize)
    {
      curPictureBox.BackColor = Color.Red;
      curPictureBox.ClientSize = curSize;
      curPictureBox.Visible = true;
    }

    private void createClicks(UserData userData) 
    {
      userData.clicks = new bool[nRanks][];
      int counts = 0;
      int user;
      for (int rank = 0; rank < nRanks; rank++)
      {
        userData.clicks[rank] = new bool[userData.nUsers];
      }

      for (user = 0; user < userData.nClicks[(int)clickType.FT]; user++)
      {
        userData.clicks[0][counts] = false;
        userData.clicks[1][counts] = true;
        counts = counts + 1;
      }

      for (user = 0; user < userData.nClicks[(int)clickType.TT]; user++)
      {
        userData.clicks[0][counts] = true;
        userData.clicks[1][counts] = true;
        counts = counts + 1;
      }

      for (user = 0; user < userData.nClicks[(int)clickType.TF]; user++)
      {
        userData.clicks[0][counts] = true;
        userData.clicks[1][counts] = false;
        counts = counts + 1;
      }

      for (user = 0; user < userData.nClicks[(int)clickType.FF]; user++)
      {
        userData.clicks[0][counts] = false;
        userData.clicks[1][counts] = false;
        counts = counts + 1;
      }
    }

    private void trackBarClicks_ValueChanged(object sender, EventArgs e)
    {
      TrackBar tbar = (TrackBar)sender;
      int index = (int)tbar.Tag;

      updatePictureBox(tbar, userClickPictureBoxes[index]);

      userInput.nClicks[index] = tbar.Value;
      userClickLabels[index].Text = userInput.nClicks[index].ToString(CultureInfo.InvariantCulture);
      userInput.nUsers = 0;
      for (int i = 0; i < userInput.nClicks.Length; i++)
      {
        userInput.nUsers += userInput.nClicks[i];
      }

      startBackgroundThread();
    }

    private void DebugButton_Click(object sender, EventArgs e)
    {
      Console.WriteLine("Debug");
    }

    private void textBoxNoOfIters_KeyUp(object sender, KeyEventArgs e)
    {
      if (e.KeyCode == Keys.Enter)
      {
        try
        {
          userInput.nIters = int.Parse(textBoxNoOfIters.Text);
          startBackgroundThread();
        }
        catch
        {
        }
      }
    }
  }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
namespace ClickThroughModel
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }

            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.trackBarTT = new System.Windows.Forms.TrackBar();
            this.trackBarTF = new System.Windows.Forms.TrackBar();
            this.trackBarFT = new System.Windows.Forms.TrackBar();
            this.pictureBoxTT = new System.Windows.Forms.PictureBox();
            this.pictureBoxTF = new System.Windows.Forms.PictureBox();
            this.pictureBoxFT = new System.Windows.Forms.PictureBox();
            this.panel2 = new System.Windows.Forms.Panel();
            this.label7 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.labelFF = new System.Windows.Forms.Label();
            this.labelFT = new System.Windows.Forms.Label();
            this.labelTF = new System.Windows.Forms.Label();
            this.labelTT = new System.Windows.Forms.Label();
            this.pictureBox4 = new System.Windows.Forms.PictureBox();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.pictureBox3 = new System.Windows.Forms.PictureBox();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.pictureBoxFF = new System.Windows.Forms.PictureBox();
            this.trackBarFF = new System.Windows.Forms.TrackBar();
            this.panel3 = new System.Windows.Forms.Panel();
            this.DebugButton = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.textBoxNoOfIters = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.pictureBoxRelDoc2 = new System.Windows.Forms.PictureBox();
            this.pictureBoxRelDoc1 = new System.Windows.Forms.PictureBox();
            this.pictureBoxSummaryDoc2 = new System.Windows.Forms.PictureBox();
            this.pictureBoxSummaryDoc1 = new System.Windows.Forms.PictureBox();
            this.pictureBox5 = new System.Windows.Forms.PictureBox();
            this.textBox11 = new System.Windows.Forms.TextBox();
            this.labelSummaryDoc1 = new System.Windows.Forms.Label();
            this.labelRelDoc1 = new System.Windows.Forms.Label();
            this.labelSummaryDoc2 = new System.Windows.Forms.Label();
            this.labelRelDoc2 = new System.Windows.Forms.Label();
            this.pictureBoxModel = new System.Windows.Forms.PictureBox();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.trackBarClickNotRel = new System.Windows.Forms.TrackBar();
            this.labelClickNotRel = new System.Windows.Forms.Label();
            this.trackBarClickRel = new System.Windows.Forms.TrackBar();
            this.labelClickRel = new System.Windows.Forms.Label();
            this.trackBarNoClick = new System.Windows.Forms.TrackBar();
            this.labelNoClick = new System.Windows.Forms.Label();
            this.bgw = new System.ComponentModel.BackgroundWorker();
            this.panel1 = new System.Windows.Forms.Panel();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarTT)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarTF)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarFT)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxTT)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxTF)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxFT)).BeginInit();
            this.panel2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxFF)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarFF)).BeginInit();
            this.panel3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxRelDoc2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxRelDoc1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxSummaryDoc2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxSummaryDoc1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxModel)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarClickNotRel)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarClickRel)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarNoClick)).BeginInit();
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // trackBarTT
            // 
            this.trackBarTT.AutoSize = false;
            this.trackBarTT.Location = new System.Drawing.Point(58, 74);
            this.trackBarTT.Margin = new System.Windows.Forms.Padding(0);
            this.trackBarTT.Maximum = 100;
            this.trackBarTT.Name = "trackBarTT";
            this.trackBarTT.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.trackBarTT.Size = new System.Drawing.Size(68, 157);
            this.trackBarTT.TabIndex = 7;
            this.trackBarTT.Tag = "clickType.TT";
            this.trackBarTT.TickFrequency = 10;
            this.trackBarTT.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarTT.Value = 50;
            this.trackBarTT.ValueChanged += new System.EventHandler(this.trackBarClicks_ValueChanged);
            // 
            // trackBarTF
            // 
            this.trackBarTF.Location = new System.Drawing.Point(190, 74);
            this.trackBarTF.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.trackBarTF.Maximum = 100;
            this.trackBarTF.Name = "trackBarTF";
            this.trackBarTF.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.trackBarTF.Size = new System.Drawing.Size(64, 157);
            this.trackBarTF.TabIndex = 8;
            this.trackBarTF.Tag = "clickType.TF";
            this.trackBarTF.TickFrequency = 10;
            this.trackBarTF.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarTF.Value = 50;
            this.trackBarTF.ValueChanged += new System.EventHandler(this.trackBarClicks_ValueChanged);
            // 
            // trackBarFT
            // 
            this.trackBarFT.BackColor = System.Drawing.Color.OldLace;
            this.trackBarFT.Location = new System.Drawing.Point(316, 74);
            this.trackBarFT.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.trackBarFT.Maximum = 100;
            this.trackBarFT.Name = "trackBarFT";
            this.trackBarFT.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.trackBarFT.Size = new System.Drawing.Size(64, 157);
            this.trackBarFT.TabIndex = 9;
            this.trackBarFT.Tag = "clickType.FT";
            this.trackBarFT.TickFrequency = 10;
            this.trackBarFT.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarFT.Value = 50;
            this.trackBarFT.ValueChanged += new System.EventHandler(this.trackBarClicks_ValueChanged);
            // 
            // pictureBoxTT
            // 
            this.pictureBoxTT.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.pictureBoxTT.BackColor = System.Drawing.Color.Sienna;
            this.pictureBoxTT.Location = new System.Drawing.Point(94, 94);
            this.pictureBoxTT.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBoxTT.Name = "pictureBoxTT";
            this.pictureBoxTT.Size = new System.Drawing.Size(52, 117);
            this.pictureBoxTT.TabIndex = 11;
            this.pictureBoxTT.TabStop = false;
            this.pictureBoxTT.Visible = false;
            // 
            // pictureBoxTF
            // 
            this.pictureBoxTF.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.pictureBoxTF.BackColor = System.Drawing.Color.Sienna;
            this.pictureBoxTF.Location = new System.Drawing.Point(225, 94);
            this.pictureBoxTF.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBoxTF.Name = "pictureBoxTF";
            this.pictureBoxTF.Size = new System.Drawing.Size(52, 117);
            this.pictureBoxTF.TabIndex = 12;
            this.pictureBoxTF.TabStop = false;
            this.pictureBoxTF.Visible = false;
            // 
            // pictureBoxFT
            // 
            this.pictureBoxFT.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.pictureBoxFT.BackColor = System.Drawing.Color.Sienna;
            this.pictureBoxFT.Location = new System.Drawing.Point(351, 94);
            this.pictureBoxFT.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBoxFT.Name = "pictureBoxFT";
            this.pictureBoxFT.Size = new System.Drawing.Size(52, 117);
            this.pictureBoxFT.TabIndex = 13;
            this.pictureBoxFT.TabStop = false;
            this.pictureBoxFT.Visible = false;
            // 
            // panel2
            // 
            this.panel2.BackColor = System.Drawing.Color.OldLace;
            this.panel2.Controls.Add(this.label7);
            this.panel2.Controls.Add(this.label6);
            this.panel2.Controls.Add(this.label5);
            this.panel2.Controls.Add(this.label4);
            this.panel2.Controls.Add(this.labelFF);
            this.panel2.Controls.Add(this.labelFT);
            this.panel2.Controls.Add(this.labelTF);
            this.panel2.Controls.Add(this.labelTT);
            this.panel2.Controls.Add(this.pictureBox4);
            this.panel2.Controls.Add(this.textBox2);
            this.panel2.Controls.Add(this.pictureBox3);
            this.panel2.Controls.Add(this.pictureBox2);
            this.panel2.Controls.Add(this.pictureBox1);
            this.panel2.Controls.Add(this.pictureBoxFF);
            this.panel2.Controls.Add(this.pictureBoxFT);
            this.panel2.Controls.Add(this.pictureBoxTF);
            this.panel2.Controls.Add(this.pictureBoxTT);
            this.panel2.Controls.Add(this.trackBarFT);
            this.panel2.Controls.Add(this.trackBarTF);
            this.panel2.Controls.Add(this.trackBarTT);
            this.panel2.Controls.Add(this.trackBarFF);
            this.panel2.Location = new System.Drawing.Point(860, 46);
            this.panel2.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(543, 415);
            this.panel2.TabIndex = 14;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(466, 212);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(48, 20);
            this.label7.TabIndex = 28;
            this.label7.Text = "users";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(352, 211);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(48, 20);
            this.label6.TabIndex = 27;
            this.label6.Text = "users";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(228, 211);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(48, 20);
            this.label5.TabIndex = 26;
            this.label5.Text = "users";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(98, 212);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(48, 20);
            this.label4.TabIndex = 25;
            this.label4.Text = "users";
            // 
            // labelFF
            // 
            this.labelFF.AutoSize = true;
            this.labelFF.BackColor = System.Drawing.Color.Transparent;
            this.labelFF.Font = new System.Drawing.Font("Calibri", 11.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelFF.ForeColor = System.Drawing.Color.Black;
            this.labelFF.Location = new System.Drawing.Point(468, 172);
            this.labelFF.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelFF.Name = "labelFF";
            this.labelFF.Size = new System.Drawing.Size(36, 28);
            this.labelFF.TabIndex = 24;
            this.labelFF.Text = "50";
            // 
            // labelFT
            // 
            this.labelFT.AutoSize = true;
            this.labelFT.BackColor = System.Drawing.Color.Transparent;
            this.labelFT.Font = new System.Drawing.Font("Calibri", 11.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelFT.ForeColor = System.Drawing.Color.Black;
            this.labelFT.Location = new System.Drawing.Point(358, 172);
            this.labelFT.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelFT.Name = "labelFT";
            this.labelFT.Size = new System.Drawing.Size(36, 28);
            this.labelFT.TabIndex = 23;
            this.labelFT.Text = "50";
            // 
            // labelTF
            // 
            this.labelTF.AutoSize = true;
            this.labelTF.BackColor = System.Drawing.Color.Transparent;
            this.labelTF.Font = new System.Drawing.Font("Calibri", 11.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelTF.ForeColor = System.Drawing.Color.Black;
            this.labelTF.Location = new System.Drawing.Point(232, 172);
            this.labelTF.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelTF.Name = "labelTF";
            this.labelTF.Size = new System.Drawing.Size(36, 28);
            this.labelTF.TabIndex = 22;
            this.labelTF.Text = "50";
            // 
            // labelTT
            // 
            this.labelTT.AutoSize = true;
            this.labelTT.BackColor = System.Drawing.Color.Transparent;
            this.labelTT.Font = new System.Drawing.Font("Calibri", 11.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelTT.ForeColor = System.Drawing.Color.Black;
            this.labelTT.Location = new System.Drawing.Point(102, 172);
            this.labelTT.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelTT.Name = "labelTT";
            this.labelTT.Size = new System.Drawing.Size(36, 28);
            this.labelTT.TabIndex = 21;
            this.labelTT.Text = "50";
            // 
            // pictureBox4
            // 
            this.pictureBox4.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox4.Image")));
            this.pictureBox4.Location = new System.Drawing.Point(428, 257);
            this.pictureBox4.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBox4.Name = "pictureBox4";
            this.pictureBox4.Size = new System.Drawing.Size(68, 129);
            this.pictureBox4.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox4.TabIndex = 19;
            this.pictureBox4.TabStop = false;
            // 
            // textBox2
            // 
            this.textBox2.BackColor = System.Drawing.Color.OldLace;
            this.textBox2.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.textBox2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.textBox2.Font = new System.Drawing.Font("Calibri", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBox2.ForeColor = System.Drawing.Color.Black;
            this.textBox2.Location = new System.Drawing.Point(0, 0);
            this.textBox2.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBox2.Name = "textBox2";
            this.textBox2.ReadOnly = true;
            this.textBox2.Size = new System.Drawing.Size(543, 35);
            this.textBox2.TabIndex = 2;
            this.textBox2.TabStop = false;
            this.textBox2.Text = "Clicks received for 2 documents";
            this.textBox2.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.textBox2.WordWrap = false;
            // 
            // pictureBox3
            // 
            this.pictureBox3.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox3.Image")));
            this.pictureBox3.Location = new System.Drawing.Point(315, 257);
            this.pictureBox3.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBox3.Name = "pictureBox3";
            this.pictureBox3.Size = new System.Drawing.Size(86, 129);
            this.pictureBox3.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox3.TabIndex = 18;
            this.pictureBox3.TabStop = false;
            // 
            // pictureBox2
            // 
            this.pictureBox2.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox2.Image")));
            this.pictureBox2.Location = new System.Drawing.Point(190, 257);
            this.pictureBox2.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(86, 129);
            this.pictureBox2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox2.TabIndex = 17;
            this.pictureBox2.TabStop = false;
            // 
            // pictureBox1
            // 
            this.pictureBox1.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox1.Image")));
            this.pictureBox1.Location = new System.Drawing.Point(62, 257);
            this.pictureBox1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(86, 129);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 16;
            this.pictureBox1.TabStop = false;
            // 
            // pictureBoxFF
            // 
            this.pictureBoxFF.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.pictureBoxFF.BackColor = System.Drawing.Color.Sienna;
            this.pictureBoxFF.Location = new System.Drawing.Point(462, 94);
            this.pictureBoxFF.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBoxFF.Name = "pictureBoxFF";
            this.pictureBoxFF.Size = new System.Drawing.Size(52, 117);
            this.pictureBoxFF.TabIndex = 14;
            this.pictureBoxFF.TabStop = false;
            this.pictureBoxFF.Visible = false;
            // 
            // trackBarFF
            // 
            this.trackBarFF.BackColor = System.Drawing.Color.OldLace;
            this.trackBarFF.Location = new System.Drawing.Point(428, 74);
            this.trackBarFF.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.trackBarFF.Maximum = 100;
            this.trackBarFF.Name = "trackBarFF";
            this.trackBarFF.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.trackBarFF.Size = new System.Drawing.Size(64, 157);
            this.trackBarFF.TabIndex = 20;
            this.trackBarFF.Tag = "clickType.FF";
            this.trackBarFF.TickFrequency = 10;
            this.trackBarFF.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarFF.Value = 50;
            this.trackBarFF.ValueChanged += new System.EventHandler(this.trackBarClicks_ValueChanged);
            // 
            // panel3
            // 
            this.panel3.BackColor = System.Drawing.Color.Moccasin;
            this.panel3.Controls.Add(this.DebugButton);
            this.panel3.Controls.Add(this.label1);
            this.panel3.Controls.Add(this.label3);
            this.panel3.Controls.Add(this.textBoxNoOfIters);
            this.panel3.Controls.Add(this.label2);
            this.panel3.Controls.Add(this.pictureBoxRelDoc2);
            this.panel3.Controls.Add(this.pictureBoxRelDoc1);
            this.panel3.Controls.Add(this.pictureBoxSummaryDoc2);
            this.panel3.Controls.Add(this.pictureBoxSummaryDoc1);
            this.panel3.Controls.Add(this.pictureBox5);
            this.panel3.Controls.Add(this.textBox11);
            this.panel3.Controls.Add(this.labelSummaryDoc1);
            this.panel3.Controls.Add(this.labelRelDoc1);
            this.panel3.Controls.Add(this.labelSummaryDoc2);
            this.panel3.Controls.Add(this.labelRelDoc2);
            this.panel3.ForeColor = System.Drawing.Color.Navy;
            this.panel3.Location = new System.Drawing.Point(860, 523);
            this.panel3.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(543, 495);
            this.panel3.TabIndex = 15;
            // 
            // DebugButton
            // 
            this.DebugButton.BackColor = System.Drawing.Color.Silver;
            this.DebugButton.Location = new System.Drawing.Point(58, 346);
            this.DebugButton.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.DebugButton.Name = "DebugButton";
            this.DebugButton.Size = new System.Drawing.Size(112, 35);
            this.DebugButton.TabIndex = 18;
            this.DebugButton.Text = "Debug";
            this.DebugButton.UseVisualStyleBackColor = false;
            this.DebugButton.Visible = false;
            this.DebugButton.Click += new System.EventHandler(this.DebugButton_Click);
            // 
            // label1
            // 
            this.label1.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Calibri", 8.25F);
            this.label1.Location = new System.Drawing.Point(160, 442);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(233, 21);
            this.label1.TabIndex = 20;
            this.label1.Text = "Number of algorithm iterations:";
            this.label1.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label3
            // 
            this.label3.BackColor = System.Drawing.Color.Green;
            this.label3.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(255)))), ((int)(((byte)(128)))));
            this.label3.Location = new System.Drawing.Point(333, 88);
            this.label3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(120, 75);
            this.label3.TabIndex = 36;
            this.label3.Text = "Document relevance";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // textBoxNoOfIters
            // 
            this.textBoxNoOfIters.AcceptsReturn = true;
            this.textBoxNoOfIters.Location = new System.Drawing.Point(406, 435);
            this.textBoxNoOfIters.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxNoOfIters.Name = "textBoxNoOfIters";
            this.textBoxNoOfIters.Size = new System.Drawing.Size(46, 26);
            this.textBoxNoOfIters.TabIndex = 19;
            this.textBoxNoOfIters.KeyUp += new System.Windows.Forms.KeyEventHandler(this.textBoxNoOfIters_KeyUp);
            // 
            // label2
            // 
            this.label2.BackColor = System.Drawing.Color.Purple;
            this.label2.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(255)))), ((int)(((byte)(128)))));
            this.label2.Location = new System.Drawing.Point(186, 88);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(120, 75);
            this.label2.TabIndex = 35;
            this.label2.Text = "Appeal of summary";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // pictureBoxRelDoc2
            // 
            this.pictureBoxRelDoc2.BackColor = System.Drawing.Color.Green;
            this.pictureBoxRelDoc2.Location = new System.Drawing.Point(333, 271);
            this.pictureBoxRelDoc2.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBoxRelDoc2.Name = "pictureBoxRelDoc2";
            this.pictureBoxRelDoc2.Size = new System.Drawing.Size(100, 40);
            this.pictureBoxRelDoc2.TabIndex = 34;
            this.pictureBoxRelDoc2.TabStop = false;
            // 
            // pictureBoxRelDoc1
            // 
            this.pictureBoxRelDoc1.BackColor = System.Drawing.Color.Green;
            this.pictureBoxRelDoc1.Location = new System.Drawing.Point(333, 197);
            this.pictureBoxRelDoc1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBoxRelDoc1.Name = "pictureBoxRelDoc1";
            this.pictureBoxRelDoc1.Size = new System.Drawing.Size(100, 40);
            this.pictureBoxRelDoc1.TabIndex = 32;
            this.pictureBoxRelDoc1.TabStop = false;
            // 
            // pictureBoxSummaryDoc2
            // 
            this.pictureBoxSummaryDoc2.BackColor = System.Drawing.Color.Purple;
            this.pictureBoxSummaryDoc2.Location = new System.Drawing.Point(184, 271);
            this.pictureBoxSummaryDoc2.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBoxSummaryDoc2.Name = "pictureBoxSummaryDoc2";
            this.pictureBoxSummaryDoc2.Size = new System.Drawing.Size(100, 40);
            this.pictureBoxSummaryDoc2.TabIndex = 30;
            this.pictureBoxSummaryDoc2.TabStop = false;
            // 
            // pictureBoxSummaryDoc1
            // 
            this.pictureBoxSummaryDoc1.BackColor = System.Drawing.Color.Purple;
            this.pictureBoxSummaryDoc1.Location = new System.Drawing.Point(186, 197);
            this.pictureBoxSummaryDoc1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBoxSummaryDoc1.Name = "pictureBoxSummaryDoc1";
            this.pictureBoxSummaryDoc1.Size = new System.Drawing.Size(100, 40);
            this.pictureBoxSummaryDoc1.TabIndex = 28;
            this.pictureBoxSummaryDoc1.TabStop = false;
            // 
            // pictureBox5
            // 
            this.pictureBox5.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox5.Image")));
            this.pictureBox5.Location = new System.Drawing.Point(57, 77);
            this.pictureBox5.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBox5.Name = "pictureBox5";
            this.pictureBox5.Size = new System.Drawing.Size(117, 235);
            this.pictureBox5.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox5.TabIndex = 25;
            this.pictureBox5.TabStop = false;
            // 
            // textBox11
            // 
            this.textBox11.BackColor = System.Drawing.Color.Moccasin;
            this.textBox11.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.textBox11.Dock = System.Windows.Forms.DockStyle.Fill;
            this.textBox11.Font = new System.Drawing.Font("Calibri", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBox11.ForeColor = System.Drawing.Color.Black;
            this.textBox11.Location = new System.Drawing.Point(0, 0);
            this.textBox11.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBox11.Name = "textBox11";
            this.textBox11.ReadOnly = true;
            this.textBox11.Size = new System.Drawing.Size(543, 39);
            this.textBox11.TabIndex = 20;
            this.textBox11.TabStop = false;
            this.textBox11.Text = "Inference";
            this.textBox11.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.textBox11.WordWrap = false;
            // 
            // labelSummaryDoc1
            // 
            this.labelSummaryDoc1.BackColor = System.Drawing.Color.White;
            this.labelSummaryDoc1.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelSummaryDoc1.ForeColor = System.Drawing.Color.White;
            this.labelSummaryDoc1.Location = new System.Drawing.Point(186, 197);
            this.labelSummaryDoc1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelSummaryDoc1.Name = "labelSummaryDoc1";
            this.labelSummaryDoc1.Size = new System.Drawing.Size(120, 42);
            this.labelSummaryDoc1.TabIndex = 37;
            this.labelSummaryDoc1.Text = "  ";
            this.labelSummaryDoc1.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // labelRelDoc1
            // 
            this.labelRelDoc1.BackColor = System.Drawing.Color.White;
            this.labelRelDoc1.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelRelDoc1.ForeColor = System.Drawing.Color.White;
            this.labelRelDoc1.Location = new System.Drawing.Point(333, 197);
            this.labelRelDoc1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelRelDoc1.Name = "labelRelDoc1";
            this.labelRelDoc1.Size = new System.Drawing.Size(120, 42);
            this.labelRelDoc1.TabIndex = 38;
            this.labelRelDoc1.Text = "  ";
            this.labelRelDoc1.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // labelSummaryDoc2
            // 
            this.labelSummaryDoc2.BackColor = System.Drawing.Color.White;
            this.labelSummaryDoc2.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelSummaryDoc2.ForeColor = System.Drawing.Color.White;
            this.labelSummaryDoc2.Location = new System.Drawing.Point(184, 271);
            this.labelSummaryDoc2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelSummaryDoc2.Name = "labelSummaryDoc2";
            this.labelSummaryDoc2.Size = new System.Drawing.Size(120, 42);
            this.labelSummaryDoc2.TabIndex = 39;
            this.labelSummaryDoc2.Text = "  ";
            this.labelSummaryDoc2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // labelRelDoc2
            // 
            this.labelRelDoc2.BackColor = System.Drawing.Color.White;
            this.labelRelDoc2.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelRelDoc2.ForeColor = System.Drawing.Color.White;
            this.labelRelDoc2.Location = new System.Drawing.Point(333, 271);
            this.labelRelDoc2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelRelDoc2.Name = "labelRelDoc2";
            this.labelRelDoc2.Size = new System.Drawing.Size(120, 42);
            this.labelRelDoc2.TabIndex = 40;
            this.labelRelDoc2.Text = "  ";
            this.labelRelDoc2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // pictureBoxModel
            // 
            this.pictureBoxModel.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(254)))), ((int)(((byte)(251)))), ((int)(((byte)(226)))));
            this.pictureBoxModel.Image = ((System.Drawing.Image)(resources.GetObject("pictureBoxModel.Image")));
            this.pictureBoxModel.Location = new System.Drawing.Point(34, 57);
            this.pictureBoxModel.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBoxModel.Name = "pictureBoxModel";
            this.pictureBoxModel.Size = new System.Drawing.Size(730, 915);
            this.pictureBoxModel.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBoxModel.TabIndex = 0;
            this.pictureBoxModel.TabStop = false;
            // 
            // textBox1
            // 
            this.textBox1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(254)))), ((int)(((byte)(251)))), ((int)(((byte)(226)))));
            this.textBox1.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.textBox1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.textBox1.Font = new System.Drawing.Font("Calibri", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBox1.ForeColor = System.Drawing.Color.Black;
            this.textBox1.Location = new System.Drawing.Point(0, 0);
            this.textBox1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBox1.Name = "textBox1";
            this.textBox1.ReadOnly = true;
            this.textBox1.Size = new System.Drawing.Size(759, 35);
            this.textBox1.TabIndex = 1;
            this.textBox1.TabStop = false;
            this.textBox1.Text = "User model: State transition diagram";
            this.textBox1.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.textBox1.WordWrap = false;
            // 
            // trackBarClickNotRel
            // 
            this.trackBarClickNotRel.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(254)))), ((int)(((byte)(251)))), ((int)(((byte)(226)))));
            this.trackBarClickNotRel.Location = new System.Drawing.Point(591, 663);
            this.trackBarClickNotRel.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.trackBarClickNotRel.Maximum = 100;
            this.trackBarClickNotRel.Minimum = 1;
            this.trackBarClickNotRel.Name = "trackBarClickNotRel";
            this.trackBarClickNotRel.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.trackBarClickNotRel.Size = new System.Drawing.Size(64, 105);
            this.trackBarClickNotRel.TabIndex = 4;
            this.trackBarClickNotRel.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarClickNotRel.Value = 90;
            this.trackBarClickNotRel.ValueChanged += new System.EventHandler(this.trackBarParameters_ValueChanged);
            // 
            // labelClickNotRel
            // 
            this.labelClickNotRel.AutoSize = true;
            this.labelClickNotRel.Font = new System.Drawing.Font("Calibri", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelClickNotRel.Location = new System.Drawing.Point(633, 711);
            this.labelClickNotRel.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelClickNotRel.Name = "labelClickNotRel";
            this.labelClickNotRel.Size = new System.Drawing.Size(29, 27);
            this.labelClickNotRel.TabIndex = 5;
            this.labelClickNotRel.Text = "hi";
            // 
            // trackBarClickRel
            // 
            this.trackBarClickRel.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(254)))), ((int)(((byte)(251)))), ((int)(((byte)(226)))));
            this.trackBarClickRel.Location = new System.Drawing.Point(346, 863);
            this.trackBarClickRel.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.trackBarClickRel.Maximum = 100;
            this.trackBarClickRel.Minimum = 1;
            this.trackBarClickRel.Name = "trackBarClickRel";
            this.trackBarClickRel.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.trackBarClickRel.Size = new System.Drawing.Size(64, 105);
            this.trackBarClickRel.TabIndex = 6;
            this.trackBarClickRel.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarClickRel.Value = 20;
            this.trackBarClickRel.ValueChanged += new System.EventHandler(this.trackBarParameters_ValueChanged);
            // 
            // labelClickRel
            // 
            this.labelClickRel.AutoSize = true;
            this.labelClickRel.Font = new System.Drawing.Font("Calibri", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelClickRel.Location = new System.Drawing.Point(398, 892);
            this.labelClickRel.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelClickRel.Name = "labelClickRel";
            this.labelClickRel.Size = new System.Drawing.Size(29, 27);
            this.labelClickRel.TabIndex = 7;
            this.labelClickRel.Text = "hi";
            // 
            // trackBarNoClick
            // 
            this.trackBarNoClick.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(254)))), ((int)(((byte)(251)))), ((int)(((byte)(226)))));
            this.trackBarNoClick.Location = new System.Drawing.Point(602, 358);
            this.trackBarNoClick.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.trackBarNoClick.Maximum = 100;
            this.trackBarNoClick.Name = "trackBarNoClick";
            this.trackBarNoClick.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.trackBarNoClick.Size = new System.Drawing.Size(64, 105);
            this.trackBarNoClick.TabIndex = 2;
            this.trackBarNoClick.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarNoClick.Value = 80;
            this.trackBarNoClick.ValueChanged += new System.EventHandler(this.trackBarParameters_ValueChanged);
            // 
            // labelNoClick
            // 
            this.labelNoClick.AutoSize = true;
            this.labelNoClick.Font = new System.Drawing.Font("Calibri", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelNoClick.Location = new System.Drawing.Point(642, 405);
            this.labelNoClick.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelNoClick.Name = "labelNoClick";
            this.labelNoClick.Size = new System.Drawing.Size(29, 27);
            this.labelNoClick.TabIndex = 3;
            this.labelNoClick.Text = "hi";
            // 
            // bgw
            // 
            this.bgw.WorkerReportsProgress = true;
            this.bgw.WorkerSupportsCancellation = true;
            this.bgw.DoWork += new System.ComponentModel.DoWorkEventHandler(this.bgw_DoWork);
            this.bgw.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.bgw_RunWorkerCompleted);
            this.bgw.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(this.bgw_ProgressChanged);
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(254)))), ((int)(((byte)(251)))), ((int)(((byte)(226)))));
            this.panel1.Controls.Add(this.labelClickRel);
            this.panel1.Controls.Add(this.labelNoClick);
            this.panel1.Controls.Add(this.trackBarClickRel);
            this.panel1.Controls.Add(this.trackBarNoClick);
            this.panel1.Controls.Add(this.labelClickNotRel);
            this.panel1.Controls.Add(this.trackBarClickNotRel);
            this.panel1.Controls.Add(this.textBox1);
            this.panel1.Controls.Add(this.pictureBoxModel);
            this.panel1.Location = new System.Drawing.Point(18, 46);
            this.panel1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(759, 972);
            this.panel1.TabIndex = 17;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(1494, 1086);
            this.Controls.Add(this.panel3);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.panel1);
            this.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.MaximizeBox = false;
            this.Name = "Form1";
            this.Text = "Learning relevance from clickthrough data using Infer.NET";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.trackBarTT)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarTF)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarFT)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxTT)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxTF)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxFT)).EndInit();
            this.panel2.ResumeLayout(false);
            this.panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxFF)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarFF)).EndInit();
            this.panel3.ResumeLayout(false);
            this.panel3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxRelDoc2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxRelDoc1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxSummaryDoc2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxSummaryDoc1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxModel)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarClickNotRel)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarClickRel)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarNoClick)).EndInit();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TrackBar trackBarTT;
        private System.Windows.Forms.TrackBar trackBarTF;
        private System.Windows.Forms.TrackBar trackBarFT;
        private System.Windows.Forms.PictureBox pictureBoxTT;
        private System.Windows.Forms.PictureBox pictureBoxTF;
        private System.Windows.Forms.PictureBox pictureBoxFT;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.PictureBox pictureBoxFF;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.PictureBox pictureBox3;
        private System.Windows.Forms.PictureBox pictureBox2;
        private System.Windows.Forms.TextBox textBox2;
        private System.Windows.Forms.PictureBox pictureBox4;
        private System.Windows.Forms.Panel panel3;
        private System.Windows.Forms.TextBox textBox11;
        private System.Windows.Forms.PictureBox pictureBoxModel;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.TrackBar trackBarClickNotRel;
        private System.Windows.Forms.Label labelClickNotRel;
        private System.Windows.Forms.TrackBar trackBarClickRel;
        private System.Windows.Forms.Label labelClickRel;
        private System.Windows.Forms.TrackBar trackBarNoClick;
        private System.Windows.Forms.Label labelNoClick;
        private System.ComponentModel.BackgroundWorker bgw;
        private System.Windows.Forms.TrackBar trackBarFF;
        private System.Windows.Forms.PictureBox pictureBox5;
        private System.Windows.Forms.Label labelTT;
        private System.Windows.Forms.Label labelFT;
        private System.Windows.Forms.Label labelTF;
        private System.Windows.Forms.Label labelFF;
        private System.Windows.Forms.PictureBox pictureBoxSummaryDoc1;
        private System.Windows.Forms.PictureBox pictureBoxSummaryDoc2;
        private System.Windows.Forms.PictureBox pictureBoxRelDoc2;
        private System.Windows.Forms.PictureBox pictureBoxRelDoc1;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Button DebugButton;
        private System.Windows.Forms.TextBox textBoxNoOfIters;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label labelSummaryDoc1;
        private System.Windows.Forms.Label labelRelDoc1;
        private System.Windows.Forms.Label labelSummaryDoc2;
        private System.Windows.Forms.Label labelRelDoc2;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Reflection;
using System.Drawing;
using System.Text.RegularExpressions;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    /// <summary>
    /// A view of a set of examples or tutorials which shows both source code and program output.
    /// </summary>
    public class ExamplesViewer : UserControl, IComparer<TreeNode>
    {
        private SplitContainer splitContainer1;
        private TreeView tutorialsTree;
        private SplitContainer splitContainer2;
        private RichTextBox sourceTextBox;
        private Panel panel2;
        private Button button1;
        private TableLayoutPanel tableLayoutPanel1;
        private CheckBox timingsCheckBox;
        private CheckBox progressCheckBox;
        private CheckBox factorGraphCheckBox;
        private CheckBox mslCheckBox;
        private CheckBox browserCheckBox;
        private RichTextBox outputTextBox;
        private RichTextBox exampleSummaryBox;
        private CheckBox scheduleCheckBox;
        private Panel panel3;
        private Label label2;
        private ComboBox algorithmComboBox;
        private Label label1;
        private IAlgorithm[] algs;

        /// <summary>
        /// Creates an examples viewer.
        /// </summary>
        internal ExamplesViewer(IAlgorithm[] algs)
        {
            InitializeComponent();
            sourceTextBox.SelectionTabs = new int[] { 20, 40, 60, 80, 100, 120 };
            tutorialsTree.AfterSelect += new TreeViewEventHandler(tutorialsTree_AfterSelect);
            this.algs = algs;
            foreach (IAlgorithm alg in algs) algorithmComboBox.Items.Add(alg.ShortName);
            algorithmComboBox.SelectedIndex = 0; // Array.IndexOf(algs, InferenceEngine.DefaultEngine.Algorithm);
            this.algorithmComboBox.SelectedIndexChanged += new System.EventHandler(this.algorithmComboBox_SelectedIndexChanged);
            progressCheckBox.Checked = InferenceEngine.DefaultEngine.ShowProgress;
            InferenceEngine.DefaultEngine.Compiler.WriteSourceFiles = false;
        }

        /// <summary>
        /// Creates a view of the examples in the same assembly as the supplied type.
        /// </summary>
        /// <param name="exampleType">The type to use to find example classes</param>
        /// <param name="algs"></param>
        public ExamplesViewer(Type exampleType, IAlgorithm[] algs)
            : this(algs)
        {
            ExampleType = exampleType;
        }

        private void tutorialsTree_AfterSelect(object sender, TreeViewEventArgs e)
        {
            OnSelectionChanged();
        }

        /// <summary>
        /// The currently selected example.
        /// </summary>
        public Type SelectedExample
        {
            get
            {
                TreeNode nd = tutorialsTree.SelectedNode;
                if ((nd == null) || (!(nd.Tag is Type))) return null;
                return (Type)nd.Tag;
            }
        }

        /// <summary>
        /// Selection changed handler
        /// </summary>
        protected void OnSelectionChanged()
        {
            Type exampleClass = SelectedExample;
            if (exampleClass == null) return;
            SuspendLayout();
            exampleSummaryBox.Clear();
            exampleSummaryBox.SelectionColor = Color.DarkBlue;
            exampleSummaryBox.SelectionFont = new Font(FontFamily.GenericSansSerif, 11f, FontStyle.Bold);
            exampleSummaryBox.AppendText(exampleClass.Name + Environment.NewLine);
            //label1.Text = "'" + tutorialClass.Name + "' source code";
            ExampleAttribute exa = GetExampleAttribute(exampleClass);
            exampleSummaryBox.SelectionFont = new Font(FontFamily.GenericSansSerif, 10f, FontStyle.Regular);
            exampleSummaryBox.SelectionColor = Color.FromArgb(0, 0, 100);

            string desc = "";
            if (exa != null) desc = exa.Description;
            exampleSummaryBox.AppendText(desc);
            exampleSummaryBox.Size = exampleSummaryBox.GetPreferredSize(exampleSummaryBox.Size);
            exampleSummaryBox.Height += 10;
            exampleSummaryBox.Refresh();
            string filename = GetSourceCodeFilename(exampleClass);
            DoubleBuffered = true;
            //sourceTextBox.SuspendLayout();
            try
            {
                if (filename == null)
                {
                    sourceTextBox.Text = "Example source code was not found.  " + Environment.NewLine +
                                             "Go to the properties of the source file in Visual Studio and set 'Copy To Output Directory' to 'Copy if newer'.";
                }
                else
                {
                    RichTextBox tempBox = new RichTextBox();
                    tempBox.Font = sourceTextBox.Font;
                    tempBox.SelectionTabs = sourceTextBox.SelectionTabs;
                    EfficientTextBox etb = new EfficientTextBox(tempBox);
                    StreamReader sr = new StreamReader(filename);
                    bool isHeader = true;
                    while (true)
                    {
                        string line = sr.ReadLine();
                        if (line == null) break;
                        if (isHeader)
                        {
                            string trimmed = line.TrimStart();
                            if (trimmed.Length == 0 || trimmed.StartsWith("// ")) continue;
                            else isHeader = false;
                        }
                        PrintWithSyntaxHighlighting(etb, line);
                    }
                    etb.Flush();
                    sr.Close();
                    sourceTextBox.Rtf = tempBox.Rtf;
                }
                //sourceTextBox.ResumeLayout(true);
            }
            finally
            {
                ResumeLayout(true);
            }
            //this.PerformLayout();
        }


        private Regex reg = new Regex(@"[\w]+");

        /// <summary>
        /// Very simple syntax highlighting
        /// </summary>
        /// <param name="targetTextBox"></param>
        /// <param name="s"></param>
        internal void PrintWithSyntaxHighlighting(EfficientTextBox targetTextBox, string s)
        {
            if (s.Trim().StartsWith("[Example(")) return;
            if (s.Trim().StartsWith("//"))
            {
                targetTextBox.SelectionColor = Color.Green;
                targetTextBox.AppendText(s + Environment.NewLine);
                targetTextBox.SelectionColor = Color.Black;
                return;
            }
            targetTextBox.SelectionBackColor = Color.White;
            if (s.Contains("//highlight"))
            {
                s = s.Replace("//highlight", "");
                targetTextBox.SelectionBackColor = Color.Yellow;
            }
            MatchCollection mc = reg.Matches(s);
            int ind = 0;
            foreach (Match m in mc)
            {
                targetTextBox.AppendText(s.Substring(ind, m.Index - ind));
                ind = m.Index + m.Length;
                string word = s.Substring(m.Index, m.Length);
                bool reserved = IsReservedWord(word);
                if (reserved) targetTextBox.SelectionColor = Color.Blue;
                if (IsKnownTypeName(word)) targetTextBox.SelectionColor = Color.DarkCyan;
                targetTextBox.AppendText(word);
                targetTextBox.SelectionColor = Color.Black;
            }
            if (ind < s.Length) targetTextBox.AppendText(s.Substring(ind));
            targetTextBox.AppendText(Environment.NewLine);
            //sourceTextBox.AppendText(s + Environment.NewLine);
        }

        private Set<string> knownTypes = new Set<string>();

        private bool IsKnownTypeName(string s)
        {
            if (knownTypes.Count == 0)
            {
                knownTypes.AddRange(new string[]
                    {
                        "Variable", "Range", "InferenceEngine", "ConstantArray",
                        "RandomVariable", "VariableArray", "Given", "GivenArray", "ExpectationPropagation", "VariationalMessagePassing", "Console",
                        "Vector", "VectorGaussian", "IVariableArray", "Gaussian", "Gamma", "Bernoulli", "Rand"
                    });
            }
            return knownTypes.Contains(s);
        }

        private Dictionary<string, bool> reservedSet = new Dictionary<string, bool>();

        private static string[] RESERVED_WORDS =
            {
                "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
                "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else", "enum", "event", "explicit", "extern", "false",
                "finally", "fixed", "float", "for", "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock", "long", "namespace",
                "new", "null", "object", "operator", "out", "override", "params", "private", "protected", "public", "readonly", "ref", "return", "sbyte",
                "sealed", "short", "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong",
                "unchecked", "unsafe", "ushort", "using", "virtual", "volatile", "void", "while"
            };

        private bool IsReservedWord(string word)
        {
            if (reservedSet.Count == 0)
            {
                foreach (string s in RESERVED_WORDS) reservedSet[s] = true;
            }
            return reservedSet.ContainsKey(word);
        }

        private string GetSourceCodeFilename(Type exampleClass)
        {
            // First look in the same folder
            var fileName = Path.GetFileName(exampleClass.Name + ".cs");
            if (File.Exists(fileName))
            {
                return fileName;
            }

            var parentDirectory = Directory.GetParent(Directory.GetCurrentDirectory());
            if (parentDirectory != null)
            {
                // If not, perhaps we are in VisualStudio scenario
                var parentParentDirectory = parentDirectory.Parent;
                if (parentParentDirectory != null)
                {
                    var visualStudioPath = Path.Combine(parentParentDirectory.ToString(), fileName);
                    if (File.Exists(visualStudioPath))
                    {
                        return visualStudioPath;
                    }
                }

                // If not, look in the release folder. This allows running
                // the examples browser from the bin folder in the release. If the
                // release folders change, this must be updated.
                var samplesPath = Path.Combine(parentDirectory.ToString(), "Samples", "C#", "ExamplesBrowser", fileName);
                if (File.Exists(samplesPath))
                {
                    return samplesPath;
                }
            }

            // Source file not found
            return null;
        }

        /// <summary>
        /// Field for ExampleType property
        /// </summary>
        protected Type exampleType;

        /// <summary>
        /// The type to use to find examples - the assembly containing this type will be searched.
        /// </summary>
        public Type ExampleType
        {
            get { return exampleType; }
            set
            {
                exampleType = value;
                OnExampleTypeChanged();
            }
        }


        /// <summary>
        /// Called when the example type changes
        /// </summary>
        protected void OnExampleTypeChanged()
        {
            Type[] types = exampleType.Assembly.GetTypes();
            tutorialsTree.Nodes.Clear();
            tutorialsTree.ShowNodeToolTips = true;

            // To ensure that the known sections will appear in order
            foreach (var categoryName in new[] { "Tutorials", "String tutorials", "Applications" })
            {
                TreeNode categoryNode = tutorialsTree.Nodes.Add(categoryName);
                categoryNode.Tag = categoryName;
                categoryNode.Name = categoryName;
            }

            foreach (Type t in types)
            {
                if (t.GetMethod("Run") == null) continue;
                string category = "Examples";
                ExampleAttribute exa = GetExampleAttribute(t);
                if (exa != null) category = exa.Category;

                // Find an existing category node
                TreeNode par = null;
                foreach (TreeNode nd in tutorialsTree.Nodes)
                {
                    if (category.Equals(nd.Tag)) par = nd;
                }

                if (par == null)
                {
                    // Handle unknown categories
                    par = tutorialsTree.Nodes.Add(category);
                    par.Tag = category;
                    //par.NodeFont = new Font(tutorialsTree.Font, FontStyle.Bold);
                    par.Name = category;
                }

                string name = t.Name;
                if ((exa != null) && (exa.Prefix != null)) name = exa.Prefix + " " + name;
                TreeNode nd2 = par.Nodes.Add(name);
                if (exa != null) nd2.ToolTipText = exa.Description;
                nd2.Tag = t;
            }
            tutorialsTree.ExpandAll();
            foreach (TreeNode nd in tutorialsTree.Nodes)
            {
                SortChildNodesByName(nd);
            }
            if ((tutorialsTree.Nodes.Count > 0) && (tutorialsTree.Nodes[0].Nodes.Count > 0))
            {
                tutorialsTree.SelectedNode = tutorialsTree.Nodes[0].Nodes[0];
            }
        }

        private void SortChildNodesByName(TreeNode nd)
        {
            List<TreeNode> nodes = new List<TreeNode>();
            foreach (TreeNode nd2 in nd.Nodes) nodes.Add(nd2);
            nodes.Sort(this);
            nd.Nodes.Clear();
            nd.Nodes.AddRange(nodes.ToArray());
        }

        /// <summary>
        /// Compare two tree nodes
        /// </summary>
        /// <param name="x">First node</param>
        /// <param name="y">Second node</param>
        /// <returns></returns>
        public int Compare(TreeNode x, TreeNode y)
        {
            return String.Compare(x.Text, y.Text, StringComparison.InvariantCulture);
        }

        private ExampleAttribute GetExampleAttribute(Type exampleClass)
        {
            object[] arr = exampleClass.GetCustomAttributes(typeof(ExampleAttribute), true);
            if ((arr != null) && (arr.Length > 0)) return ((ExampleAttribute)arr[0]);
            return null;
        }

        /// <summary>
        ///  Disposes of the resources (other than memory) used by the <see cref="ContainerControl"/>.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            this.splitContainer1.Dispose();
            this.tutorialsTree.Dispose();
            this.splitContainer2.Dispose();
            this.sourceTextBox.Dispose();
            this.exampleSummaryBox.Dispose();
            this.outputTextBox.Dispose();
            this.tableLayoutPanel1.Dispose();
            this.progressCheckBox.Dispose();
            this.factorGraphCheckBox.Dispose();
            this.browserCheckBox.Dispose();
            this.label1.Dispose();
            this.timingsCheckBox.Dispose();
            this.mslCheckBox.Dispose();
            this.scheduleCheckBox.Dispose();
            this.panel3.Dispose();
            this.label2.Dispose();
            this.algorithmComboBox.Dispose();
            this.panel2.Dispose();
            this.button1.Dispose();
        }

        private void InitializeComponent()
        {
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.tutorialsTree = new System.Windows.Forms.TreeView();
            this.splitContainer2 = new System.Windows.Forms.SplitContainer();
            this.sourceTextBox = new System.Windows.Forms.RichTextBox();
            this.exampleSummaryBox = new System.Windows.Forms.RichTextBox();
            this.outputTextBox = new System.Windows.Forms.RichTextBox();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.progressCheckBox = new System.Windows.Forms.CheckBox();
            this.factorGraphCheckBox = new System.Windows.Forms.CheckBox();
            this.browserCheckBox = new System.Windows.Forms.CheckBox();
            this.label1 = new System.Windows.Forms.Label();
            this.timingsCheckBox = new System.Windows.Forms.CheckBox();
            this.mslCheckBox = new System.Windows.Forms.CheckBox();
            this.scheduleCheckBox = new System.Windows.Forms.CheckBox();
            this.panel3 = new System.Windows.Forms.Panel();
            this.label2 = new System.Windows.Forms.Label();
            this.algorithmComboBox = new System.Windows.Forms.ComboBox();
            this.panel2 = new System.Windows.Forms.Panel();
            this.button1 = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).BeginInit();
            this.splitContainer2.Panel1.SuspendLayout();
            this.splitContainer2.Panel2.SuspendLayout();
            this.splitContainer2.SuspendLayout();
            this.tableLayoutPanel1.SuspendLayout();
            this.panel3.SuspendLayout();
            this.panel2.SuspendLayout();
            this.SuspendLayout();
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.tutorialsTree);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.splitContainer2);
            this.splitContainer1.Size = new System.Drawing.Size(975, 493);
            this.splitContainer1.SplitterDistance = 267;
            this.splitContainer1.TabIndex = 0;
            // 
            // tutorialsTree
            // 
            this.tutorialsTree.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tutorialsTree.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tutorialsTree.HideSelection = false;
            this.tutorialsTree.Location = new System.Drawing.Point(0, 0);
            this.tutorialsTree.Name = "tutorialsTree";
            this.tutorialsTree.Size = new System.Drawing.Size(267, 493);
            this.tutorialsTree.TabIndex = 0;
            // 
            // splitContainer2
            // 
            this.splitContainer2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer2.Location = new System.Drawing.Point(0, 0);
            this.splitContainer2.Name = "splitContainer2";
            this.splitContainer2.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainer2.Panel1
            // 
            this.splitContainer2.Panel1.Controls.Add(this.sourceTextBox);
            this.splitContainer2.Panel1.Controls.Add(this.exampleSummaryBox);
            // 
            // splitContainer2.Panel2
            // 
            this.splitContainer2.Panel2.Controls.Add(this.outputTextBox);
            this.splitContainer2.Panel2.Controls.Add(this.tableLayoutPanel1);
            this.splitContainer2.Panel2.Controls.Add(this.panel2);
            this.splitContainer2.Size = new System.Drawing.Size(704, 493);
            this.splitContainer2.SplitterDistance = 320;
            this.splitContainer2.TabIndex = 0;
            // 
            // sourceTextBox
            // 
            this.sourceTextBox.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.sourceTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.sourceTextBox.Font = new System.Drawing.Font("Courier New", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sourceTextBox.Location = new System.Drawing.Point(0, 96);
            this.sourceTextBox.Name = "sourceTextBox";
            this.sourceTextBox.ReadOnly = true;
            this.sourceTextBox.ShowSelectionMargin = true;
            this.sourceTextBox.Size = new System.Drawing.Size(704, 224);
            this.sourceTextBox.TabIndex = 3;
            this.sourceTextBox.Text = "";
            this.sourceTextBox.WordWrap = false;
            // 
            // exampleSummaryBox
            // 
            this.exampleSummaryBox.BackColor = System.Drawing.SystemColors.Control;
            this.exampleSummaryBox.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.exampleSummaryBox.Dock = System.Windows.Forms.DockStyle.Top;
            this.exampleSummaryBox.Location = new System.Drawing.Point(0, 0);
            this.exampleSummaryBox.Margin = new System.Windows.Forms.Padding(8, 3, 8, 3);
            this.exampleSummaryBox.Name = "exampleSummaryBox";
            this.exampleSummaryBox.ReadOnly = true;
            this.exampleSummaryBox.ScrollBars = System.Windows.Forms.RichTextBoxScrollBars.None;
            this.exampleSummaryBox.ShowSelectionMargin = true;
            this.exampleSummaryBox.Size = new System.Drawing.Size(704, 96);
            this.exampleSummaryBox.TabIndex = 2;
            this.exampleSummaryBox.Text = "";
            // 
            // outputTextBox
            // 
            this.outputTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.outputTextBox.Font = new System.Drawing.Font("Courier New", 9F);
            this.outputTextBox.Location = new System.Drawing.Point(0, 32);
            this.outputTextBox.Name = "outputTextBox";
            this.outputTextBox.ReadOnly = true;
            this.outputTextBox.Size = new System.Drawing.Size(704, 108);
            this.outputTextBox.TabIndex = 4;
            this.outputTextBox.Text = "";
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 9;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.Controls.Add(this.progressCheckBox, 1, 0);
            this.tableLayoutPanel1.Controls.Add(this.factorGraphCheckBox, 2, 0);
            this.tableLayoutPanel1.Controls.Add(this.browserCheckBox, 5, 0);
            this.tableLayoutPanel1.Controls.Add(this.label1, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.timingsCheckBox, 4, 0);
            this.tableLayoutPanel1.Controls.Add(this.mslCheckBox, 6, 0);
            this.tableLayoutPanel1.Controls.Add(this.scheduleCheckBox, 7, 0);
            this.tableLayoutPanel1.Controls.Add(this.panel3, 8, 0);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 140);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 1;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(704, 29);
            this.tableLayoutPanel1.TabIndex = 5;
            // 
            // progressCheckBox 
            // 
            this.progressCheckBox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.progressCheckBox.AutoSize = true;
            this.progressCheckBox.Location = new System.Drawing.Point(47, 7);
            this.progressCheckBox.Margin = new System.Windows.Forms.Padding(3, 3, 3, 1);
            this.progressCheckBox.Name = "progressCheckBox";
            this.progressCheckBox.Size = new System.Drawing.Size(67, 17);
            this.progressCheckBox.TabIndex = 0;
            this.progressCheckBox.Text = "Progress";
            this.progressCheckBox.UseVisualStyleBackColor = true;
            this.progressCheckBox.CheckedChanged += new System.EventHandler(this.progressCheckBox_CheckedChanged);
            // 
            // factorGraphCheckBox
            // 
            this.factorGraphCheckBox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.factorGraphCheckBox.AutoSize = true;
            this.factorGraphCheckBox.Location = new System.Drawing.Point(120, 7);
            this.factorGraphCheckBox.Margin = new System.Windows.Forms.Padding(3, 3, 3, 1);
            this.factorGraphCheckBox.Name = "factorGraphCheckBox";
            this.factorGraphCheckBox.Size = new System.Drawing.Size(86, 17);
            this.factorGraphCheckBox.TabIndex = 0;
            this.factorGraphCheckBox.Text = "Factor graph";
            this.factorGraphCheckBox.UseVisualStyleBackColor = true;
            this.factorGraphCheckBox.CheckedChanged += new System.EventHandler(this.factorGraphCheckBox_CheckedChanged);
            // 
            // browserCheckBox
            // 
            this.browserCheckBox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.browserCheckBox.AutoSize = true;
            this.browserCheckBox.Location = new System.Drawing.Point(280, 7);
            this.browserCheckBox.Margin = new System.Windows.Forms.Padding(3, 3, 3, 1);
            this.browserCheckBox.Name = "browserCheckBox";
            this.browserCheckBox.Size = new System.Drawing.Size(113, 17);
            this.browserCheckBox.TabIndex = 0;
            this.browserCheckBox.Text = "Transform browser";
            this.browserCheckBox.UseVisualStyleBackColor = true;
            this.browserCheckBox.CheckedChanged += new System.EventHandler(this.browserCheckBox_CheckedChanged);
            // 
            // label1
            // 
            this.label1.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(3, 8);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(38, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Show";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // timingsCheckBox
            // 
            this.timingsCheckBox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.timingsCheckBox.AutoSize = true;
            this.timingsCheckBox.Location = new System.Drawing.Point(212, 7);
            this.timingsCheckBox.Margin = new System.Windows.Forms.Padding(3, 3, 3, 1);
            this.timingsCheckBox.Name = "timingsCheckBox";
            this.timingsCheckBox.Size = new System.Drawing.Size(62, 17);
            this.timingsCheckBox.TabIndex = 0;
            this.timingsCheckBox.Text = "Timings";
            this.timingsCheckBox.UseVisualStyleBackColor = true;
            this.timingsCheckBox.CheckedChanged += new System.EventHandler(this.timingsCheckBox_CheckedChanged);
            // 
            // mslCheckBox
            // 
            this.mslCheckBox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.mslCheckBox.AutoSize = true;
            this.mslCheckBox.Location = new System.Drawing.Point(280, 7);
            this.mslCheckBox.Margin = new System.Windows.Forms.Padding(3, 3, 3, 1);
            this.mslCheckBox.Name = "mslCheckBox";
            this.mslCheckBox.Size = new System.Drawing.Size(48, 17);
            this.mslCheckBox.TabIndex = 0;
            this.mslCheckBox.Text = "MSL";
            this.mslCheckBox.UseVisualStyleBackColor = true;
            this.mslCheckBox.CheckedChanged += new System.EventHandler(this.mslCheckBox_CheckedChanged);
            // 
            // scheduleCheckBox
            // 
            this.scheduleCheckBox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.scheduleCheckBox.AutoSize = true;
            this.scheduleCheckBox.Location = new System.Drawing.Point(334, 7);
            this.scheduleCheckBox.Margin = new System.Windows.Forms.Padding(3, 3, 3, 1);
            this.scheduleCheckBox.Name = "scheduleCheckBox";
            this.scheduleCheckBox.Size = new System.Drawing.Size(71, 17);
            this.scheduleCheckBox.TabIndex = 0;
            this.scheduleCheckBox.Text = "Schedule";
            this.scheduleCheckBox.UseVisualStyleBackColor = true;
            this.scheduleCheckBox.CheckedChanged += new System.EventHandler(this.scheduleCheckBox_CheckedChanged);
            // 
            // panel3
            // 
            this.panel3.Controls.Add(this.label2);
            this.panel3.Controls.Add(this.algorithmComboBox);
            this.panel3.Location = new System.Drawing.Point(411, 3);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(200, 23);
            this.panel3.TabIndex = 2;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(4, 5);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(103, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Default algorithm";
            // 
            // algorithmComboBox
            // 
            this.algorithmComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.algorithmComboBox.FormattingEnabled = true;
            this.algorithmComboBox.Location = new System.Drawing.Point(111, 1);
            this.algorithmComboBox.Name = "algorithmComboBox";
            this.algorithmComboBox.Size = new System.Drawing.Size(60, 21);
            this.algorithmComboBox.TabIndex = 3;
            // 
            // panel2
            // 
            this.panel2.Controls.Add(this.button1);
            this.panel2.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel2.Location = new System.Drawing.Point(0, 0);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(704, 32);
            this.panel2.TabIndex = 3;
            // 
            // button1
            // 
            this.button1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.button1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button1.Location = new System.Drawing.Point(0, 0);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(704, 32);
            this.button1.TabIndex = 2;
            this.button1.Text = "Run this example";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // ExamplesViewer
            // 
            this.Controls.Add(this.splitContainer1);
            this.DoubleBuffered = true;
            this.Name = "ExamplesViewer";
            this.Size = new System.Drawing.Size(975, 493);
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.splitContainer2.Panel1.ResumeLayout(false);
            this.splitContainer2.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).EndInit();
            this.splitContainer2.ResumeLayout(false);
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel1.PerformLayout();
            this.panel3.ResumeLayout(false);
            this.panel3.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.ResumeLayout(false);
        }

        private async void button1_Click(object sender, EventArgs e)
        {
            Type tp = SelectedExample;
            if (tp == null) return;
            button1.Enabled = false;
            button1.Text = tp.Name + " running...";
            button1.Refresh();
            outputTextBox.Clear();
            outputTextBox.Refresh();
            await Task.Run(() =>
            {
                TextWriter tw = Console.Out;
                Console.SetOut(new GUITextWriter(this));
                if (tp != null)
                {
                    Stopwatch sw = new Stopwatch();
                    sw.Start();
                    try
                    {
                        Console.WriteLine("====== Output from " + tp.Name + " ======");
                        Console.WriteLine();
                        object obj = Activator.CreateInstance(tp);
                        MethodInfo mi = tp.GetMethod("Run");
                        if (mi != null) mi.Invoke(obj, new object[0]);
                    }
                    catch (Exception ex)
                    {
                        while (ex is TargetInvocationException) ex = ((TargetInvocationException)ex).InnerException;
                        Console.WriteLine("Example failed with exception: " + ex);
                    }
                    if (timingsCheckBox.Checked) Console.WriteLine("Time to run example " + sw.ElapsedMilliseconds + "ms.");
                }
                Console.SetOut(tw);
            });
            button1.Enabled = true;
            button1.Text = "Run this example";
        }

        /// <summary>
        /// Append output text
        /// </summary>
        /// <param name="s"></param>
        public void AppendOutputText(string s)
        {
            outputTextBox.AppendText(s);
            NativeMethods.SendMessage(Handle, WM_VSCROLL, (IntPtr)SB_PAGEBOTTOM, IntPtr.Zero);
            //outputTextBox.sc
            //outputTextBox.Select(outputTextBox.TextLength, 0);
            //outputTextBox.ScrollToCaret();
        }

        private class NativeMethods
        {
            [DllImport("user32.dll", CharSet = CharSet.Auto)]
            public static extern int SendMessage(IntPtr hWnd, int wMsg, IntPtr wParam, IntPtr lParam);
        }

        private const int WM_SCROLL = 276; // Horizontal scroll
        private const int WM_VSCROLL = 277; // Vertical scroll
        private const int SB_LINEUP = 0; // Scrolls one line up
        private const int SB_LINELEFT = 0; // Scrolls one cell left
        private const int SB_LINEDOWN = 1; // Scrolls one line down
        private const int SB_LINERIGHT = 1; // Scrolls one cell right
        private const int SB_PAGEUP = 2; // Scrolls one page up
        private const int SB_PAGELEFT = 2; // Scrolls one page left
        private const int SB_PAGEDOWN = 3; // Scrolls one page down
        private const int SB_PAGERIGTH = 3; // Scrolls one page right
        private const int SB_PAGETOP = 6; // Scrolls to the upper left
        private const int SB_LEFT = 6; // Scrolls to the left
        private const int SB_PAGEBOTTOM = 7; // Scrolls to the upper right
        private const int SB_RIGHT = 7; // Scrolls to the right
        private const int SB_ENDSCROLL = 8; // Ends scroll


        /// <summary>
        /// Run the browser
        /// </summary>
        public void RunBrowser()  // Must not be called "Run"
        {
            Compiler.Visualizers.WindowsVisualizer.FormHelper.RunInForm(this, "Infer.NET Examples Browser", false);
        }

        private void timingsCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            InferenceEngine.DefaultEngine.ShowTimings = timingsCheckBox.Checked;
        }

        private void progressCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            InferenceEngine.DefaultEngine.ShowProgress = progressCheckBox.Checked;
        }

        private void factorGraphCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            InferenceEngine.DefaultEngine.ShowFactorGraph = factorGraphCheckBox.Checked;
            InferenceEngine.DefaultEngine.SaveFactorGraphToFolder = factorGraphCheckBox.Checked ? "." : null;
        }

        private void mslCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            InferenceEngine.DefaultEngine.ShowMsl = mslCheckBox.Checked;
        }

        private void browserCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            InferenceEngine.DefaultEngine.Compiler.BrowserMode = browserCheckBox.Checked ? Compiler.BrowserMode.Always : Compiler.BrowserMode.OnError;
        }

        private void scheduleCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            InferenceEngine.DefaultEngine.ShowSchedule = scheduleCheckBox.Checked;
        }

        private void algorithmComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            InferenceEngine.DefaultEngine.Algorithm = algs[algorithmComboBox.SelectedIndex];
        }
    }

    /// <summary>
    /// Used to redirect console output to a text box
    /// </summary>
    internal class GUITextWriter : TextWriter
    {
        private ExamplesViewer view;

        public delegate void UpdateTextCallback(string text);

        internal GUITextWriter(ExamplesViewer view)
        {
            this.view = view;
        }

        public override void Write(char value)
        {
            if (value != '\r')
            {
                view.Invoke(new UpdateTextCallback(view.AppendOutputText), value.ToString());
            }
        }

        public override Encoding Encoding
        {
            get
            {
                return Encoding.UTF8;
            }
        }
    }

    internal class EfficientTextBox
    {
        private RichTextBox rtb;
        private StringBuilder buffer = new StringBuilder();
        private Color foreColor, backColor;
        private Font font;
        private Color foreColorOfBuffer, backColorOfBuffer;
        private Font fontOfBuffer;

        internal EfficientTextBox(RichTextBox rtb)
        {
            this.rtb = rtb;
            foreColor = rtb.SelectionColor;
            backColor = rtb.SelectionBackColor;
            font = rtb.SelectionFont;
            foreColorOfBuffer = foreColor;
            backColorOfBuffer = backColor;
            fontOfBuffer = font;
        }

        internal Color SelectionColor
        {
            get { return foreColor; }
            set { foreColor = value; }
        }

        internal Color SelectionBackColor
        {
            get { return backColor; }
            set { backColor = value; }
        }

        internal Font SelectionFont
        {
            get { return font; }
            set { font = value; }
        }

        internal void Flush()
        {
            rtb.AppendText(buffer.ToString());
            buffer.Length = 0;
            // comparing backColor to rtb.SelectionBackColor does not work since rtb.SelectionBackColor does not always return the same value that was set.
            // foreColor and font have the same problem.
            if (backColor != backColorOfBuffer)
            {
                rtb.SelectionBackColor = backColor;
                backColorOfBuffer = backColor;
            }
            if (foreColor != foreColorOfBuffer)
            {
                rtb.SelectionColor = foreColor;
                foreColorOfBuffer = foreColor;
            }
            if (font != fontOfBuffer)
            {
                rtb.SelectionFont = font;
                fontOfBuffer = font;
            }
        }

        internal void AppendText(string s)
        {
            if ((backColor != backColorOfBuffer) || (foreColor != foreColorOfBuffer) || (font != fontOfBuffer)) Flush();
            buffer.Append(s);
        }
    }
}
---
layout: default 
--- 
[Infer.NET user guide](index.md)

## Transform browser

The transform browser shows the output of each stage of the Infer.NET model compiler, starting from [the Model Specification Language](The Model Specification Language.md) and ending at the [generated code](Structure of generated inference code.md).  It is useful for understanding why a model fails to compile, why the generated code is less efficient than expected, how the compiler works, and debugging the compiler.  

The transform browser is enabled by setting the `BrowserMode` property of `InferenceEngine.Compiler`.  The next time that `InferenceEngine` compiles a model, the browser will appear in a new window.  To compare the transform outputs across different runs of the compiler, use the `BrowserMode.WriteFiles` option.  The output files can be compared across runs using any file comparison utility.

The appearance of the transform browser depends on the `InferenceEngine.Visualizer` property.  By default, the browser appears in a web page.  On the left hand side is a list of transforms.  Selecting a transform changes the code shown on the right hand side.

When using `WindowsVisualizer`, the transform browser is a custom window with multiple parts:

1. The top of the window has a series of buttons, one for each transform.  Press the button to show the output of that transform.  Only transforms that modified the code are shown.
1. The search bar lets you filter the code.  Only the lines of code that contain some word in the query string will be shown, with matches highlighted.  To only show exact matches of a word, enclose part of the query in double quotes.  Query text inside double quotes must consist of letters and numbers only.  Clicking on a word in the code adds it to the query string. The query string persists across transforms.  To select the query string, double click in the search box or type Ctrl+A.  To clear the query string, select it then press Delete, or click the X button.  Right click gives a Cut/Copy/Paste menu.
1. The code window displays the output of the selected transform.  Right click on a line of code to get a menu.  "Show attributes" opens a pane on the right listing the attributes attached to the line of code by the compiler.  This includes attributes attached to sub-expressions of the line, and attributes attached to enclosing blocks.  The attributes pane will stay open and update to show the attributes of the currently selected statement.  The X button closes the attributes pane.
1. Some transforms output additional information besides the code.  In this case, buttons will appear at the bottom of the code window, to navigate between the different outputs.
1. If the transform raised errors, they will be shown at the bottom in an error window.  Selecting an error will scroll the code window to the offending code element.


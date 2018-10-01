% Licensed to the .NET Foundation under one or more agreements.
% The .NET Foundation licenses this file to you under the MIT license.
% See the LICENSE file in the project root for more information.
function FAresults
W = load('../TestApp/faresultsW.txt');
%W = load('../TestApp/faresultsWvec.txt')';
X = load('../TestApp/faresultsX.txt');
load ../Tests/Data/digits.txt
num=16;
draw(digits(1:num,:),'Original',1);
draw(W,'W',2);
D = X*W;
draw(D(1:num,:),'Reconstructed',3);




function draw(W,titleString,fignum)
figure(fignum);clf;
sq = ceil(sqrt(size(W,1)));
for j=1:size(W,1)
  Wim = reshape(W(j,:),[8 8])';
  subplot(sq,sq,j);
  imagesc(Wim); axis image off;
  colormap(gray);
  %    colorbar;
  if (mod(j,sq)==0) drawnow; end;
  if(j==1) title(titleString); end;
end



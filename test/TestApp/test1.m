load('bin/DebugFull/net472/points.csv')

if 0
load derivs.csv
points = points(1:end-1,:);
quiver(points(:,1),points(:,2),derivs(:,1),derivs(:,2))
draw_line_clip(1,0,1)
else
	i = (1:50)+1000;
	plot(points(i,1),points(i,2),'.-')
	axis square
end

load points.csv
load derivs.csv
points = points(1:end-1,:);
quiver(points(:,1),points(:,2),derivs(:,1),derivs(:,2))
draw_line_clip(1,0,1)

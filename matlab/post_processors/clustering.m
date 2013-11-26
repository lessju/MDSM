addpath('/home/lessju/Code/MDSM/matlab/rfi_filters');

polyfit();

% Load data
S = load('test_data.mat');
dedisped_data = S.dedisped_data;
thresh = 4;

% Calculate sum and stddev for all dedisperd power series
mean   = mean2(dedisped_data);
stddev = std2(dedisped_data);

% Apply threshold
[dm, time] = find(dedisped_data > mean + stddev * thresh);
vals       = dedisped_data(dedisped_data > mean + stddev * thresh);

% Massage data for clustering
values = zeros(size(vals, 1), 3);
for i = 1 : size(values, 1)
    values(i, :) = [dm(i) time(i) vals(i)];
end
   
% Cluster (k-means)

%clusters = 6;
%[idx,ctrs] = kmeans(values, clusters, 'distance', 'cityblock');
%figure;
%hold on
%colours = ['r.' 'b.' 'g.' 'y.' 'm.' 'c.' 'w.' 'k.'];
%for i = 1 : clusters    
%    scatter3(values(idx==i,1), values(idx==i,2), values(idx==i, 3), colours(i))
%end
%scatter3(ctrs(:,1), ctrs(:,2), ctrs(:,3), 'kx','MarkerSize',12,'LineWidth',2)

% Cluster (dbscan)
tic
[class, type] = dbscan(values, 5, 20);
toc

figure;
hold on
colours = ['r.' 'b.' 'g.' 'y.' 'm.' 'c.' 'w.' 'k.'];
clusters = unique(class);
for i = 1 : size(clusters, 2)
   cluster = clusters(i);
   scatter3(values(class==cluster,1), values(class==cluster,2), values(class==cluster, 3), colours(mod(size(colours, 1), i) + 1));
end
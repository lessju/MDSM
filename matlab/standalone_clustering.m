addpath('/home/lessju/Code/MDSM/matlab/post_processors');

%% Load CSV file for processing
% filename = '/data/Data/Medicina/B0329+54_Nov_08/B0329+54_beam_0_2012-11-07_23:22:49.dat';
% fid = fopen(filename,'r');
% 
% disp('Reading CSV Data');
% entries{1} = textscan(fid, '%f', 3, 'delimiter', ',');
% i = 1;
% while (~feof(fid))
%     i = i + 1;
%     entries{i} = textscan(fid, '%f', 3, 'delimiter', ',');
% end
% 
% disp('Reformatting data');
% number = size(entries, 2);
% values = zeros(number, 3);
% for i=1:size(values,1)-1
%     a = entries{i}{1}(1);
%     b = entries{i}{1}(2);
%     c = entries{i}{1}(3);
%     values(i,:) = [a b c];
% end

% Normalise data
% minA = min(values(:,1));
% maxA = max(values(:,1));
% minB = min(values(:,2));
% maxB = max(values(:,2));
% minC = min(values(:,3));
% maxC = max(values(:,3));
% 
% values(:,1) = (values(:,1) - minA) / (maxA - minA);
% values(:,2) = (values(:,2) - minB) / (maxB - minB);
% values(:,3) = (values(:,3) - minB) / (maxC - minC);


%% Start clustering
disp('Started DBSCAN');
tic
[class, type] = dbscan(values(1:8192, :), 20, 0.5);
toc

figure;
hold on
colours = ['r.' 'b.' 'g.' 'y.' 'm.' 'c.' 'w.' 'k.'];
clusters = unique(class);
for i = 1 : size(clusters, 2)
   cluster = clusters(i);
   scatter3(values(class==cluster,1), values(class==cluster,2), values(class==cluster, 3), colours(mod(size(colours, 1), i) + 1));
end
hold off
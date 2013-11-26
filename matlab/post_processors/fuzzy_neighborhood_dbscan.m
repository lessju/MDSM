function [class,type]=fuzzy_neighborhood_dbscan(X, MinCard, Eps)

    function [card, ind] = linear_fuzzy_neighborhood(i, X, dmax, alpha)
        
        % Find distance to all other points
        x          = repmat(X(i,:), size(X, 1), 1);
        membership = 1 - sqrt(sum((x - X ) .^ 2, 2)) / dmax;
        ind        = find(membership <= alpha);
        card = sum(membership(ind));
    end

[num_points, ndims] = size(X);
assigned = zeros(num_points, 1);
type     = zeros(num_points, 1);
class    = zeros(num_points, 1);

% Get minimum and maximum value in each dimension
mins = min(X);
maxs = max(X);

% Normalise data to get an epsilon value between [0,1]
for i=1:num_points
    X(i,:) = (X(i,:) - mins) ./ ((maxs - mins) .* sqrt(ndims));
end

dmax = 1;

% Loop over all data points
curr_cluster = 1;
for i = 1 : num_points
    
    point = X(i,:);
    
    % If data point is already assigned, skip
    if assigned(i)
        continue
    end
    
    % Get all neighbouring 
    [cardinality ind] = linear_fuzzy_neighborhood(i, X, dmax, Eps);
    
%        if length(ind)>=k+1; 
%           type(i)=1;
%           class(ind)=ones(length(ind),1)*max(no);
%           
%           while ~isempty(ind)
%                 ob=x(ind(1),:);
%                 touched(ind(1))=1;
%                 ind(1)=[];
%                 D=dist(ob(2:n),x(:,2:n));
%                 i1=find(D<=Eps);
%      
%                 if length(i1)>1
%                    class(i1)=no;
%                    if length(i1)>=k+1;
%                       type(ob(1))=1;
%                    else
%                       type(ob(1))=0;
%                    end
% 
%                    for i=1:length(i1)
%                        if touched(i1(i))==0
%                           touched(i1(i))=1;
%                           ind=[ind i1(i)];   
%                           class(i1(i))=no;
%                        end                    
%                    end
%                 end
%           end
%           no=no+1; 
%        end
    
end

end

% 
% [m,n]=size(x);
% 
% if nargin<3 || isempty(Eps)
%    [Eps]=epsilon(x,k);
% end
% 
% x=[[1:m]' x];
% [m,n]=size(x);
% type=zeros(1,m);
% no=1;
% touched=zeros(m,1);
% 
% 
% for i=1:m
%     if touched(i)==0;
%        ob=x(i,:);
%        D=dist(ob(2:n),x(:,2:n));
%        ind=find(D<=Eps);
%     
%        if length(ind)>1 && length(ind)<k+1       
%           type(i)=0;
%           class(i)=0;    
%        end
%        if length(ind)==1
%           type(i)=-1;
%           class(i)=-1;  
%           touched(i)=1;
%        end
% 
%        if length(ind)>=k+1; 
%           type(i)=1;
%           class(ind)=ones(length(ind),1)*max(no);
%           
%           while ~isempty(ind)
%                 ob=x(ind(1),:);
%                 touched(ind(1))=1;
%                 ind(1)=[];
%                 D=dist(ob(2:n),x(:,2:n));
%                 i1=find(D<=Eps);
%      
%                 if length(i1)>1
%                    class(i1)=no;
%                    if length(i1)>=k+1;
%                       type(ob(1))=1;
%                    else
%                       type(ob(1))=0;
%                    end
% 
%                    for i=1:length(i1)
%                        if touched(i1(i))==0
%                           touched(i1(i))=1;
%                           ind=[ind i1(i)];   
%                           class(i1(i))=no;
%                        end                    
%                    end
%                 end
%           end
%           no=no+1; 
%        end
%    end
% end
% 
% i1=find(class==0);
% class(i1)=-1;
% type(i1)=-1;

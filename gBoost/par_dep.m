function [bin_pred, bin_counts, pd_diff, pd_stat] = par_dep(pred, vals1, vals2, nbins)
    nbins1 = min(nbins, length(unique(vals1)));
    nbins2 = min(nbins, length(unique(vals2)));
    [bin_vals1, bin_counts1, cuts1] = bin_vals(vals1, pred, nbins1);
    [bin_vals2, bin_counts2, cuts2] = bin_vals(vals2, pred, nbins2);

    [svals, sidx] = sortrows([vals1 vals2]);
    start = 1;
    bin_pred = nan(nbins2, nbins1);
    bin_counts = zeros(nbins2, nbins1);
    
    for i = 2:nbins1
        new_start = find(svals(start:end, 1) >= cuts1(i), 1, 'first');
        % new_start shouldn't be empty given how cuts one was created
        assert(~isempty(new_start)); 
        new_start = start - 1 + new_start;
        % Now split the predictions across the bins for vals2 for
        % the given bin of vals1.
        [bin_pred(:, i - 1), bin_counts(:, i - 1)] = bin_vals(svals(start:(new_start - 1), 2), ...
                                                          pred(sidx(start:(new_start - 1))), cuts2);
        start = new_start;
    end
    if start < length(pred) + 1
        [bin_pred(:, end), bin_counts(:, end)] = bin_vals(svals(start:end, ...
                                                      2), pred(sidx(start:end)), cuts2);
    end
    
    pd_diff = zeros(nbins2, nbins1);
    for i = 1:nbins1
        for j = 1:nbins2
            pd_diff(j, i) = (bin_pred(j, i) - bin_vals1(i) - bin_vals2(j))^2;
        end
    end
    pd_stat = nansum(pd_diff(:) .* bin_counts(:)) / nansum(bin_pred(:).^2 .* bin_counts(:));
end

function [bin_pred, bin_counts, cuts] = bin_vals(vals, pred, nbins)
% Bins the values in vals and computes the average of pred in each bin.
% vals and pred are assumed to be matched so the prediction for the i-th
% value of vals is pred(i).
% If nbins has length > 1, then it is assumed to be the set of 
% breaks for binning. These MUST be sorted in ascending order.

[svals, idx] = sort(vals); % Sort the values for easy binning.
if length(nbins) > 1 % nbins is the set of breaks
    cuts = nbins;
    nbins = length(cuts) - 1; % One more break than bins
else
    cuts = linspace(svals(1), svals(end), nbins + 1); % One more break than bins
end

bin_pred = nan(nbins, 1);
bin_counts = zeros(nbins, 1);
start = 1;

for i = 2:nbins
    % This will be the beginning of the values for the next bin.
    new_start = find(svals(start:end) >= cuts(i), 1, 'first');
    if isempty(new_start)
        % There is no value greater than the current cutoff. 
        % All remaining values go to this bin.
        new_start = length(pred) + 1;
    else
        new_start = start - 1 + new_start; 
    end
    bin_counts(i - 1) = new_start - start;
    bin_pred(i - 1) = nanmean(pred(idx(start:(new_start - 1))));
    start = new_start;
end
if start < length(pred) + 1
    bin_counts(end) = length(pred) - start + 1;
    bin_pred(end) = nanmean(pred(idx(start:end)));
end
end
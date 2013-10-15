function [bin_pred, bin_counts, pd_diff, pd_stat] = par_dep(pred, vals1, vals2, thresh1, thresh2)
%[bin_vals1, ~, cuts1] = bin_vals(vals1, pred, nbins);
%[bin_vals2, ~, cuts2] = bin_vals(vals2, pred, nbins);
thresh1 = [min(vals1); thresh1; max(vals1)];
thresh2 = [min(vals2); thresh2; max(vals2)];
cuts1 = cuts_from_thresh(thresh1, 0.001);
cuts2 = cuts_from_thresh(thresh2, 0.001);
bin_vals1 = count_at_bins(vals1, pred, cuts1);
bin_vals2 = count_at_bins(vals2, pred, cuts2);
nbins1 = length(bin_vals1);
nbins2 = length(bin_vals2);
%assert(all(~isnan(bin_vals1)));
%assert(all(~isnan(bin_vals2)));

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
  [bin_pred(:, i - 1), bin_counts(:, i - 1)] = count_at_bins(svals(start:(new_start - 1), 2), ...
    pred(sidx(start:(new_start - 1))), cuts2);
  start = new_start;
end
if start < length(pred) + 1
  [bin_pred(:, end), bin_counts(:, end)] = count_at_bins(svals(start:end, 2), ...
    pred(sidx(start:end)), cuts2);
end

pd_diff = zeros(nbins2, nbins1);
for i = 1:nbins1
  for j = 1:nbins2
    if isnan(bin_pred(j, i))
      pd_diff(j, i) = 0;
    else
      pd_diff(j, i) = (bin_pred(j, i) - bin_vals1(i) - bin_vals2(j))^2;
    end
  end
end
pd_stat = nansum(pd_diff(:) .* bin_counts(:)) / nansum(bin_pred(:).^2 .* bin_counts(:));
end

function [bin_pred, bin_counts, cuts] = bin_vals(vals, pred, nbins)
% Bins the values in vals into bins with the same number of elements and computes 
% the average of pred in each bin.
% vals and pred are assumed to be matched so the prediction for the i-th
% value of vals is pred(i).

[svals, idx] = sort(vals); % Sort the values for easy binning.

bin_pred = nan(nbins, 1);
bin_counts = zeros(nbins, 1);
cuts = zeros(nbins + 1, 1);
cuts(1) = svals(1);
nvals = length(svals);
vals_per_bin = ceil(nvals / nbins);
start = 1;
i = 0;

while start < nvals + 1
  new_start = min(start + vals_per_bin, nvals + 1); % start of next bin
  if new_start < nvals + 1 && svals(new_start) == svals(new_start - 1)
    % Extend the bin to include all the elements equal to the last
    tmp_start = find(svals(start:end) > svals(new_start - 1), 1, 'first');
    if isempty(tmp_start)
      new_start = nvals + 1;
    else
      new_start = start - 1 + tmp_start;
    end
  end
  i = i + 1;
  if new_start < nvals + 1
    cuts(i + 1) = svals(new_start);
  else
    cuts(i + 1) = svals(end);
  end
  bin_counts(i) = new_start - start;
  bin_pred(i) = nanmean(pred(idx(start:(new_start - 1))));
  start = new_start;
end
bin_pred = bin_pred(1:i);
bin_counts = bin_counts(1:i);
cuts = cuts(1:(i+1));
end

function [bin_pred, bin_counts, cuts] = bin_vals_linspace(vals, pred, nbins)
% Bins the values in vals into uniformly spaced bins and computes the average of pred in each bin.
% vals and pred are assumed to be matched so the prediction for the i-th
% value of vals is pred(i).
% If nbins has length > 1, then it is assumed to be the set of 
% breaks for binning. These MUST be sorted in ascending order.

if length(nbins) > 1 % nbins is the set of breaks
    cuts = nbins;
else
    cuts = linspace(min(vals), max(vals), nbins + 1); % One more break than bins
end

[bin_pred, bin_counts] = count_at_bins(vals, pred(idx), cuts);

end

function [bin_pred, bin_counts] = count_at_bins(vals, pred, cuts)
% svals and cuts must be sorted in ascending order. 

[svals, idx] = sort(vals); % Sort the values for easy binning.
pred = pred(idx);

nbins = length(cuts) - 1;
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
    bin_pred(i - 1) = nanmean(pred(start:(new_start - 1)));
    start = new_start;
end
if start < length(pred) + 1
    bin_counts(end) = length(pred) - start + 1;
    bin_pred(end) = nanmean(pred(start:end));
end
end

function cuts = cuts_from_thresh(thresh, eps)
sthresh = sort(thresh);
cuts = [sthresh(1); sthresh(find(diff(sthresh) > eps) + 1)];
end
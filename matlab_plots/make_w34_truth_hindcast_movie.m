function make_w34_truth_hindcast_movie(ncFile, outMovie, varargin)
%MAKE_W34_TRUTH_HINDCAST_MOVIE Create a side-by-side W34 truth/hindcast MP4.
%
% Usage:
%   make_w34_truth_hindcast_movie('w34_heatcast_ens_stack.nc')
%
% Optional name-value arguments:
%   'FrameStep'    : plot every Nth time slice. Default: 1
%   'FrameRate'    : movie frame rate. Default: 8
%   'StartIndex'   : first time index. Default: 1
%   'EndIndex'     : last time index. Default: all times
%   'CLim'         : color limits for z-score fields. Default: [-2.5 2.5]
%   'TruthVar'     : NetCDF truth variable. Default: 'ground_truth_3d'
%   'HindcastVar'  : NetCDF hindcast variable. Default: 'model_output_3d'
%
% The script reads one time slice at a time. Output arrays are displayed as
% latitude x longitude maps with time as the third dimension.

if nargin < 1 || isempty(ncFile)
    ncFile = 'w34_heatcast_ens_stack.nc';
end
if nargin < 2 || isempty(outMovie)
    outMovie = fullfile('matlab_plots', 'outputs', 'w34_truth_hindcast_movie.mp4');
end

p = inputParser;
addParameter(p, 'FrameStep', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'FrameRate', 8, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'StartIndex', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'EndIndex', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
addParameter(p, 'CLim', [-2.5 2.5], @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'TruthVar', 'ground_truth_3d', @(x) ischar(x) || isstring(x));
addParameter(p, 'HindcastVar', 'model_output_3d', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;

ncFile = char(ncFile);
ncFile = resolveNetcdfPath(ncFile);
outMovie = char(outMovie);
truthVar = char(opt.TruthVar);
hindcastVar = char(opt.HindcastVar);

time = ncread(ncFile, 'time');
targetDate = ncread(ncFile, 'target_date_yyyymmdd');
nt = numel(time);
if isempty(opt.EndIndex)
    endIndex = nt;
else
    endIndex = min(nt, round(opt.EndIndex));
end
startIndex = max(1, round(opt.StartIndex));
frameStep = max(1, round(opt.FrameStep));

lat = ncread(ncFile, 'lat');
lon = ncread(ncFile, 'lon');

truthInfo = ncinfo(ncFile, truthVar);
truthSize = truthInfo.Size;
if numel(truthSize) ~= 3
    error('%s must be a 3-D variable.', truthVar);
end

% Read first slice to determine MATLAB's ncread orientation.
truth0 = squeeze(ncread(ncFile, truthVar, [1 1 startIndex], [Inf Inf 1]));
hind0 = squeeze(ncread(ncFile, hindcastVar, [1 1 startIndex], [Inf Inf 1]));
[latPlot, lonPlot] = orientGridToField(lat, lon, truth0);

lonLim = [min(lonPlot(:), [], 'omitnan'), max(lonPlot(:), [], 'omitnan')];
latLim = [min(latPlot(:), [], 'omitnan'), max(latPlot(:), [], 'omitnan')];

outDir = fileparts(outMovie);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end

writer = VideoWriter(outMovie, 'MPEG-4');
writer.FrameRate = opt.FrameRate;
writer.Quality = 95;
open(writer);
cleanupObj = onCleanup(@() close(writer));

fig = figure('Color', 'w', 'Position', [100 100 1600 720]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
colormap(fig, blueWhiteRed(256));

axTruth = nexttile(tl, 1);
hTruth = imagesc(axTruth, lonLim, latLim, truth0);
set(axTruth, 'YDir', 'normal');
axis(axTruth, 'image');
caxis(axTruth, opt.CLim);
colorbar(axTruth);
title(axTruth, 'Observed W34 truth');
xlabel(axTruth, 'Longitude');
ylabel(axTruth, 'Latitude');
set(hTruth, 'AlphaData', isfinite(truth0));

axHind = nexttile(tl, 2);
hHind = imagesc(axHind, lonLim, latLim, hind0);
set(axHind, 'YDir', 'normal');
axis(axHind, 'image');
caxis(axHind, opt.CLim);
colorbar(axHind);
title(axHind, 'HeatCast W34 hindcast');
xlabel(axHind, 'Longitude');
ylabel(axHind, 'Latitude');
set(hHind, 'AlphaData', isfinite(hind0));

for t = startIndex:frameStep:endIndex
    truth = squeeze(ncread(ncFile, truthVar, [1 1 t], [Inf Inf 1]));
    hindcast = squeeze(ncread(ncFile, hindcastVar, [1 1 t], [Inf Inf 1]));

    set(hTruth, 'CData', truth, 'AlphaData', isfinite(truth));
    set(hHind, 'CData', hindcast, 'AlphaData', isfinite(hindcast));

    dateText = dateLabel(targetDate(t));
    title(axTruth, sprintf('Observed W34 truth | %s', dateText));
    title(axHind, sprintf('HeatCast W34 hindcast | %s', dateText));
    sgtitle(tl, sprintf('W34 continuous z-score fields | frame %d of %d', t, nt), 'FontWeight', 'bold');

    drawnow;
    writeVideo(writer, getframe(fig));
end

fprintf('Wrote movie: %s\n', outMovie);

end

function ncFile = resolveNetcdfPath(ncFile)
% Resolve common local/HPC export locations and fail with useful candidates.
if isfile(ncFile)
    return;
end

[~, name, ext] = fileparts(ncFile);
if isempty(ext)
    ext = '.nc';
end

candidates = {
    ncFile
    fullfile('matlab_plots', [name ext])
    fullfile('matlab_exports', [name ext])
    fullfile('.', [name ext])
    fullfile('matlab_plots', 'w34_heatcast_ens_stack.nc')
    fullfile('matlab_exports', 'w34_heatcast_ens_stack.nc')
    fullfile('.', 'w34_heatcast_ens_stack.nc')
    };

for i = 1:numel(candidates)
    if isfile(candidates{i})
        ncFile = candidates{i};
        fprintf('Using NetCDF file: %s\n', ncFile);
        return;
    end
end

matches = [dir('*.nc'); dir(fullfile('matlab_plots', '*.nc')); dir(fullfile('matlab_exports', '*.nc'))];
fprintf('Could not open requested NetCDF: %s\n', ncFile);
if ~isempty(matches)
    fprintf('Available NetCDF candidates:\n');
    for i = 1:numel(matches)
        fprintf('  %s\n', fullfile(matches(i).folder, matches(i).name));
    end
end
error('NetCDF file not found. Pass one of the available paths printed above.');
end

function [latOut, lonOut] = orientGridToField(lat, lon, field)
% Match lat/lon orientation to a displayed 2-D field.
if isequal(size(lat), size(field)) && isequal(size(lon), size(field))
    latOut = lat;
    lonOut = lon;
elseif isequal(size(lat'), size(field)) && isequal(size(lon'), size(field))
    latOut = lat';
    lonOut = lon';
else
    error('lat/lon size %s does not match field size %s.', mat2str(size(lat)), mat2str(size(field)));
end
end

function s = dateLabel(value)
value = double(value);
if isfinite(value) && value > 10000000
    txt = sprintf('%08d', round(value));
    s = sprintf('%s-%s-%s', txt(1:4), txt(5:6), txt(7:8));
else
    s = sprintf('time index %g', value);
end
end

function cmap = blueWhiteRed(n)
if nargin < 1
    n = 256;
end
n = max(2, round(n));
half = floor(n / 2);
blue = [0.103, 0.318, 0.639];
white = [1, 1, 1];
red = [0.698, 0.094, 0.168];
lower = [linspace(blue(1), white(1), half)', linspace(blue(2), white(2), half)', linspace(blue(3), white(3), half)'];
upper = [linspace(white(1), red(1), n - half)', linspace(white(2), red(2), n - half)', linspace(white(3), red(3), n - half)'];
cmap = [lower; upper];
end

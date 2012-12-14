function varargout = RFIForm(varargin)
% RFIFORM M-file for RFIForm.fig
%      RFIFORM, by itself, creates a new RFIFORM or raises the existing
%      singleton*.
%
%      H = RFIFORM returns the handle to a new RFIFORM or the handle to
%      the existing singleton*.
%
%      RFIFORM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RFIFORM.M with the given input arguments.
%
%      RFIFORM('Property','Value',...) creates a new RFIFORM or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RFIForm_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RFIForm_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RFIForm

% Last Modified by GUIDE v2.5 22-Aug-2012 15:20:57

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RFIForm_OpeningFcn, ...
                   'gui_OutputFcn',  @RFIForm_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before RFIForm is made visible.
function RFIForm_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for RFIForm
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% Initialise UI
initialize_gui(hObject, handles, false);

setappdata(handles.rfi_panel, 'rfi', struct());
if length(varargin) < 2
    disp('Form requires center frequency and bandwidth as input parameters');
    close;
else
    setappdata(handles.rfi_panel, 'center_frequency', varargin{1});
    setappdata(handles.rfi_panel, 'bandwidth', varargin{2});
    uiwait(handles.figure1);
end

% --- Outputs from this function are returned to the command line.
function varargout = RFIForm_OutputFcn(hObject, eventdata, handles)

% Get default command line output from handles structure
if isstruct(handles)
    rfi = getappdata(handles.rfi_panel, 'rfi');
    if isstruct(rfi) && ~isempty(fieldnames(rfi))
        varargout{1} = getappdata(handles.rfi_panel, 'rfi');
    else
        varargout{1} = [];
    end
end

% Object no longer needed
if ~isempty(handles)
    delete(hObject);
end


% --------------------------------------------------------------------
function initialize_gui(fig_handle, handles, isreset)

if isfield(handles, 'metricdata') && ~isreset
    return;
end

% Update handles structure
guidata(handles.figure1, handles);

% --- Executes on button press in channel_rfi_checkbox.
function channel_rfi_checkbox_Callback(hObject, eventdata, handles)

if get(hObject, 'Value') == 0.0
    set(handles.text12, 'Enable', 'off');
    set(handles.text13, 'Enable', 'off');
    set(handles.channel_rfi_freqs_edit, 'Enable', 'off');
    set(handles.channel_rfi_snr_edit, 'Enable', 'off');
else
    set(handles.text12, 'Enable', 'on');
    set(handles.text13, 'Enable', 'on');
    set(handles.channel_rfi_freqs_edit, 'Enable', 'on');
    set(handles.channel_rfi_snr_edit, 'Enable', 'on');
end
    

% --- Executes on button press in rfi_spike_checkbox.
function rfi_spike_checkbox_Callback(hObject, eventdata, handles)

if get(hObject, 'Value') == 0.0
    set(handles.text14, 'Enable', 'off');
    set(handles.text15, 'Enable', 'off');
    set(handles.rfi_spike_freq_edit, 'Enable', 'off');
    set(handles.rfi_spike_params_edit, 'Enable', 'off');
else
    set(handles.text14, 'Enable', 'on');
    set(handles.text15, 'Enable', 'on');
    set(handles.rfi_spike_freq_edit, 'Enable', 'on');
    set(handles.rfi_spike_params_edit, 'Enable', 'on');
end


% --- Executes on button press in done_rfi_button.
function done_rfi_button_Callback(hObject, eventdata, handles)

% Read all user input and generate parameter struct
rfi = struct();

center_frequency = getappdata(handles.rfi_panel, 'center_frequency');
bandwidth = getappdata(handles.rfi_panel, 'bandwidth');

% Collect channel RFI parameters
chan_freqs = []; chan_snr = [];
rfi.chan_freqs = chan_freqs; rfi.chan_snr = chan_snr;
if get(handles.channel_rfi_checkbox, 'Value') == 1.0
    eval(['chan_freqs = [' get(handles.channel_rfi_freqs_edit, 'String') '] * 1e6;']);
    eval(['chan_snr   = [' get(handles.channel_rfi_snr_edit, 'String') '] * 1e6;']);
    
    if size(chan_freqs) ~= size(chan_snr)
        msgbox(['Invalid Channel-RFI options. Number of items in "frequency channels"'...
                'and channel RFI SNR should be the same'], 'Input Error', 'error');
        return;
    end
    
   if (size(chan_freqs(chan_freqs < center_frequency - bandwidth/2),2) > 0) || ...
      (size(chan_freqs(chan_freqs > center_frequency + bandwidth/2),2) > 0)
       msgbox('Invalid Channel-RFI options, frequency out of band', ...
               'Input Error', 'error');
       return;
   end
    
    % Save RFI parameters to appdata
    if size(chan_freqs,2) >= 1
        x = struct('frequency',chan_freqs(1), 'chan_snr', chan_snr(1));
        channel_rfi = [x];
        for j=2:size(chan_freqs, 2)
            channel_rfi(j) = struct('frequency', chan_freqs(j), 'chan_snr', chan_snr(j));
        end
    end

    rfi.chan_freqs = chan_freqs;
    rfi.chan_snr   = chan_snr;
end

% Collect RFI spikes parameters
if get(handles.rfi_spike_checkbox, 'Value') == 1.0
    num_spikes = str2num(get(handles.rfi_spike_freq_edit, 'String'));
    eval(['spike_snr = [' get(handles.rfi_spike_params_edit, 'String') '];']);
    
    if size(spike_snr, 2) ~= 2 && size(spike_snr) ~= []
        msgbox('Invalid SNR spikes parameters.', 'Input Error', 'error');
        return;
    end
    
    if size(num_spikes,2) == []
        msgbox('Invalid number of SNR spikes.', 'Input Error', 'error');
        return;
    end
    
    % Save RFI parameters to appdata
    rfi.num_spikes = num_spikes;
    rfi.spike_snr  = spike_snr;
    
else
    rfi.num_spikes = 0;    
    rfi.spike_snr  = 0;
end

setappdata(handles.rfi_panel, 'rfi', rfi);
close; 


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)

if isequal(get(hObject, 'waitstatus'), 'waiting')
    % The GUI is still in UIWAIT, us UIRESUME
    uiresume(hObject);
else
    % The GUI is no longer waiting, just close it
    delete(hObject);
end

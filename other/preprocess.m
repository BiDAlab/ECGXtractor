% preproc_fir3_complete_third

list = dir(); % enter your path in the brackets
list = list(3:length(list)-1);

Fs = 500;

% Low pass filter
Fstop = 120;
Fpass = 90;
Dstop = 0.0001;
Dpass = 0.05;
F = [0 Fpass Fstop Fs/2]/(Fs/2); % Frequency vector
A = [1 1 0 0]; % Amplitude vector
D = [Dpass Dstop];   % Deviation (ripple) vector
b  = firgr('minorder',F,A,D);
LP = dsp.FIRFilter('Numerator',b);
% fvtool(b,'Color','White');

% Filter out 50 Hz
Fpass1  = 48;
Fstop = 50;
Fpass2 = 52;
N = 300;
F = [0 Fpass1 Fstop Fpass2 Fs/2]/(Fs/2);
A = [1 1 0 1 1];
S = {'n' 'n' 's' 'n' 'n'};
b = firgr(N,F,A,S);
BP = dsp.FIRFilter('Numerator',b);
% fvtool(b,'Color','White');

% High pass filter (remove signal wandering)
Fstop = 0.1;
Fpass = 0.7;
Dstop = 0.0001;
Dpass = 0.05;
F = [0 Fstop Fpass Fs/2]/(Fs/2); % Frequency vector
A = [0 0 1 1]; % Amplitude vector
D = [Dstop Dpass];   % Deviation (ripple) vector
b  = firgr('minorder',F,A,D);
HP = dsp.FIRFilter('Numerator',b);
% fvtool(b,'Color','White');

for i=1:length(list)
    l = list(i);
    data = readtable(l.folder+"\"+l.name);
    data = data(:, [1:12]);
    new_data = zeros(5000, 12);
    for j = 1:12
        d = table2array(data(1:5000, j));  % d contains the original signal to filter
        z = zeros(10000,1); % z is a support to remove the filter delay
        
        % add pre and post signal
        sup_len = 1000;  % length of the pre and post signals
        z2 = zeros(sup_len, 1);
        %d_pre = 2*d(1) - flipud(d(2:sup_len+1));
        %d_post = 2*d(end) - flipud(d(end-sup_len:end-1));
        d_pre = d(1) + z2;
        d_post = d(end) + z2;
        d2 = [d_pre;d;d_post];
        z = d(end) + z;
        z(1:7000, :) = d2;
        d2 = z;  % d2 contains the modified signal to filter
        
        temp_d = BP(d2);
        temp_d = HP(temp_d);
        filtered_d = LP(temp_d);
        
        delay1 = mean(grpdelay(BP));
        delay2 = mean(grpdelay(LP));
        delay3 = mean(grpdelay(HP));
        delay = delay1 + delay2 + delay3;

        new_data(:, j) = filtered_d(delay+sup_len+1:delay+sup_len+5000);
    end
    writematrix(new_data, l.folder+"\preproc\"+l.name);
end


% plot(d)
% hold on
% plot(d2)
% hold off
% % 
% plot(d)
% hold on
% plot(filtered_d(delay+sup_len+1:delay+sup_len+5300))
% hold off

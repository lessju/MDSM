% This program quantizes a sound signal using mu-law quantization

function []=quant_mulaw(inname, N, mu)
if nargin < 3
   disp('Usage: quant_mulaw(inname,outname, N, mu)');
   disp('inname: input .wav file name');
   disp('outname: output .wav file name');
   disp('N: quantization level, N should be a positive integer');
   disp('mu:  1<=mu <=255')
   return;
end;

%read in input  signal
fid = fopen(inname, 'r');
x = fread(fid, 65536*4, 'single');

xmin=min(x); xmax=max(x);

x = x./ max(x);
xmax = 1;
xmin = 0;

magmax=max(abs(x));
xmin=-magmax, xmax=magmax;
Q=1/N;
disp('xmin,xmax,N,Q,mu');
disp([xmin,xmax,N,Q,mu]);

%apply mu-law transform to original sample
y=log10(1+abs(x)*(mu))/log10(1+mu);

%apply uniform quantization on the absolute value each sample
yq=floor((y)/Q)*Q+Q/2;
     
%apply inverse mu-law transform to the quantized sequence
%also use the original sign
xq=(1/mu)*(10.^((log10(1+mu))*yq)-1).*sign(x);

figure; plot(yq,'r-');
hold on; plot(xq,'b-');
%plot(x-xq, 'k');
axis tight; grid on;
legend('original','mulaw quantized')

% Calculate the MSE
D=x-xq;
MSE=mean(D.^2);
fprintf('\n Error between original and quantized = %g\n\n',MSE )
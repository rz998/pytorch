% load data

load('MGAN_field','u_MGAN');
load('darcy_data_noiseless_latentdim25.mat','x_svecr', 'x_mean');

% compute inverse projection
x_MGAN = u_MGAN * x_svecr' + x_mean;

% determine number of test samples
N = size(x_MGAN,1);
K1 = 32;

coeff_r = x_MGAN;
figure;
imagesc(reshape(mean(coeff_r,1),K1,K1)); colorbar;
% figure;
% imagesc(reshape(var(coeff_r,[],1),K1,K1)); colorbar;
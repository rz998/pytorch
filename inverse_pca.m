% load data

load('MGAN_field','u_MGAN');
load('darcy_data_noiseless_latentdim25.mat','x_svecr', 'x_mean');

% % remove column mean
% x_flat = reshape(x, size(x,1), size(x,2)*size(x,3));
% x_mean = mean(x_flat,1);
% x_flat = (x_flat - x_mean);
% 
% 
% % svd
% [~, S_vals, x_svec] = svds(x_flat, 200);
% 
% % extract sigular vectors with latent dimension
% ldim = size(u_MGAN,2);
% x_svecr  = x_svec(:,1:ldim);

% compute inverse projection
x_MGAN = u_MGAN * x_svecr' + x_mean;

% determine number of test samples
N = size(x_MGAN,1);
K1 = 128;

coeff_r = x_MGAN;
figure;
imagesc(reshape(mean(coeff_r,1),K1,K1)); colorbar;
% figure;
% imagesc(reshape(var(coeff_r,[],1),K1,K1)); colorbar;
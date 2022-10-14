load('true_field','x','y')
K1=256;
coeff_r = reshape(x,K1,K1);
figure;
imagesc(coeff_r); colorbar;
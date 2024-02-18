function K = gaussianKernel(a, b, sigma)
 
 la = size(a, 1);
 cb = size(b, 2);
 K = zeros(la, cb);
 
for i = 1:la
   for j = 1:cb
        K(i,j) = exp(-( norm(a(i, :)'-b(:, j)).^2./(2*sigma^2)));
   end
end

end

filename = ['hyperband_evals.txt'];
M = dlmread(filename);

semilogx(M(:,1), 100-M(:,2), 'color', 'red', 'LineWidth', 5);

xlabel('Evaluations','fontsize',16);
ylabel('Best Validation error (%)','fontsize',16);
title('Hyperband (max_iter=60s)','fontsize',16);
ylim([0.0 2.0]);
xlim([1 1000]);
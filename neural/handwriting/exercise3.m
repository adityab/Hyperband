function unt()

%clear all;
close all;
set(gcf,'color','w');

colors = {'red','blue','green','black','magenta','cyan', ...
    [0.4 0.7 0.1],[0.7 0.4 0.1],[0.1 0.4 0.7],[0.7, 0.7, 0]};

# RANDOM
irun = 1;
filename = ['solutions_' num2str(5) '_' num2str(irun) '.txt'];
M = dlmread(filename);  % epoch_index   total_time  train_loss  val_loss    val_error

nevaluations = size(M,1);   % number of evaluated solutions
nsimulatedruns = 1000;        % number of simulations of random search
for iter=1:nsimulatedruns 
    indexes = 1:nevaluations;
    simidx = randsample(indexes,nevaluations);   % random sequence without repetations
    fbest(iter,1) = 100.0 - M(simidx(1),5);
    for t=2:nevaluations    % find the best error so far at step t
        newval = 100.0 - M(simidx(t),5);
        fbest(iter,t) = fbest(iter,t-1);
        if (newval < fbest(iter,t-1))
            fbest(iter,t) = newval;
        end;
    end;
end;

filename = ['hyperband/hyperband_evals.txt'];
E = dlmread(filename);
median_fbest = median(fbest);
semilogx(1:nevaluations, fbest(1,:), 'color', 'blue');    hold on;
semilogx(1:nevaluations, median_fbest, 'color', 'red', 'LineWidth', 5);    hold on;
semilogx(E(:,1), 100-E(:,2), 'color', 'green', 'LineWidth', 5); hold on;
nsimulationsToPlot = 20;
for iter=2:nsimulationsToPlot
    semilogx(1:nevaluations, fbest(iter,:), 'color', 'blue');    hold on;
end;

# HYPERBAND 
legend('Simulated run','Median simulated run (random search)', 'Hyperband (max\_iter=60s)');

xlabel('Evaluations','fontsize',16);
ylabel('Best Validation error (%)','fontsize',16);
ylim([0.0 2.0]);
xlim([1 1000]);

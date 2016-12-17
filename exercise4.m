function unt()

%clear all;
close all;
set(gcf,'color','w');

iscenario = 1;
colors = {'red','blue','green','black','magenta','cyan', ...
    [0.4 0.7 0.1],[0.7 0.4 0.1],[0.1 0.4 0.7],[0.7, 0.7, 0]};

if (iscenario == 1)
    filename = 'fprofile.txt';
    M = dlmread(filename);
    ii = 0;
    nx = size(M,2);
    xvals = (0:nx-1) * 1.0 / (nx - 1);
    leg = [];
    for nepochs = [1, 3, 9, 27, 81]
        ii = ii + 1;
        yvals = M(ii,:);
        plot(xvals, yvals, 'color', colors{ii}); hold on;
        leg{ii} = ['nepochs = ' num2str(nepochs) ];
    end;
    xlabel('hyperparameter value','fontsize',16);
    ylabel('validation loss','fontsize',16); 
    legend(leg);
    
end;

if (iscenario == 2)
    % copy-paste from iscenario 1 =)
    filename = 'fprofile.txt';
    M = dlmread(filename);
    ii = 0;
    nx = size(M,2);
    xvals = (0:nx-1) * 1.0 / (nx - 1);
    leg = [];
    for nepochs = [1, 3, 9, 27, 81]
        ii = ii + 1;
        yvals = M(ii,:);
        plot(xvals, yvals, 'color', colors{ii}); hold on;
        leg{ii} = ['nepochs = ' num2str(nepochs) ];
    end;
    xlabel('hyperparameter value','fontsize',16);
    ylabel('validation loss','fontsize',16); 
    legend(leg);
    
    % plot hyperband evaluations
    max_iter = 81;
    eta = 3;
    s_max = floor(log(max_iter)/log(eta)); 
    B = (s_max+1)*max_iter;
    
    handles = [];
    for s=[s_max:-1:0]
        delete(handles);
        handles = [];
        filename = ['hband_benchmark_' num2str(s) '.txt'];
        M = dlmread(filename);

        n = ceil(floor(B/max_iter/(s+1))*eta^s);

        yshift = 0;
        for ii=0:s
            n_i = n*eta^-ii;
            xvals = M((1+yshift):(yshift+n_i), 1);
            yvals = M((1+yshift):(yshift+n_i), 2);
            yshift = yshift + n_i;
            h = plot(xvals, yvals, 'o', 'color', colors{ii+1}, 'LineWidth', 5);
            handles = [handles h];
        end;
        title(['HyperBand, s=' num2str(s)]);
        pause(2);
    end;
end;

if (iscenario == 3)
    
    for ialgo=[1, 2]
        nruns = 100;
        ybest_runs = [];
        for irun=1:nruns
            if (ialgo == 1)    filename = ['randomsearch_' num2str(irun-1) '.txt'];     end;
            if (ialgo == 2)    filename = ['hyperband_' num2str(irun-1) '.txt'];        end;
            M = dlmread(filename);
            xvals = M(:,1);
            yvals = M(:,2);
            ybest = yvals;  % already computed in our python code
            ybest_runs(irun,:) = ybest;
        end;
        ybest_median = mean(ybest_runs);
        semilogx(xvals, ybest_median, 'color', colors{ialgo}); hold on;
    end;
    ylim([0 1]);
    legend({'Random','Hyperband'});
    xlabel('number of function evaluations','fontsize',16);
    ylabel('best validation loss','fontsize',16); 
    title('random search vs hyperband','fontsize',16);
end;

%export_fig('1.png', ['-r300']);
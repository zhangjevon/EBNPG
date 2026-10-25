% Modified from the MPG repository (https://github.com/lfprudente/MPG); see CHANGES.md.
% SPDX-License-Identifier: GPL-3.0-or-later
function [x,info,iter,ngev,nhev,time] = ProxGrad(n,m,l,u,x,lsopt,dimA,A,b)

tic;

% Initial parameters
tol     = 10^(-4);
itermax = 200;

% Counters
ngev = 0;
nhev = 0;
iter = 0;

% Evaluate F
for i = 1:m
    G(i) = evalg(n,x,i);
    ngev = ngev + 1;

    [H(i),flag] = evalh(n,x,i,A,b);
    nhev = nhev + 1;
    if ( flag ~= 0 )
        info = -1;
        time = toc;
        print_final(info, iter, ngev, nhev, time, G, H);
        return
    end 
end

F = G + H;

% Print the header (only in single-run or detailed mode)
% In a multi-start test, comment out the prints below to keep the output concise
% fprintf('----------------------------------------------------------------------\n')
% fprintf('   Proximal Gradient Method for Composite Multiojective Optimization  \n')
% fprintf('----------------------------------------------------------------------\n')
% if ( lsopt == 1 )
%     fprintf('Version              : Explicit line search \n')
% elseif ( lsopt == 2 )
%     fprintf('Version              : Armijo line search \n')
% elseif ( lsopt == 3 )
%     fprintf('Version              : Implicit line search \n')
% end
% fprintf('Number of variables  : %i \n',n)
% fprintf('Number of objectives : %i \n',m)
% fprintf('Optimality tolerance: %.0e \n',tol)
% fprintf('\n')

% Main Loop
while(1)

    % Compute Jacobian of G
    for i = 1:m
        JG(i,:) = evalgradg(n,x,i);
    end

    % Solve subproblem
    alpha = 1;
    [theta,tau,p,flagIS] = subproblemProxGrad(n,m,l,u,x,alpha,JG,H,dimA,A,b,[]);
    if ( flagIS ~= 0 )
        info = -2;
        time = toc;
        print_final(info, iter, ngev, nhev, time, G, H);
        return
    end 

    % Compute direction and step norm ||d^k|| = ||p - x||
    d = p - x;
    normd_sub = norm(d);

    % Print the iteration information (optional)
    if ( mod(iter,10) == 0 && iter > 0 )
        fprintf('%-5s   %-9s  %-8s       %-8s   %-8s %-8s %-8s \n','it','||d^k||','|theta|','fun1','fun2','IS','LS')
    end
    if ( iter == 0 )
        fprintf('%5d       %-4s   %8.2e    %+8.2e   %+8.2e   %-8i %-8s\n',iter,'-',abs(theta),F(1),F(2),flagIS,'-')
    else
        fprintf('%5d    %8.2e  %8.2e    %+8.2e   %+8.2e   %-8i %-8i\n',iter,normd_sub,abs(theta),F(1),F(2),flagIS,flagLS)
    end
    
    % Stopping criteria: ||d^k|| <= tol
    if ( iter > 0  && normd_sub <= tol )
        info = 0;
        time = toc;
        print_final(info, iter, ngev, nhev, time, G, H);
        return
    end

    if ( iter >= itermax )
        info = 1;
        time = toc;
        print_final(info, iter, ngev, nhev, time, G, H);
        return
    end 

    % Iterate
    iter = iter + 1;
    xprev = x;

    % Define direction (d already computed above for stopping criterion)
    if ( lsopt == 3 ), d = p - x; end
    
    % Explicit
    if ( lsopt == 1 )
        stp = 1;
        DG0 = JG * d;
        [~,imax] = max(DG0);
        [stp,Gtrialimax,ngevLS,flagLS] = explicitLSone(n,x,d,stp,imax,G(imax),DG0(imax));
        ngev = ngev + ngevLS;
        if ( flagLS == -1 )
            info = -3;
            time = toc;
            print_final(info, iter, ngev, nhev, time, G, H);
            return
        end 
        xtrial = x + stp * d;
        Ftrial(1:m) = 0; Gtrial(1:m) = 0; Htrial(1:m) = 0;
        Gtrial(imax) = Gtrialimax;
        Fdec = true;
        for i = 1:m
            if ( i == imax ) continue; end
            [Gtrial(i)] = evalg(n,xtrial,i); ngev = ngev + 1;
            [Htrial(i),flag] = evalh(n,xtrial,i,A,b); nhev = nhev + 1;
            if ( flag ~= 0 )
                info = -1; time = toc;
                print_final(info, iter, ngev, nhev, time, G, H);
                return
            end 
            Ftrial(i) = Gtrial(i) + Htrial(i);
            if ( Ftrial(i) > F(i) )
                Fdec = false; ifirst = i; break;
            end
        end
        if ( Fdec )
            [Htrial(imax),flag] = evalh(n,xtrial,imax,A,b); nhev = nhev + 1;
            if ( flag ~= 0 )
                info = -1; time = toc;
                print_final(info, iter, ngev, nhev, time, G, H);
                return
            end 
            Ftrial(imax) = Gtrial(imax) + Htrial(imax);
            x = xtrial; G = Gtrial; H = Htrial; F = Ftrial;
        else
            [stp,G,ngevLS,flagLS] = explicitLS(n,m,x,d,stp,ifirst,G,DG0,DG0);
            ngev = ngev + ngevLS;
            if ( flagLS == -1 )
                info = -3; time = toc;
                print_final(info, iter, ngev, nhev, time, G, H);
                return
            end 
            x = x + stp * d;
            for i = 1:m
                [H(i),flag] = evalh(n,x,i,A,b); nhev = nhev + 1;
                if ( flag ~= 0 )
                    info = -1; time = toc;
                    print_final(info, iter, ngev, nhev, time, G, H);
                    return
                end
            end
            F = G + H; 
        end
    end
    
    % Armijo
    if ( lsopt == 2 )
        [stp,G,H,ngevLS,nhevLS,flagLS] = armijo(n,m,x,d,F,JG,tau,A,b);
        ngev = ngev + ngevLS; nhev = nhev + nhevLS;
        if ( flagLS == -1 )
            info = -3; time = toc;
            print_final(info, iter, ngev, nhev, time, G, H);
            return
        end 
        x = x + stp * d;
        F = G + H;
    end
    
    % Implicit
    if ( lsopt == 3 )
        [p,G,ngevLS,flagLS] = implicitLS(n,m,l,u,x,p,alpha,G,JG,H,dimA,A,b);
        ngev = ngev + ngevLS;
        if ( flagLS == -1 )
            info = -3; time = toc;
            print_final(info, iter, ngev, nhev, time, G, H);
            return
        end 
        x = p;
        for i = 1:m
            [H(i),flag] = evalh(n,x,i,A,b); nhev = nhev + 1;
            if ( flag ~= 0 )
                info = -1; time = toc;
                print_final(info, iter, ngev, nhev, time, G, H);
                return
            end
            F = G + H;
        end
    end
end
end

% Helper: print the final result in a uniform format
% Helper: print the final result in a uniform format (any number of objectives)
function print_final(info, iter, ngev, nhev, time, G, H)
    fprintf('\n========================================\n');
    if info == 0
        fprintf('Termination status: %d (solution found)\n', info);
    elseif info == 1
        fprintf('Termination status: %d (max iterations reached)\n', info);
    elseif info == -1
        fprintf('Termination status: %d (error in H evaluation)\n', info);
    elseif info == -2
        fprintf('Termination status: %d (error in subproblem)\n', info);
    elseif info == -3
        fprintf('Termination status: %d (error in line search)\n', info);
    else
        fprintf('Termination status: %d (unknown)\n', info);
    end
    fprintf('Iterations        : %d\n', iter);
    fprintf('Grad evaluations  : %d\n', ngev);
    fprintf('H evaluations     : %d\n', nhev);
    fprintf('CPU time (s)      : %.2f\n', time);
    if nargin >= 7 && ~isempty(G) && ~isempty(H)
        % Get the number of objectives
        if length(G) == length(H)
            m_actual = length(G);
        else
            m_actual = min(length(G), length(H));
        end
        if m_actual >= 1
            fprintf('Final F           : ');
            for i = 1:min(m_actual, 5)  % print at most the first 5 objectives
                if i > 1
                    fprintf('   ');
                end
                fprintf('%+8.2e', G(i)+H(i));
            end
            if m_actual > 5
                fprintf(' ...');
            end
            fprintf('\n');
        end
    end
    fprintf('========================================\n');
end
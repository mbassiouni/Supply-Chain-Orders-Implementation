function  [P1,T1] = convertlabels(P, T)
     for i = 1 : length (T)
        if P(i) == 0
            P1{i} = 'NotLate';
        else
            P1{i} = 'Late';
        end
        
        if T(i) == 0
            T1{i} = 'NotLate';
        else
            T1{i} = 'Late';
        end
     end
end
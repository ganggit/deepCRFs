function Xcell = deep_project_cell(Xcell, layers, numlayers)


if nargin <3
    
    numlayers =3;
end

numtrain = length(Xcell);

for i =1: numtrain
    Xcell{i} = deep_project(Xcell{i}, layers, numlayers);

end
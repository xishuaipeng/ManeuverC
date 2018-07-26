function [remove_list] = remove_mat(path, matlist)
n_case = size(matlist,1);
remove_list = [];
for i =1:n_case
   mat_path = fullfile(path, matlist{i}) ;
   if ~exist(mat_path,'file')
      disp([mat_path,'not exist']) ;
      continue;
   end
   delete(mat_path);
   remove_list = [remove_list;{mat_path}];
    
end

end


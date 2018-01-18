function selfDataset(number)
% number is the number of examples

files = dir('spam_2');
m = number; % m examples
X = zeros(m,1899);
y = ones(m,1);
for i = 1:m, % take the first m emails as data set 
    file_name = files(i+2).name; % the first two elements are "." and ".."
    fprintf("processing email %s \n", file_name);
    file_contents = readFile(file_name);
    word_indices = processEmail(file_contents);
    X(i,:) = emailFeatures(word_indices);
end
save selfData.mat X y;
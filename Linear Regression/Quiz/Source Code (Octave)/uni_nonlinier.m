data = csvread("trainset1-15.csv");

x = data(:,1);
y = data(:,2);

% Polynomial 3
mat_X = [];
mat_X(:,2) = x;
mat_X(:,1) = 1;
mat_X(:,3) = x .^ 2;
mat_X(:,4) = x .^ 3;

w_pol3 = (inverse(transpose(mat_X) * mat_X)) * transpose(mat_X) * y;
y_pol3 = mat_X * w_pol3;

sse(1,1) = sum((y-y_pol3).^2);


% Polynomial 5
mat_X = [];
mat_X(:,2) = x;
mat_X(:,1) = 1;
mat_X(:,3) = x .^ 2;
mat_X(:,4) = x .^ 3;
mat_X(:,5) = x .^ 4;
mat_X(:,6) = x .^ 5;

w_pol5 = (inverse(transpose(mat_X) * mat_X)) * transpose(mat_X) * y
y_pol5 = mat_X * w_pol5

sse(1,2) = sum((y-y_pol5).^2);


% Polynomial 7
mat_X = [];
mat_X(:,2) = x;
mat_X(:,1) = 1;
mat_X(:,3) = x .^ 2;
mat_X(:,4) = x .^ 3;
mat_X(:,5) = x .^ 4;
mat_X(:,6) = x .^ 5;
mat_X(:,7) = x .^ 6;
mat_X(:,8) = x .^ 7;


w_pol7 = (inverse(transpose(mat_X) * mat_X)) * transpose(mat_X) * y
y_pol7 = mat_X * w_pol7

sse(1,3) = sum((y-y_pol7).^2);


data_test = csvread("testset1-15.csv");
x_test = data_test(:,1)

mat_X = [];
mat_X(:,2) = x_test;
mat_X(:,1) = 1;
mat_X(:,3) = x_test .^ 2;
mat_X(:,4) = x_test .^ 3;
mat_X(:,5) = x_test .^ 4;
mat_X(:,6) = x_test .^ 5;

y_test = mat_X * w_pol5

plot(x,y_pol3, "r");
hold on;
plot(x,y_pol5, "g");
hold on;
plot(x,y_pol7, "b");
hold on;
scatter(x,y); grid on;
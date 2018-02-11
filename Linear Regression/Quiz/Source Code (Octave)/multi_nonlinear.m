data = csvread("trainset2-15.csv");

x1 = data(:,1);
x2 = data(:,2);
y = data(:,3);

% Polynomial 3 langsung (x1+x2) ^ 3
mat_X = [];
mat_X(:,2) = x1;
mat_X(:,1) = 1;
mat_X(:,3) = x2;
mat_X(:,4) = (x1 + x2) .^ 3;

w_pol3sung = (inverse(transpose(mat_X) * mat_X)) * transpose(mat_X) * y
y_pol3sung = mat_X * w_pol3sung

sse(1,1) = sum((y-y_pol3sung).^2);


% Polynomial 3 ada x1*x2
mat_X = [];
mat_X(:,2) = x1;
mat_X(:,1) = 1;
mat_X(:,3) = x2;
mat_X(:,4) = x1 .* x2;
mat_X(:,5) = (x1 + x2) .^ 3;

w_pol3 = (inverse(transpose(mat_X) * mat_X)) * transpose(mat_X) * y
y_pol3 = mat_X * w_pol3

sse(1,2) = sum((y-y_pol3).^2);

% Polynomial 5 ada x1*x2
mat_X = [];
mat_X(:,2) = x1;
mat_X(:,1) = 1;
mat_X(:,3) = x2;
mat_X(:,4) = x1 .* x2;
mat_X(:,5) = (x1 + x2) .^ 3;
mat_X(:,6) = (x1 + x2) .^ 5;

w_pol5 = (inverse(transpose(mat_X) * mat_X)) * transpose(mat_X) * y
y_pol5 = mat_X * w_pol5

sse(1,3) = sum((y-y_pol5).^2);

% Polynomial 7 ada x1*x2
mat_X = [];
mat_X(:,2) = x1;
mat_X(:,1) = 1;
mat_X(:,3) = x2;
mat_X(:,4) = x1 .* x2;
mat_X(:,5) = (x1 + x2) .^ 3;
mat_X(:,6) = (x1 + x2) .^ 5;
mat_X(:,7) = (x1 + x2) .^ 7;

w_pol7 = (inverse(transpose(mat_X) * mat_X)) * transpose(mat_X) * y
y_pol7 = mat_X * w_pol7

sse(1,4) = sum((y-y_pol7).^2);


plot3(x1,x2, y_pol7, "k");
hold on;
plot3(x1,x2, y_pol5, "b");
hold on;
plot3(x1,x2, y_pol3sung, "r");
hold on;
plot3(x1, x2, y_pol3, "g");
hold on;
scatter3(x1, x2, y);
grid on;

data_test = csvread("testset2-15.csv");
x1_test = data_test(:,1)
x2_test = data_test(:,2)

mat_X_test = [];
mat_X_test(:,2) = x1_test;
mat_X_test(:,1) = 1;
mat_X_test(:,3) = x2_test;
mat_X_test(:,4) = x1_test .* x2_test;
mat_X_test(:,5) = (x1_test + x2_test) .^ 3;
mat_X_test(:,6) = (x1_test + x2_test) .^ 5;

y_test = mat_X_test * w_pol5

sse(1,3) = sum((y-y_pol5).^2);


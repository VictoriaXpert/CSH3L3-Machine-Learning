data = csvread("trainset2-15.csv");

x = data(:,1:2);
y = data(:,3);

w = (inverse(transpose(x) * x)) * transpose(x) * y

y_pred = x * w

sse = sum((y-y_pred).^2);

plot3(x(:,1), x(:,2), y_pred);
%plot(x(:,1), y_pred)
hold on;
scatter3(x(:,1), x(:,2), y);
%scatter(x(:,1), y)
grid on;

data_test = csvread("testset2-15.csv");
x_test = data_test(:,:)
y_test = x_test * w;
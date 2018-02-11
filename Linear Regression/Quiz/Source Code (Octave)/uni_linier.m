data = csvread("trainset1-15.csv");

x = data(:,1);
y = data(:,2);

w = sum((x-mean(x)).*(y-mean(y)))/sum((x-mean(x)).^2);

a = mean(y) - (w * mean(x));

y_pred = (w .* x) + a;

sse = sum((y-y_pred).^2);

plot(x,y_pred)
hold on
scatter(x,y); grid on;

data_test = csvread("testset1-15.csv");
x_test = data_test(:,1)
y_test = (w .* x_test) + a;
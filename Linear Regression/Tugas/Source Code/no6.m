data = []
data(1,:) = [-2.864, 73.128]
data(2,:) = [1.489,20.271]
data(3,:) = [-0.235,-3.220]
data(4,:) = [1.092,0.347]
data(5,:) = [2.898, -75.691]

x = data(:,1)
y = data(:,2)

y_pred =  21.274 - 10.608 .* x + 0.921 .* x .^ 2

sse = sum((y-y_pred) .^ 2)
t = [1, 2, 3, 4, 5 ,6 ,7,8, 9, 10]
error = [0.5, 0.3, 0.6, 0.6, 0.5, 0.7, 0.8, 1, 1.2, 1]

rate_of_change = [(error[i+1] - error[i]) / (t[i+1] - t[i]) for i in range(len(error)-1)]
print(rate_of_change)
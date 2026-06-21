#part-1
import numpy as np
data=np.loadtxt('assign1_data.txt', skiprows=1)

#print(data)

#m and b for x1

x1=data[:,0]
y=data[:,2]

avg_x1=np.mean(x1)
avg_y=np.mean(y)

num1=np.sum((x1 - avg_x1) * (y - avg_y))
deno1=np.sum((x1 - avg_x1) ** 2)

m1=num1/deno1
b1=avg_y - m1 * avg_x1

print("==========Calculations for part-1===========\n")

print("For x1: ")
print("m1:", m1)
print("b1:", b1)

#m and b for x2

x2=data[:,1]

avg_x2=np.mean(x2)

num2=np.sum((x2 - avg_x2) * (y - avg_y))
deno2=np.sum((x2 - avg_x2) ** 2)

m2=num2/deno2
b2=avg_y - m2 * avg_x2

print("\nFor x2: ")
print("m2:", m2)
print("b2:", b2)


#part-2

print("\n==========Calculations for part-2===========\n")


A = np.vstack([x1, x2, np.ones(len(x1))]).T

w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]

print("Full linear regression:\n")
print("w1:", w1)
print("w2:", w2)
print("b:", b)


#part-3
print("\n==========Calculations for part-3===========\n")

z= data[:,3]
#this gives an array of True/False values based on the criterion
criterion= w1*x1 + w2*x2 + b > 0
#print(criterion)
success= (sum(criterion == z).astype('int')/len(z))*100
print("Success (%) ", success)


#part-4
print("\n==========Calculations for part-4===========\n")
"""
"In machine learning, we really want to train a model based on some data and then expect the model to do well on “out of sample” data. 
Try this with the code you wrote for Part 3: Train the model on the first {25, 50, 75} examples in the data set 
and test the model on the final {75, 50, 25} examples, reporting percentage correct for each size. 
As a baseline test, do a final report on percentage correct when w1=w2=b=0.

"""
for t in [25, 50, 75]:
    w1t, w2t, bt = np.linalg.lstsq(np.vstack([x1[:t], x2[:t], np.ones(t)]).T, y[:t], rcond=None)[0]

    obtained = (w1t*x1[t:] + w2t*x2[t:] + bt > 0).astype(int)
    success = sum((obtained == z[t:]).astype(int)) / len(z[t:]) * 100
    print(f"Train {t} / Test {len(z)-t} success (%): {success}")

# zero model baseline
baseline = sum((np.zeros(len(z), dtype=int) == z).astype(int)) / len(z) * 100
print("Zero Model success (%):", baseline)

#part-5
print("\n==========Images should pop up for part-5===========\n")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#plot for y vs x1
plt.figure()
plt.scatter(x1, y, label="x1 Values")

x1_line = np.linspace(min(x1), max(x1), 100)
y_line = m1*x1_line + b1

plt.plot(x1_line, y_line, color='black', label="Line of Best Fit")

plt.xlabel("x1")
plt.ylabel("y")
plt.title("y vs x1")
plt.legend()
plt.show()

plt.figure()
plt.scatter(x2, y, label="x2 Values")

#plot for y vs x2

x2_line = np.linspace(min(x2), max(x2), 100)
y2_line = m2*x2_line + b2

plt.plot(x2_line, y2_line, color='black', label="Line of Best Fit")

plt.xlabel("x2")
plt.ylabel("y")
plt.title("y vs x2")
plt.legend()
plt.show()


# 3D plot for classifier plane

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x1[z==0], x2[z==0], y[z==0], marker='x', alpha=0.6, label='z=0', c='blue')
ax.scatter(x1[z==1], x2[z==1], y[z==1], marker='o', alpha=0.6, label='z=1', c='red')

x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1.min(), x1.max(), 20),
    np.linspace(x2.min(), x2.max(), 20)
)
y_grid = w1*x1_grid + w2*x2_grid + b

ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.3, color='green')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D Classifier Plane')
ax.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

matrix = pd.DataFrame(np.random.randint(0, 100, size=(5, 5)))
value = matrix.max().max()

print(matrix)
print('value:', value)

n = 100
m = 50
import numpy as np   

A = np.random.randint(0, 100, (n, m))

B = np.random.randint(0, 100, (n, n))
for i in range(0,n):
    for j in range(0, n):
        B[i][j] = B[j][i]

one = np.random.randint(1, 100, size = n)
two = np.random.randint(1, 100, size = n-1)
three = np.random.randint(1, 100, size = n-1)
C = np.diag(one) + np.diag(two, k=1) + np.diag(three, k=-1)

print(f"A: \n {A} \n")
print(f"B: \n {B} \n")
print(f"C: \n {C}")

a = -5
b = 6
delta = 1
z = np.arange(a, b, step = delta)

y = np.linspace(a, b, num = b-a, endpoint = False, dtype=int)

print("arange: \n", z, "\n")
print("linspace: \n", y)

file = "data.csv"
df = pd.read_csv(file)

# # find the size of the dataframe df - number of columns, rows, amount of memory used
print(f"Columns: {df.columns.size} \n")
print(f"Rows: {df.index.size} \n")
print(f"Memory Used:\n{df.memory_usage().sum()}\n")

# # extract and print the first two columns of the dataframe as numpy arrays. Name the columns C1 and C2. 
C1 = np.array(df.iloc[:, 0])
print(f"C1:\n {C1} \n")
C2 = np.array(df.iloc[:, 1])
print(f"C2:\n {C2}")

dfA = pd.DataFrame(A)
print(f"A: \n{dfA}\n")

dfB = pd.DataFrame(B)
print(f"B: \n{dfB}\n")

dfC = pd.DataFrame(C)
print(f"C: \n{dfC}\n")

# find the size and amount of memory used by these data frames
print(f"Memory Usage of A: {dfA.memory_usage().sum()}\n")
print(f"Memory Usage of B: {dfB.memory_usage().sum()}\n")
print(f"Memory Usage of C: {dfC.memory_usage().sum()}\n")

def swap(df, col1, col2):
    newdf = df.copy()
    newdf[col1], newdf[col2] = newdf[col2].copy(), newdf[col1].copy()
    newdf.rename(columns = {col1:col2, col2:col1}, inplace = True)
    return newdf
    
col1 = 'genre'
col2 = 'movie'
df2 = swap(df, col1, col2)
df2

print(f"Head:\n {df.head()} \n")
print(f"Tail:\n {df.tail()} \n")
print(f"Info:\n {df.info()} \n")
print(f"Describe:\n {df.describe()}")

print(f"Shallow:\n {df.memory_usage().sum()} \n")
print(f"Deep\n {df.memory_usage(deep = True).sum()} \n")


# %matplotlib inline

t = np.linspace(start=0, stop=np.pi, num=30)
plt.plot(t, 3*np.sin(2*np.pi*t), "r+")

plt.xticks(ticks=np.linspace(0.0, np.pi, 3), labels=[0, "π/2", "π"])
plt.yticks(ticks=np.linspace(-10.0, 10.0, 9))
plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("f(t) = 3sin(2πt)")
plt.axis((0, np.pi, -10, 10))
plt.show()



def outputgraph(aval, fval, subplotnum):
    ax = plt.subplot(subplotnum)
    ax.plot(t, f(aval, fval, t))
    ax.set_xlabel(f"a: {aval}")
    ax.set_ylabel(f"f: {fval}")
    return ax

plt.figure(figsize =(8,8))
plt.suptitle("Sine waves with varying a=[2,8], f=[2,8]")        

t = np.linspace(0.0, np.pi, 500)
def f(a, f, t):
    return a * np.sin(2 * np.pi * f * t)

ax1 = outputgraph(2, 2, 221)
ax2 = outputgraph(8, 2, 222)
ax3 = outputgraph(2, 8, 223)
ax4 = outputgraph(8, 8, 224)

axs = [ax1, ax2, ax3, ax4]

for ax in axs:
    ax.set_xlim(0, np.pi)
    ax.set_ylim(-10.0, 10.0)
    ax.set_xticks(ticks = [0.0, np.pi/2, np.pi], labels = (0, "π/2", "π"))

plt.subplots_adjust(left = 0.09, right = 0.99, top = 0.85, bottom = 0.004, wspace = 0.5, hspace = 0.5)
plt.show()


def sinwave(a, f, t):
    return a * np.sin(2 * np.pi * f * t)

ax = plt.subplot(111)
t = np.linspace(0, np.pi, 500)

ax.plot(t, sinwave(2, 2, t), "b-", t, sinwave(8, 8, t), "r--")

ax.set_xlim(0.0, np.pi)
ax.set_ylim(-10.0, 10.0)
ax.set_xticks(ticks = [0, np.pi/2, np.pi], labels = (0, "π/2", "π"))
ax.set_yticks(ticks = (np.linspace(-10.0, 10.0, 9)))
ax.legend(["a: 2, f: 2", "a: 8, f: 8"])
plt.grid()
plt.show()


from mpl_toolkits import mplot3d
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
   return ((x**2) + (y**2) - x - y)

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');


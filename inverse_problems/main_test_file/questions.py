import numpy as np

arr0 = [[1,2,3],[4,5,6],[7,8,9]]
arr1 = [[11,12,13],[14,15,16],[17,18,19]]
arr2 = [[21,22,23],[24,25,26],[27,28,29]]

print("new")
print(len(arr0))
print(arr0[0])
print(arr0[0][0])

print("new")
arrz = np.ones([2,3])
print(len(arrz))
print(arrz[0])
print(arrz[0][0])


print("new")
print(np.concatenate([arr0,arr1,arr2]))
arr = [arr0,arr1,arr2]
arr = np.array(arr)
print(arr)
print(arr.flatten(order="F"))
print(arr.flatten(order="C"))
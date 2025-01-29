def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

data = [3, 9, 10, 27, 38, 43, 82]
target = 27
result = binary_search(data, target)
print("Target found at index:", result)
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        print(mid,"mid")
        left_half = arr[:mid]
        print(left_half,"left_half")
        
        right_half = arr[mid:]
        print(right_half,"right_half")

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

data = [38, 27, 43, 3, 9, 82, 10]
merge_sort(data)
print("Sorted array:", data)
def merge_sort(A):
    n = len(A)
    if n == 1:
        return A
    
    L = []
    R = []
    
    for i in range(n // 2):
        L.append(A[i])
    
    for i in range(n // 2, n):
        R.append(A[i])
    
    L = merge_sort(L)
    R = merge_sort(R)
    
    return merge(L, R)

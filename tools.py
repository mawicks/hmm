def find(vector, value):
    '''Return the index of the first element of "vector" that equals or exceeds "value" ("vector" must be pre-sorted). If no such element exist return None'''

    def helper(vector, first, n, value):
        if n == 1:
            if vector[first] >= value:
                return first
            else:
                return None
            
        half_n = n // 2
        mid = first + half_n - 1

        if vector[mid] >= value:
            return helper(vector, first, half_n, value)
        else:
            return helper(vector, first + half_n, n - half_n, value)

    if len(vector) > 0:
        return helper(vector, 0, len(vector), value)
    else:
        return None
    

def cumsum(sequence, initial = 0):
    sum = initial
    for item in sequence:
        sum += item
        yield sum
    
    

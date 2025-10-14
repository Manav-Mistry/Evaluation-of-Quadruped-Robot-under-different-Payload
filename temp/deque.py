from collections import deque
error_stream = [(1, 1, 1), (2, 2, 2),(3, 3, 3),(4, 4, 4)]

errors = deque(maxlen=2)
for e in error_stream:
    errors.append(e)
    if len(errors) == 2:
        print(list(errors))  # always last two


print(errors[0][2], errors[1])

def get_index(lines, pattern, stoppattern=None, maxiter=None):

    for i, line in enumerate(lines):

        if pattern in line:
            return i

        if stoppattern and stoppattern in line:
            return None

        if maxiter and i > maxiter:
            return None

    return None

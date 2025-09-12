def get_divisors(n):
    divisors = list()
    for i in range(2, int(n / 2) + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors

def average_difference(values):
    if len(values) < 2:
        return 0
    differences = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return sum(differences) / len(differences)
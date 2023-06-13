my= ['mitoticr', '151', 'cell', 'mitotics', '245', 'cell', 's']

# numeric_values = [x for x in my if x.isdigit()]

my_list = [float(x)/100 if x.isdigit() else x for x in my]

# Subtract 1 from each numeric value greater than 1
my_list = [num-1 if isinstance(num, float) and num > 1 else num for num in my_list]

# Use a list comprehension to round each value to 2 decimal points
my_list = [round(num, 2) if isinstance(num, float) else num for num in my_list]

my_list = [x for x in my_list if isinstance(x, (int, float))]

print(my_list)

# print(my_list)

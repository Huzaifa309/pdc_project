import random

def generate_3d_points(num_points, file_name):
    with open(file_name, "w") as file:
        for _ in range(num_points):
            # Generate x, y, z coordinates as random doubles between 0 and 9
            x = round(random.uniform(0, 9), 6)
            y = round(random.uniform(0, 9), 6)
            z = round(random.uniform(0, 9), 6)
            file.write(f"{x} {y} {z}\n")
    print(f"{num_points} 3D points saved to '{file_name}'.")

# Generate 1000 3D points and save to 'points.txt'
generate_3d_points(10000000, "points.txt")

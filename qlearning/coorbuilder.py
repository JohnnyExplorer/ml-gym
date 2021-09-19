c= 0
for y in range(-1,2):
    for x in range(-1,2):
        print('--',c)
        print(f"self.move(x={x}, y={y})")
        c += 1
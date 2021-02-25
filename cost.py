
#ix_min = 5
#ix_max = 5

print('ix_len')
ix_len = input()

while (ix_len != 'stop'):
	n = 1000000.0
	P = 1024.0

	w = float(ix_len)

	Pm = P - w

	nPm = ((Pm - 1.0) + w + n)/Pm

	nP = ((P - 1.0) + n)/P

	C1 = nPm * P + 10*P*Pm
	C2 = nP * P - P
	print('shared tile')
	print(C1)
	print('global read')
	print(C2)
	print('ix_len')
	ix_len = input()
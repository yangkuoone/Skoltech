︠388c63cc-efd5-49da-bcb5-5f704e1e323cs︠
import numpy as np
︡a48438ee-c690-469a-91fa-33962eaad9d0︡{"done":true}︡
︠93391ec0-ca40-4ef1-81f4-1c797512e087s︠
p = 5
g = sage.groups.matrix_gps.linear.SL(2, p)
elements_list = g.list()
num_elements = len(elements_list)
# print 'num of group elements', num_elements
A = Matrix([[1, 2],[0, 1]])
B = Matrix([[1, 0],[2, 1]])
S1 = [A^2, B, A*B*A.inverse()]
S2 = [A.inverse()^2, B.inverse(), A*B.inverse()*A.inverse()]
print 'Generating Cayley graphs .....'
G1 = g.cayley_graph(connecting_set=S1)
G2 = g.cayley_graph(connecting_set=S2)
print 'Generating adjacency matricies of these graphs'
adj_matrix1 = G1.adjacency_matrix()
adj_matrix2 = G2.adjacency_matrix()
adj_bip_matrix = np.concatenate((adj_matrix1, adj_matrix2), axis=1)
np.savetxt('margullis_parity_check.txt', adj_bip_matrix)










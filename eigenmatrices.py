#In this file, we will compute the eigenmatrices for the association scheme
#of totally isotropic subspaces of a 2n dimensional space by computing eigenvectors of the L1 matrix.

import numpy as np
import math
import scipy.special as sci
import matplotlib.pyplot as plt

def gauss_coefficient(a,b,p):
    if a==b:
        return 1
    if b > a:
        return 0
    if b ==0:
        return 1
    if b<0 or a <0:
        return 0
    tot=1
    for i in range(b):
        tot = tot*(1-p**(a-i))/(1-p**(i+1))
    return tot


def iso_subspaces(n,m,p=2):#counts the number of m-isotropic subspaces from a total space of dimension 2n.
    tot = 1
    for i in range(m):
        factor = (p**(2*n-2*i) - 1)/(p**(m-i) - 1)

        tot = tot * factor
    return tot

def valence(i,n,p):
    tot = 1
    for j in range(i):
        tot= tot* (p**(2*n-j) - p**(n))/(p**n - p**(n-i+j))
    return tot

def tot_vertices(n,p): #isosubspaces(n,n)
    tot = 1
    for j in range(n):
        tot = tot * ( p**(2*n-j) - p**j) /(p**n - p**j)
    return tot

def intersection_array(x,i,n,p):
    if x == 'a':
        return valence(1,n,p) - intersection_array('b',i,n,p) - intersection_array('c',i,n,p)
    if x == 'b':
        return ( p**(n+1)-p**(i+1) )/ (p-1)

    if x == 'c':
        return (p**i-1 )/(p-1)

def gen_L1_matrix(n,p):
    def helper_func(c,r):
        if c==r:
            return intersection_array('a',r,n,p)
        elif c == r-1:
            return intersection_array('c',r,n,p)
        elif c == r+1:
            return intersection_array('b',r,n,p)
        else:
            return 0
    return np.array([ [int(helper_function(c,r))  for c in range(n+1)] for r in range(n+1) ])

def u_vectors(L1,n,p): #returns column vectors
    #print(np.linalg.eig(L1))
    eig = np.linalg.eig(L1)
    eig_vects = list(eig[1].T)
    eig_vects.reverse()
    eig_vects = [np.array([(eig / eig[0])]).T for eig in eig_vects] #normalize the first component to 1
    return eig_vects
def v_vectors(L1,n,p):
    diag = np.diag([valence(i,n,p) for i in range(n+1)])
    v_vects = [diag @ u for u in u_vectors(L1,n,p)]
    return v_vects
    #return [np.array([v_vect/v_vect[0]]) for v_vect in v_vects]
def multiplicities_general(L1): #prop 2.2.2
    us = u_vectors(L1,n,p)
    vs = v_vectors(L1,n,p)
    tot = tot_vertices(n,p)
    return([tot / np.inner(us[i].T,vs[i].T)[0][0] for i in range(n+1)])
def multiplicities(L1): #See page 276 of Brouwer et. al's Distance Regular Graphs.
    def helper_product(j):
        tot=1
        for i in range(1,j+1):
            tot = tot*(1+p**(n+1-i))/(1+p**(i-1))
        return tot
    return[ p**j * gauss_coefficient(n,j,p) *(1+p**(n+1-2*j))/(1+p**(n+1-j)) *helper_product(j) for j in range(n+1)]

def Q_matrix(L1,n,p):
    fs = multiplicities(L1)
    u= u_vectors(L1,n,p)
    return (np.hstack([fs[i]*u[i] for i in range(n+1)]) )

def P_matrix(L1,n,p):
    vs = v_vectors(L1,n,p)
    return np.vstack([v.T[0] for v in vs])
#---#
def pi(j,r,s,m,p):
    def helper_product(i):
        tot = 1
        if r+s-i <=m:
            for t in range(r+s-i, m):
                tot = tot * (p**(2*m-t)-p**t)/(p**m - p**t)
            return tot
        else:
            return 0 # in this case, there are 0 ways to extend a+b to z. Tricky, because the empty product is 1.
    return sum([gauss_coefficient(j,i,p)*gauss_coefficient(m-r,s-i,p)*p**((j-i)*(s-i))*helper_product(i) for i in range(j+1)])

def piT(j,r,s,w,m,p):
    #returns the pi(j,r,s) function for the semilattice of isotropic spaces above a given space of dimension w.
    def helper_product(i):
        tot = 1
        if r+s-i+w <=m:
            for t in range(r+s-i+w, m):
                tot = tot* (p**(2*m-t) - p**t)/(p**m - p**t)
            return tot
        else:
            return 0
    return sum([gauss_coefficient(j,i,p)*gauss_coefficient(m-r-w,s-i,p)*p**((j-i)*(s-i))*helper_product(i) for i in range(j+1)])

def piT_test():
    #Checks that when w=0, pi and piT return the same thing.
    p=2
    for n in range(2,5):
        for r in range(n+1):
            for s in range(n+1):
                for j in range(n+1):
                    assert piT(j,r,s,0,n,p)==pi(j,r,s,n,p)
piT_test()

def mobius(t,k,p):
    if k>t:
        print("warning backwards mobius")
    return (-1)**(t-k)*p**(sci.binom(t-k,2))

def rho(r,s,n,p): #equation 11
    return(sum([gauss_coefficient(r,j,p)*mobius(r,j,p) * pi(j,r,s,n,p) for j in range(r+1)]))

def rho2(r,s,n,p):#alternate way to get rho by use of P-matrix. This runs into numerical issues, but it's a good sanity check.
#Uses Del Sarte's equation 13.
    mat = gen_L1_matrix(n,p)
    P= P_matrix(mat,n,p)
    print(P)
    if r<=s:
        return(sum([gauss_coefficient(n-k,s,p) * P[r][k] for k in range(n+1)]))
    else:
        return 0

def rho_test():
    #checks that the two methods of generating rho are the same, up to some rounding error.
    epsilon = 2
    for p in [2,3,4,5]:
        for n in range(1,6): #Fails for larger n, but hopefully this is numerical error in rho2.
            for s in range(n):
                for r in range(s+1):
                    print (r,s,n,p)
                    print(rho(r,s,n,p), rho2(r,s,n,p))
                    assert abs(rho(r,s,n,p) - rho2(r,s,n,p))<epsilon
#rho_test()
def rhoT(r,s,w,n,p):
    return(sum([gauss_coefficient(r,j,p)*mobius(r,j,p) * piT(j,r,s,w,n,p) for j in range(r+1)]))

def Pmatrix_from_rho(rho_func,w,n,p): #expects rho_func to be a fuction of r,s. uses equation 13.
    def P_func(a,d):
        return sum([gauss_coefficient(i,n-a-w,p)*mobius(i,n-a-w,p)*rho_func(d,i) for i in range(n-a-w,n+1)])

    return np.array([[P_func(k,d) for k in range(n+1-w)] for d in range(n+1-w)])
# def rho_func(r,s):
#     w=1
#     n=3
#     p=3
#     return (rhoT(r,s,w,n,p))
def PmatrixT(n,w,p):
    def rho_func(r,s):
        return (rhoT(r,s,w,n,p))
    return Pmatrix_from_rho(rho_func,w,n,p)
# print(rho_func(5,3))
# n=p=3
# w=1
# print(Pmatrix_from_rho(rho_func,w,n,p))
# mat = gen_L1_matrix(n,p)
# P= P_matrix(mat,n,p)
# print(P)

def contextual_hidden_vars_eig_plot():
    #For n even,
    p=2
    def function_to_plot(n):
        assert n%2==0
        P = PmatrixT(n,1,p)
        l = max([abs(P[i][n//2]) for i in range(1,n)])
        return math.log2(l / P[0][n//2])

    t = [int(n) for n in range(4,30) if n%2==0]
    y = [function_to_plot(x) for x in t]
    fig, ax = plt.subplots()
    ax.set(xlabel='n', ylabel='log(l/d)', title="log of spectral ratio for n/2 relation on G_w")
    ax.plot(t, y)
    fig.savefig("spectral_ratio_n2.png")
    #print(np.polyfit(t,y,1,full=True))
#contextual_hidden_vars_eig_plot()
def contextual_hidden_vars_eig_plot_full():
    #For n even,
    p=2
    def function_to_plot(n):
        P = PmatrixT(n,1,p)
        l = max([abs(P[i][1]) for i in range(1,n)])
        return math.log2(l / P[0][1])

    t = [int(n) for n in range(4,30)]
    y = [function_to_plot(x) for x in t]
    fig, ax = plt.subplots()
    ax.set(xlabel='n', ylabel='log(l/d)', title="log of spectral ratio for 1-relation on G_w")
    ax.plot(t, y)
    fig.savefig("spectral_ratio_full.png")
contextual_hidden_vars_eig_plot_full()
# def experimental_plot():
#         p=2
#         t = np.arange(3, 40)
#         fig, ax = plt.subplots()
#         def function_to_plot(n):
#             c=1/5
#             return math.log2(rho(0,2,n,p)/ rho(1,2,n,p) * c/(1-c)) / math.log2(p**(2*n))
#         ax.set(xlabel='n', ylabel='fitness',
#                title="assume a code C with 2**n elements but the same expansion")
#         ax.plot(t, [function_to_plot(x) for x in t])
#         fig.savefig("coded_plot.png")
#experimental_plot()

def find_color(value, min, max):
    epsilon = 0.001
    assert value <= max + epsilon and value >= min - epsilon
    interval = max - min
    moved_val = value - min
    ratio = moved_val / float(interval)
    ratio = ratio * math.pi / 2.0

    return  math.sin(ratio), math.sin(ratio), math.cos(ratio)

# def gen_fitness_plot_grouping_qubits():
#     fig, ax = plt.subplots()
#     for group_size in range(1,5):
#         p=2**group_size
#         t = np.arange(3*group_size, 32)
#         def function_to_plot(n):
#             c=1/5
#             return math.log2(rho(0,2,n,p)/ rho(1,2,n,p) * c/(1-c)) / math.log2(tot_vertices(n,p)*p**n)
#         ax.plot(t, [function_to_plot(x) for x in [n//group_size for n in t]], label = "group " + str(group_size), color = find_color(group_size, 1,5))
#     ax.set(xlabel='physical qubits', ylabel='fitness estimate',
#            title="Estimated fitness when grouping qubits")
#     ax.set_facecolor((250/256.0, 100/256.0, 100/256.0))
#     ax.legend(ncol=2, facecolor = (250/256.0, 100/256.0, 100/256.0))
#     fig.savefig("fitness_plot_grouping_qubits.png")
#

def stepwise_bound(n,p=2):
    #Assume a probability of failure alpha_2 =1/5. Set t=2
    #Bound the probability of failure at level t+1 using the spectral bound.
    #Repeat until reaching level n.
    alpha = 8/9
    beta = 1 - alpha
    for t in range(3,n+1):
        #print(rho(1,(t-1),t,p)/ rho(0,(t-1),t,p))
        if beta >1:
            print("error")
        beta = min(rho(1,(t-1),t,p)/ rho(0,(t-1),t,p) *(1/(1-beta)), beta)
    print("---") #This function doesn't seem to give a nontrivial bound.
    return beta

def dynamic_programming_bound(n,p=2):
    #The best bound always seems to be to use the shadow at level 2. Thus we might as well use the simple bound.
    beta = [0,0,5/6]
    def bound(tp,t,beta):
        return rho(1,tp,t,p) / rho(0,tp,t,p) * 1/(1-beta)
    for t in range(3,n+1):
        beta.append(min([bound(tp,t,beta[tp]) for tp in range(2,t)] + [beta[t-1]]))
    return(beta[n])

def simple_bound(n,p=2):
    return 4*rho(1,2,n,p) / rho(0,2,n,p)


def new_fitness(log_M,log_O,log_alpha,w):
    #choose a set of measurements by taking a random walk of degree 1/alpha^2 on the measurement set.
    #print(log_M,log_O,log_alpha,w)
    return -(w*(log_alpha + math.log2(3)) ) / (log_M+2*(1-w)*log_alpha+w*log_O)

def pseudorandom_product():
    for n in range(3,30):
        log_M = math.log2(iso_subspaces(n,n))
        #log_alpha = gamma*n
        log_alpha = math.log2(simple_bound(n))
        log_O = n
        for w in range(1,10):
            print(n,w)
            print("old_fitness")
            print(-log_alpha/(log_M+log_O))
            print("new_fitness")
            print(new_fitness(log_M,log_O,log_alpha,w))
            print("----")
#pseudorandom_product()

def pseudorandom_product_twice():
    n=40
    w=100
    log_alpha = math.log2(simple_bound(n))
    log_alpha2 = w*log_alpha + (w-1)
    log_O2 = n*w
    log_M = math.log2(iso_subspaces(n,n))
    log_M2 = log_M+(w-1)*(3-2*log_alpha)
    for w2 in range(90,100):
        print(new_fitness(log_M2,log_O2,log_alpha2,w2))

def pseudorandom_product_several_times(n,w):
    #w is a walk expressed as a list w=[w0,w1,w2...]
    log_alpha = [math.log2(simple_bound(n))]
    log_M = [math.log2(iso_subspaces(n,n))]
    log_O = [n]
    if w[0]==0: #quick and dirty..
        return(-math.log2(simple_bound(n))/(log_M[0] + log_O[0]))
    for i in range(len(w)):
        log_alpha.append(w[i]*(log_alpha[i] + math.log2(3)) )
        log_M.append(log_M[i]+2*(1-w[i])*log_alpha[i])
        log_O.append(log_O[i]*w[i])
    l = len(w)
    return(-log_alpha[l]/ (log_M[l] + log_O[l]))
    #return (new_fitness(log_M[l],log_O[l], log_alpha[l], w[l]))

def random_walks_plot(n):
    p=2
    t = range(0,100)
    fig, ax = plt.subplots()
    def function_to_plot(n,w):
        return pseudorandom_product_several_times(n,[w])
    ax.set(xlabel='walk length', ylabel='fitness',
           title="fitness of a random walk n=" + str(n))
    ax.plot(t, [function_to_plot(n,x) for x in t])
    fig.savefig("random_walks_plot"+str(n)+".png")

#print(pseudorandom_product_several_times(44,[10000000000]))
# print(-math.log2(rho(1,2,n,p)/ rho(0,2,n,p) * 0.25) / math.log2(iso_subspaces(n,n)*p**n))
# for n in [3,5,10,30,40]:
#     random_walks_plot(n)
# n=30
# print(pseudorandom_product_several_times(n,[100000000]))
# print(pseudorandom_product_several_times(n,[1]))
#print(-math.log2(simple_bound(n,2))/n)
#print(-math.log2(simple_bound(n,2)) / math.log2((tot_vertices(n,p)*p**n)))
# print(pseudorandom_product_several_times([2]))
# print(pseudorandom_product_several_times([1000000]*20))
# print(dynamic_programming_bound(15))
# print(simple_bound(15))
# print(approximate_backwards_bound(15))
# for n in range(3,16):
#     print(stepwise_bound(n))
# gen_fitness_plot()
# gen_spectral_gap_plot()
#gen_fitness_plot_grouping_qubits()
    #print(rho2(n-2,n-2,n,p) /   rho2(0,n-2,n,p) )
# n,p = 10,2
# for x in range(10):
#     print(rho(x,9,n,p))
# p=2
# for n in range(10):
#     for s in range(n):
#         for r in range(s+1):
#             # print(rho
#             # print(rho(r,s,n,p))
#             print(n,s,r,rho2(r,s,n,p))
#             print(rho2(r,s,n,p) - rho(r,s,n,p))
#             assert abs(rho2(r,s,n,p) - rho(r,s,n,p))<1
# n,p = 3,2
# mat = gen_L1_matrix(n,p)
# Q, P = Q_matrix(mat), P_matrix(mat)
# print(Q)
# print(P)
#

def gen_spectral_gap_plot():
    p=2
    t = np.arange(3, 30)
    fig, ax = plt.subplots()
    c=4/5
    def function_to_plot(n):
        return math.log2((4*rho(1,2,n,p)/ rho(0,2,n,p)))
    ax.set(xlabel='n', ylabel='log(value(L_n))',
           title="log-plot of our upper bound on val(L_n)")
    ax.plot(t, [function_to_plot(x) for x in t])
    fig.savefig("approximation_factor_plot.png")
#gen_spectral_gap_plot()
def gen_fitness_plot():
        p=2
        t = np.arange(3, 30)
        fig, ax = plt.subplots()
        def function_to_plot(n):
            return -math.log2(4*rho(1,2,n,p)/ rho(0,2,n,p)) / math.log2(iso_subspaces(n,n)*p**n)
        ax.set(xlabel='n', ylabel='fitness',
               title="lower bound on fitness of stabilizer graphs")
        ax.plot(t, [function_to_plot(x) for x in t])
        fig.savefig("fitness_plot.png")
#gen_fitness_plot()
def line_of_best_fit():
    x = np.arange(10,30)
    #y = np.array([math.log2(simple_bound(X)) for X in x])
    y = np.array([math.log2(rho(1,2,X,p)/ rho(0,2,X,p) ) for X in x])
    a,b = np.polyfit(x,y,1)
    print(a,b)
#line_of_best_fit()
def best_fitness():
    p=2
    t= np.arange(3,30)
    tp = [-math.log2(simple_bound(n,p=2))/math.log2(iso_subspaces(n,n)*p**n) for n in t]
    print(tp)
    print(np.argmax(tp), max(tp))

#best_fitness()

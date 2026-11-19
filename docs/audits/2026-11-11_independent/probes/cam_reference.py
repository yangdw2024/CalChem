"""Independent 70-digit ITYH/B88 exchange point references (not molecular certification)."""
import mpmath as m
m.mp.dps = 70

def exchange(r, g):
    beta = m.mpf('0.0042')
    x = m.sqrt(g) / r**(m.mpf(4)/3)
    k = m.mpf(3)/4*(6/m.pi)**(m.mpf(1)/3) + beta*x*x/(1+6*beta*x*m.asinh(x))
    a = m.mpf('0.33')/(2*m.sqrt(9*m.pi/(2*k))*r**(m.mpf(1)/3))
    bracket = m.sqrt(m.pi)*m.erf(1/(2*a))+(2*a-4*a**3)*m.exp(-1/(4*a*a))-3*a+4*a**3
    atten = 1-m.mpf(8)/3*a*bracket
    return -r**(m.mpf(4)/3)*k*(m.mpf('.35')+m.mpf('.46')*atten)

for r, g in [('0.05','0.001'), ('0.2','0.1'), ('0.001','0.3')]:
    r, g = m.mpf(r), m.mpf(g)
    print('rho, gamma, E, dE/drho, dE/dgamma:', *(m.nstr(v, 24) for v in
        [r, g, exchange(r,g), m.diff(lambda rr: exchange(rr,g),r), m.diff(lambda gg: exchange(r,gg),g)]))

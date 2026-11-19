import numpy as np
rng=np.random.default_rng(1)
# decay-file inputs (% of B decays), from the piece-3 facts check (2026-11-19); M after piece-2 c
# upper-vertex D_s+: measured lines (after c), unmeasured lines (D_s class)
# upper-vertex D0+D+: measured (after c; line 806 removed for B0), unmeasured (D Dbar class)
# P: PYTHIA lines with an upper-vertex c quark (b->c cbar s, c cbar d, u c sbar)
inp={'B0':dict(Ms=5.683,Us=3.972,Md=3.359-0.272+0.744,Ud=2.099+0.794,P=8.16,
               Ds=(10.3,2.1,1.8),D=(10.4,1.9,1.9),Lc=(1.6,0.9,0.9)),
     'B+':dict(Ms=5.784,Us=3.959,Md=3.218+1.066,Ud=1.949+0.963,P=7.43,
               Ds=(7.9,1.4,1.3),D=(11.1,0.9,0.9),Lc=(2.1,0.9,0.6))}
def draw(v,n):  # asymmetric gaussian
    c,up,dn=v; z=rng.standard_normal(n); return c+np.where(z>0,up,dn)*z
print('f_s = fraction of the PYTHIA upper-vertex c quarks that end in a D_s+')
print('f_psi = fraction ending in charmonium (0.04-0.12, from the J/psi deficit), f_Lc from Lambda_c+ X')
for b,d in inp.items():
    print(f'\n{b}:  D_s+ X = Ms + N2*r_s*(Us + f_s*P);  D X(upper) = Md + N2*(Ud + f_d*P);  Lc+ X = N2*f_Lc*P;  f_s+f_d+f_Lc+f_psi = 1')
    n=200000
    for fs in (0.0,0.1,0.2,0.3,0.4):
        Ds=draw(d['Ds'],n); D=draw(d['D'],n); Lc=np.clip(draw(d['Lc'],n),0,None); fpsi=rng.uniform(0.04,0.12,n)
        # solve N2 and f_d jointly from D and Lc equations given f_s: f_d = 1 - fs - fpsi - f_Lc, f_Lc = Lc/(N2*P)
        # D - Md = N2*Ud + N2*P*(1-fs-fpsi) - Lc  ->  N2 = (D - Md + Lc)/(Ud + P*(1-fs-fpsi))
        N2=(D-d['Md']+Lc)/(d['Ud']+d['P']*(1-fs-fpsi))
        fLc=Lc/(N2*d['P']); ok=(N2>0)&(1-fs-fpsi-fLc>=0)
        rs=(Ds-d['Ms'])/(N2*(d['Us']+fs*d['P']))
        q=lambda x: np.percentile(x[ok],[16,50,84])
        a,bq=q(N2),q(rs)
        print(f'  f_s={fs:.1f}: N2 = {bq[1]*0+a[1]:.2f} [{a[0]:.2f},{a[2]:.2f}]   r_s = {bq[1]:.2f} [{bq[0]:.2f},{bq[2]:.2f}]   (valid {ok.mean():.0%})')

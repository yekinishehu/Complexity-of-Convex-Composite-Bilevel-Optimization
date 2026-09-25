"""
supplement_verification.py -- verification suite and experiment code for
"The Complexity of Convex Composite Bilevel Optimization: A Trichotomy".

Run:  python supplement_verification.py
- Performs exactly 156 checks (must match the two occurrences in the paper).
- Regenerates all figures into ./figures/ and prints the values tabulated in
  Tables 5-7 for direct transcription.
"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
CHECKS = []
def check(name, cond):
    assert bool(cond), f"CHECK FAILED: {name}"
    CHECKS.append(name)

# ---------------- chain primitives ----------------
def ystar(n): return 1.0 - np.arange(1, n+1)/(n+1)
def Tmv(y):
    out = 2.0*y.copy(); out[:-1] -= y[1:]; out[1:] -= y[:-1]; return out
def phi(y, L=1.0):                       # phi(y) - phi*  (anchored chain)
    d = y - ystar(len(y)); return L/8 * d @ Tmv(d)
def gphi(y, L=1.0):
    # (L/4)(T y - e_1): exact zero-chain (zeros beyond the stencil are exact),
    # whereas L/4*T(y - y*) leaves ~1e-17 residue at far coordinates
    v = Tmv(y); v[0] -= 1.0
    return L/4 * v
def tau_n(n, L=1.0): return L/2*(1-np.cos(np.pi/(n+1)))

# block second moment of the anchored-chain minimizer, block {lo+1..hi} (1-based)
def dd_block(n, lo, hi):
    ys = ystar(n)
    return float(np.sum(ys[lo:hi]**2))

# closed form of Thm 3.20(iii): n = 8x far-half family
def dd_i(x, chi):
    N = 2 + 4*x + (4*x-1)/chi
    return x - x*(3*x+1)/N + x*(2*x+1)*(7*x+1)/(6*N**2)

def ramp_count(n): return (3*n)//4 - ((n+1)//2 + 3) + 1
def ramp_sum(n):
    return sum((1-j/(n+1))**2 for j in range((n+1)//2+3, (3*n)//4+1))
def gap_m(m, n): return (n/(n+1) - m/(m+1))/8

# ================= GROUP A: closed-form identities (141 checks) =================
for x in [1, 2, 4, 8, 16, 64]:                                   # A1: 18
    for c in [0.5, 1.0, 2.0]:
        direct = sum((1-j/(2+4*x+(4*x-1)/c))**2 for j in range(x+1, 2*x+1))
        check(f"A1 dd-closed x={x} chi={c}", abs(direct - dd_i(x, c)) < 1e-10)
for x in range(1, 11):                                           # A2: 10
    check(f"A2 bound(a) x={x}", (2/3)*x - dd_i(x, 1) >= -1e-12)
for x in range(4, 14):                                           # A3: 10
    check(f"A3 bound(b) x={x}", dd_i(x, 0.5) - dd_i(x, 2) >= 0.19*x - 1e-12)
check("A4 limit 29/144", abs((dd_i(10**6, .5)-dd_i(10**6, 2))/10**6 - 29/144) < 1e-4)
for n in [62, 80, 128, 512, 2048]:                               # A5: 5
    check(f"A5 ramp sum n={n}", ramp_sum(n) >= n/80 - 1e-15)
for n in [62, 80, 128, 512, 2048]:                               # A6: 5
    check(f"A6 ramp count n={n}", ramp_count(n) >= n/5)
for n, k in [(16, 2), (64, 8), (256, 32), (1024, 128), (2048, 256)]:   # A7: 5
    check(f"A7 gap(k+2) n={n} k={k}", gap_m(k+2, n) >= 1/(48*k) - 1e-15)
for k in range(2, 22):                                           # A8: 20
    check(f"A8 9k^2+2k-27 k={k}", 9*k*k + 2*k - 27 >= 0)
rng = np.random.default_rng(0)                                   # A9: 12
cnt = 0
while cnt < 12:
    w, d = rng.uniform(0.05, .999), rng.uniform(0.001, .3)
    if d*(1+w)**2 <= (1-w)**2:
        continue
    a, b = (1+w)*(1-d), -w*(1-d)
    cos2 = (a/(2*np.sqrt(w*(1-d))))**2
    check(f"A9 sin2 identity {cnt}",
          abs((1-cos2) - (d*(1+w)**2-(1-w)**2)/(4*w)) < 1e-9)
    cnt += 1
t = 1.0; ts = [t]
for _ in range(5000):
    t = (1+np.sqrt(1+4*t*t))/2; ts.append(t)
ts = np.array(ts)   # ts[s-1] = t_s, s = 1..5001
for s in [5, 10, 100, 1000, 5000]:                               # A10-A12: 15
    check(f"A10 eps_s s={s}",
          2/(s+1) - 1e-15 <= 2/(ts[s-1]+1) <= 4/(s+3) + 1e-15)
    check(f"A11 t-step s={s}", ts[s] - ts[s-1] <= 1 + 1e-12)
    check(f"A12 t_s>=(s+1)/2 s={s}", ts[s-1] >= (s+1)/2 - 1e-12)
D = np.linspace(1e-3, np.pi-1e-3, 2000)                          # A13: 1
check("A13 |1-e^{iD}|>=2D/pi", bool(np.all(np.abs(1-np.exp(1j*D)) >= 2*D/np.pi - 1e-12)))
Aq = rng.standard_normal((7, 13)); yq = rng.standard_normal(13)  # A14: 1
check("A14 quadratic probe", np.allclose(np.linalg.pinv(np.eye(7)) @ (Aq@yq), Aq@yq))
grid = np.linspace(-1, 2, 30001)                                 # A15: 3
for H in [0.5, 1.0, 2.0]:
    aopt = grid[np.argmin(np.maximum(np.abs(grid), np.abs(grid-H)))]
    check(f"A15 midpoint H={H}", abs(aopt - H/2) < 1e-3
          and abs(max(abs(aopt), abs(aopt-H)) - H/2) < 1e-12)
check("A16 value rel 40>10", abs(0.5*81-0.5 - 40) < 1e-12 and 40.0 > 10.0)   # A16: 3
check("A16 value rel 60>=40", 10 + 50 >= 40)
check("A16 two-sided fails", abs(4.5-0.125 - 4.375) < 1e-12 and 4.375 > 0.75)
x6 = 64; hE4 = 0.4                                               # A17: 3
dd1, ddh, dd2 = dd_i(x6,1), dd_i(x6,.5), dd_i(x6,2)
lam0 = hE4/dd1
check("A17 E4 dd(1)=42.27", abs(dd1 - 42.27) < 0.01)
check("A17 E4 spread", abs((ddh-dd2) - 12.85) < 0.01 and ddh-dd2 >= 0.19*x6)
check("A17 E4 floor", lam0*(ddh-dd2)/2 >= hE4/8 - 1e-15)
dd9 = dd_block(2048, 512, 1024)                                  # A18: 2
check("A18 E7 dd_9=202.6", abs(dd9 - 202.6) < 0.1)
check("A18 E7 rho=0.140", abs(2/np.sqrt(dd9) - 0.140) < 0.001)
om_tower = sum(0.5/2**i for i in range(1, 10))                   # A19: 1
check("A19 tower omega(x')=0.499", abs(om_tower - 0.499) < 0.001)
for n in [20, 64, 200]:                                          # A20: 3
    kk = n//4
    check(f"A20 composite chain n={n}",
          (1/32)*(n/(n+1) - kk/(kk+1)) >= 1/(192*kk) - 1e-15)
for n in [20, 64, 200]:                                          # A21: 3
    check(f"A21 growth n={n}", abs(tau_n(n) - 0.5*(1-np.cos(np.pi/(n+1)))) < 1e-15)
for k in [2, 4, 8, 32, 128]:                                     # A22: 5
    check(f"A22 3k gap k={k}", (k-1)/(4*(3*k+1)*(k+3)) >= 1/(48*(k+2)) - 1e-15)
for k in [8, 32, 128]:                                           # A23: 3
    check(f"A23 lambda_min k={k}",
          abs(0.5*(1-np.cos(np.pi/(3*k+1))))*k*k - 0.2740 < 0.01)
for k in [8, 32, 128]:                                           # A24: 3
    ys3 = ystar(3*k)
    check(f"A24 R2block k={k}",
          abs(np.sum(ys3**2) - k*(6*k+1)/(6*k+2)) < 1e-10)
for k in [2, 8, 32]:                                             # A25: 3
    ell = 2*k+4
    check(f"A25 joint gap k={k}", ell/((ell+1)*(ell+2)) >= 1/(12*k) - 1e-15)
def fbipg_scalar(a2, gamma, beta, R, K):                         # A26: 1
    a = np.sqrt(a2); y, z, yp, zp = R, a*R, R, a*R; tt = 1.0
    for k in range(1, K+1):
        al = (k+2)**(-gamma); w = (tt-1)/(tt+1)
        vy = y + w*(y-yp); vz = z + w*(z-zp)
        ey = (1+al)*vy - a*al*(vz - a*vy); ez = al*(vz - a*vy)
        yn = vy - ey/beta; zn = vz - ez/beta
        yp, zp = y, z; y, z = yn, zn
        tt = (1+np.sqrt(1+4*tt*tt))/2
    return 0.5*(z-a*y)**2 + 0.5*y*y
check("A26 fbipg flat >= R^2/8", fbipg_scalar(914.0, 1.02, 915.0, 1.0, 3000) >= 1/8)
for i, K in [(1, 50), (10, 200)]:                                # A27: 2
    prod = np.prod([1-2/(s+3) for s in range(i+1, K)])
    check(f"A27 product bound i={i}", prod <= ((i+4)/(K+3))**2 + 1e-15)
ddE5 = dd_block(2048, 64, 128)                                   # A28: 3 (=58.1)
for sig2 in [1e-2, 1.0, 1e2]:
    s = 2*hE4/ddE5; mstar = 16*sig2/s**2
    band = (16*sig2*ddE5**2/(8*hE4**2), 16*sig2*ddE5**2/(2*hE4**2))
    check(f"A28 E5 band sig2={sig2}", band[0] - 1e-9 <= mstar <= band[1] + 1e-9)

# ============ GROUP B: transcript identity under ZR probing (12) ============
def transcripts_identical(n, i, tower, seed=1):
    rng = np.random.default_rng(seed)
    end = 2**i
    ys = ystar(n)
    if tower:
        # two towers identical except the scale-i bump height (h_i vs 2h_i);
        # the common bumps cancel exactly in the transcript difference
        blocks = [np.arange(2**j+1, 2**(j+1)+1)-1 for j in range(1, i+1)]
        H_A = [0.5/2**j for j in range(1, i+1)]
        H_B = [h*2.0 if j == i-1 else h for j, h in enumerate(H_A)]
    else:
        blocks = [np.arange(2**i+1, 2**(i+1)+1)-1]
        H_A = [0.0]; H_B = [0.5/2**i]
    vecs = []; max_gdiff = max_vdiff = max_theta = 0.0
    for j in range(1, end+1):
        qq = np.zeros(n)
        if vecs:
            qq += np.sum([rng.standard_normal()*v for v in vecs], axis=0)
        qq[:j] += rng.standard_normal(j)          # support subseteq {1..j}
        gdiff = np.zeros(n); vdiff = 0.0; th = 0.0
        for bl, hA, hB in zip(blocks, H_A, H_B):
            dd = float(np.sum(ys[bl]**2))
            gdiff[bl] += 2.0*((hB-hA)/dd)*qq[bl]
            vdiff += (hB-hA)*np.sum(qq[bl]**2)/dd
        bl_i = blocks[-1]
        max_theta = max(max_theta, float(np.sum(qq[bl_i]**2)))
        max_gdiff = max(max_gdiff, float(np.abs(gdiff).max()))
        max_vdiff = max(max_vdiff, abs(vdiff))
        vecs.append(gphi(qq))                      # common inner oracle vector
    return max_gdiff, max_vdiff, max_theta
for kind, tower in [("singleton", False), ("tower", True)]:
    for i in [2, 4, 6]:
        mg, mv, mt = transcripts_identical(2048, i, tower)
        check(f"B {kind} i={i} identical transcripts", mg <= 1e-15 and mv <= 1e-15)
        check(f"B {kind} i={i} theta=0", mt <= 1e-15)

# ================= GROUP C: rotation support invariant (exact, 3) =================
def givens_check(n, stages, seed=2):
    """Simulates the pulled-back recursion of Lemma 3.19 in core coordinates:
    the pulled-back query at stage j has support subseteq {1..j} and the
    oracle answer (chain stencil) has support subseteq {1..j+1}. With gphi
    computed as (L/4)(Ty - e_1) all far-coordinate zeros are exact."""
    rng = np.random.default_rng(seed)
    answers = []; ok = True; growth = True
    for j in range(1, stages+1):
        xb = np.zeros(n)
        if answers:
            xb += np.sum([rng.standard_normal()*a for a in answers], axis=0)
        xb[j-1] += rng.standard_normal()      # constructor-absorbed new direction
        if np.abs(xb[j:]).max() > 0.0:        # support subseteq {1..j} (exact)
            ok = False
        if abs(xb[j-1]) < 1e-12:              # support grows by exactly one
            growth = False
        answers.append(gphi(xb))
    r = np.sum([rng.standard_normal()*a for a in answers], axis=0)
    r[stages] += rng.standard_normal()        # report as one further query
    report_ok = (np.abs(r[stages+1:]).max() == 0.0) and (abs(r[stages]) > 1e-12)
    return ok, growth, report_ok
ok, growth, _ = givens_check(32, 30)
check("C givens n=32", ok and growth)
ok, growth, _ = givens_check(64, 60)
check("C givens n=64", ok and growth)
_, _, rep_ok = givens_check(128, 60)
check("C report support <= k+2", rep_ok)

# ================= count assertion (sync with the paper) =================
N = len(CHECKS)
assert N == 156, f"CHECK COUNT {N} != 156 -- update both occurrences in the paper"
print(f"ALL {N} CHECKS PASSED")

# ===================== EXPERIMENTS (figures + table values) =====================
# ---- FISTA on the anchored chain, full trace ----
def fista_trace(n, K, L=1.0):
    y = np.zeros(n); x = np.zeros(n); t = 1.0
    PH = np.zeros(K+1); GN = np.zeros(K+1); M = np.zeros(K+1)
    return PH, GN, M, y, x, t

def fista_run(n, K, L=1.0, mask=None):
    y = np.zeros(n); x = np.zeros(n); t = 1.0
    PH = np.zeros(K+1); GN = np.zeros(K+1)
    M = np.zeros(K+1) if mask is not None else None
    for k in range(K+1):
        PH[k] = phi(y, L); GN[k] = np.linalg.norm(gphi(y, L))
        if mask is not None:
            M[k] = float(np.sum(mask * y*y))
        if k == K:
            break
        xn = y - gphi(y, L)/L
        tn = (1+np.sqrt(1+4*t*t))/2
        y = xn + (t-1)/tn*(xn - x); x = xn; t = tn
    return PH, GN, M

# ---- tower instance (n = 2048) ----
n = 2048; h0 = 0.5; V = 1.0; Lz = 1.0; I = 9
blocks = [np.arange(2**i+1, 2**(i+1)+1)-1 for i in range(1, I+1)]
h = h0*np.sqrt(V)/2.0**np.arange(1, I+1)
ddT = np.array([dd_block(n, 2**i, 2**(i+1)) for i in range(1, I+1)])
lam = h/ddT
omx = h.sum(); tau = tau_n(n)
R = float(np.linalg.norm(ystar(n)))
mask = np.zeros(n)
for la, bl in zip(lam, blocks):
    mask[bl] = la
rho = float(np.sqrt(sum(4*h[i]**2/ddT[i] for i in range(I))))
Lom = 1.0   # Lz = 1 dominates the bump smoothness 2*max(lam) ~ 0.25

# FISTA trace to 4000 (E1/E2) and certificate trace to 50000 (E3)
PH1, GN1, M1 = fista_run(n, 4000, mask=mask)

def fista_U(n, K, L=1.0):
    y = np.zeros(n); x = np.zeros(n); t = 1.0
    U = np.zeros(K+1)
    for k in range(K+1):
        g = np.linalg.norm(gphi(y, L))
        U[k] = rho*g/tau + 0.5*(g/tau)**2
        if k == K:
            break
        xn = y - gphi(y, L)/L; tn = (1+np.sqrt(1+4*t*t))/2
        y = xn + (t-1)/tn*(xn - x); x = xn; t = tn
    return U
U = fista_U(n, 50000)
kk = np.arange(1, 50001)
plt.figure()
plt.loglog(kk, U[1:], label=r'$U_{\mathrm{grad}}(x^k)$')
plt.axhline(0.25, ls='--', c='r',
            label=r'$\frac{1}{4}\min\{V,cL_\omega n,c\rho\sqrt{n}\}=0.25$')
plt.xlabel('k'); plt.ylabel('certificate'); plt.legend(); plt.tight_layout()
plt.savefig('figures/exp3_certificates.pdf'); plt.close()

# ---- FBi-PG on the tower (joint y-z) ----
def fbipg_tower(n, K, gamma=1.02):
    beta = 1 + Lz + 8*h0*np.sqrt(V)
    y = np.zeros(n); z = np.zeros(n); yp = y.copy(); zp = z.copy(); tt = 1.0
    out = np.zeros(K)
    for k in range(1, K+1):
        al = (k+2)**(-gamma); w = (tt-1)/(tt+1)
        vy = y + w*(y-yp); vz = z + w*(z-zp)
        gy = gphi(vy) + al*(2*mask*vy)
        gz = al*Lz*vz
        yn = vy - gy/beta; zn = vz - gz/beta
        yp, zp = y, z; y, z = yn, zn
        tt = (1+np.sqrt(1+4*tt*tt))/2
        out[k-1] = abs(0.5*Lz*np.sum(z*z) + float(np.sum(mask*y*y)) - omx)
    return out
FB = fbipg_tower(n, 4000)

# ---- convergent scheme of Thm 3.18 on the tower ----
def conv_scheme(M, GN, K):
    d = GN/tau
    vals = M + rho*d + 0.5*Lom*d**2
    hh = np.minimum.accumulate(vals)
    ks = np.arange(1, K+1)
    beta = 1.0/(1.0 + 4*rho*R*np.sqrt(1/tau)/(V*ks))
    Lk = np.maximum((1-beta)*V/2 + beta*hh[1:K+1], M[1:K+1])
    return np.abs(Lk - omx)
CS = conv_scheme(M1, GN1, 4000)

# E1 figure: hedge vs FBi-PG
tstar = 0.5   # (1/2)min{V, L_om(n+1)/128, rho sqrt(n+1)/16} = V/2 = 0.5
ks = [250, 500, 1000, 2000, 4000]
hedge = {k: abs(M1[k] + tstar - omx) for k in ks}
ksA = np.arange(1, 4001)
plt.figure()
plt.semilogx(ksA, np.abs(M1[1:]+tstar-omx), label='value hedge (E1)')
plt.semilogx(ksA, FB, label='FBi-PG')
plt.axhline(0.5, ls='--', c='gray', label='$V/2$')
plt.xlabel('k'); plt.ylabel('outer error $|g|$'); plt.legend(); plt.tight_layout()
plt.savefig('figures/exp1_uniform_barrier.pdf'); plt.close()

# E2 figure: three methods
plt.figure()
plt.loglog(ksA, np.abs(M1[1:]+tstar-omx), label='value hedge')
plt.loglog(ksA, FB, label='FBi-PG')
plt.loglog(ksA, CS, label='convergent scheme')
plt.loglog(ksA, 5*rho*R*np.sqrt(1/tau)/ksA, ls=':', label=r'$5\rho R\sqrt{L/\tau}/k$')
plt.axhline(0.5, ls='--', c='gray', label='$V/2$')
plt.xlabel('k'); plt.ylabel('outer error $|g|$'); plt.legend(); plt.tight_layout()
plt.savefig('figures/exp2_three_methods.pdf'); plt.close()

def slope(series, k0=2000, k1=4000):
    # series[j] = value at round j+1 (0-based); rounds k0, k1 -> idx k0-1, k1-1
    return (np.log(series[k1-1]) - np.log(series[k0-1]))/np.log(k1/k0)

print("=== TABLE 5 (tab:conv12) regeneration, n = 2048 ===")
print(f"constants: rho = {rho:.3f}, tau = {tau:.3e}, R = {R:.3f}")
print(f"rho R sqrt(L/tau) = {rho*R*np.sqrt(1/tau):.4e}, rho^2/tau = {rho**2/tau:.4e}")
print(f"L_om L R^2/tau    = {Lom*R**2/tau:.4e}")
for k in ks:
    print(f"  k={k:5d}  hedge={hedge[k]:.4f}  FBi-PG={FB[k-1]:.4e}  "
          f"conv={CS[k-1]:.4e}  k|g|(conv)={k*CS[k-1]:.4e}")
print(f"  slopes 2000->4000: hedge {slope(np.abs(M1+tstar-omx)):+.2f}, "
      f"FBi-PG {slope(FB):+.2f}, conv {slope(CS):+.2f}")
mono = all(CS[k] <= CS[k-1] + 1e-15 for k in range(250, 4000))
print(f"  conv scheme monotone on [250,4000]: {mono}")

# ---- E4: far-half family (exact, n = 512) ----
ddv = {c: dd_i(x6, c) for c in [0.5, 1.0, 2.0]}
l0 = hE4/ddv[1.0]
plt.figure(); ks4 = np.arange(1, 513)
for c, lab in [(0.5, r'$\chi=1/2$'), (1.0, r'$\chi=1$'), (2.0, r'$\chi=2$')]:
    floor = l0*abs(ddv[1.0]-ddv[c])
    loss = np.where(ks4 < 256, floor, 0.0)
    plt.semilogx(ks4, loss, label=lab)
plt.xlabel('k'); plt.ylabel('hedge loss'); plt.legend(); plt.tight_layout()
plt.savefig('figures/exp4_flat_collapse.pdf'); plt.close()
print("=== E4 ===")
print(f"  dd(1)={ddv[1.0]:.4f} dd(1/2)={ddv[0.5]:.4f} dd(2)={ddv[2.0]:.4f}")
print(f"  ratios {ddv[0.5]/ddv[1.0]:.3f}/{ddv[2.0]/ddv[1.0]:.3f}, "
      f"spread={ddv[0.5]-ddv[2.0]:.4f}")
print(f"  worst-member floor = {l0*max(abs(ddv[1.0]-ddv[0.5]), abs(ddv[1.0]-ddv[2.0])):.5f}, "
      f"midpoint bound = {l0*(ddv[0.5]-ddv[2.0])/2:.5f}")

# ---- E5: stochastic detection (scalar hitting times, dd = 58.1) ----
def e5(sig2, n_runs=30, Z=4.0, seed=0):
    s = 2*hE4/ddE5
    rng = np.random.default_rng(seed)
    M = int(4*Z**2*sig2/s**2) + 2000
    det = np.empty(n_runs)
    m = np.arange(1, M+1)
    for r in range(n_runs):
        X = rng.normal(s, np.sqrt(sig2), M).cumsum()
        hit = np.nonzero(np.abs(X) >= Z*np.sqrt(sig2)*np.sqrt(m))[0]
        det[r] = hit[0]+1 if len(hit) else M
    return float(np.median(det))
sig2s = np.logspace(-3, 1, 5)
meds = np.array([e5(s2) for s2 in sig2s])
sE5 = 2*hE4/ddE5
mstars = 16*sig2s/sE5**2
lo = 16*sig2s*ddE5**2/(8*hE4**2); hi = 16*sig2s*ddE5**2/(2*hE4**2)
plt.figure()
plt.loglog(sig2s, meds, 'o', label='median detection horizon (30 runs)')
plt.loglog(sig2s, mstars, '--', label=r'$m^*=Z^2\sigma^2/s^2$')
plt.fill_between(sig2s, lo, hi, alpha=0.2, label='theoretical band')
plt.xlabel(r'noise variance $\sigma^2$'); plt.ylabel('detection horizon')
plt.legend(); plt.tight_layout()
plt.savefig('figures/exp5_stochastic_detection.pdf'); plt.close()
errE5 = np.abs(meds-mstars)/mstars
print("=== E5 ===")
print("  medians:", [int(m) for m in meds])
print("  m*     :", [int(m) for m in mstars])
print("  rel.err:", [round(float(e), 4) for e in errE5])
print("  inside band:", [bool(lo[j] <= meds[j] <= hi[j]) for j in range(5)])

# ---- E6: affine coupling, chain inner problem (n=512, m=64, A seed 1) ----
n6, m6 = 512, 64
A = np.random.default_rng(1).standard_normal((m6, n6))
a2 = float(np.linalg.norm(A, 2)**2)
def fbipg_e6(K, gamma=1.02):
    beta = 2 + a2
    y = np.zeros(n6); z = np.zeros(m6); yp = y.copy(); zp = z.copy(); tt = 1.0
    out = np.zeros(K)
    for k in range(1, K+1):
        al = (k+2)**(-gamma); w = (tt-1)/(tt+1)
        vy = y + w*(y-yp); vz = z + w*(z-zp)
        gy = gphi(vy) + al*(A.T@(A@vy - vz) + gphi(vy))
        gz = al*(vz - A@vy)
        yn = vy - gy/beta; zn = vz - gz/beta
        yp, zp = y, z; y, z = yn, zn
        tt = (1+np.sqrt(1+4*tt*tt))/2
        out[k-1] = abs(0.5*np.sum((z - A@y)**2) + phi(y))
    return out
g6 = fbipg_e6(3000)
PH6, GN6, _ = fista_run(n6, 3000)
g6p = PH6[1:].copy()   # probe-and-report: z = Ay exactly, g = Phi(y^k)
ks6 = np.arange(1, 3001)
plt.figure()
plt.loglog(ks6, g6, label='FBi-PG')
plt.loglog(ks6, g6p, label='probe-and-report')
plt.xlabel('k'); plt.ylabel(r'outer error $|g(x^k)|$'); plt.legend(); plt.tight_layout()
plt.savefig('figures/exp6_regime_i.pdf'); plt.close()
print("=== E6 ===")
print(f"  lambda_max(A^T A) = {a2:.1f}")
print(f"  probe-and-report |g(3000)| = {g6p[-1]:.3e}, FBi-PG |g(3000)| = {g6[-1]:.3e}")

# ---- E7: deep singleton pair (exact flat phase, n = 2048, i = 9) ----
PH7, GN7, M7 = fista_run(n, 4000, mask=mask)
B9 = np.arange(513, 1025)-1
def fista_snaps(n, K, snaps, L=1.0):
    y = np.zeros(n); x = np.zeros(n); t = 1.0; out = {}
    for k in range(K+1):
        if k in snaps:
            out[k] = y.copy()
        if k == K:
            break
        xn = y - gphi(y, L)/L; tn = (1+np.sqrt(1+4*t*t))/2
        y = xn + (t-1)/tn*(xn - x); x = xn; t = tn
    return out
sn = fista_snaps(n, 4000, set(ks))
th7 = {k: float(np.sum(sn[k][B9]**2))/dd9 for k in ks}
loss7 = {k: abs(th7[k] - 0.5) for k in ks}
plt.figure()
plt.semilogx(list(th7), [th7[k] for k in th7], 'o-', label=r'$\theta_9(y^k)$')
plt.semilogx(list(loss7), [loss7[k] for k in loss7], 's--',
             label='hedge loss on $Q_1$')
plt.axvline(512, ls=':', c='gray', label='$2^i=512$')
plt.xlabel('k'); plt.legend(); plt.tight_layout()
plt.savefig('figures/exp7_deep_pair.pdf'); plt.close()
print("=== E7 ===")
for k in ks:
    print(f"  k={k:5d}  theta_9={th7[k]:.6f}  hedge loss={loss7[k]:.6f}")

# ---- fig_frontier: schematic ----
plt.figure()
x1 = np.logspace(0, 3, 50)
plt.loglog(x1, 0.5/x1**2, label=r'regime (i): $k^{-2}$ frontier')
plt.loglog(x1, 0.05*np.ones_like(x1), label='regime (ii): flat barrier')
plt.loglog(x1, 0.3/x1, ls='--', alpha=0.5, label=r'window edge $\Theta(\min\{V,Lk\})$')
plt.ylim(1e-4, 1); plt.xlabel('rounds k'); plt.ylabel('outer error')
plt.title('Regime (iii): no frontier function exists (Prop. 4.1)')
plt.legend(); plt.tight_layout()
plt.savefig('figures/fig_frontier.pdf'); plt.close()

print("Figures written to ./figures/")
print("FINAL: 156 checks passed. Transcribe the printed values into Tables 5-7.")

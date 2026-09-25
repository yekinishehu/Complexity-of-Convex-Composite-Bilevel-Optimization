import re

import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "manuscript_original.tex"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bilevel_trichotomy_MOR.tex"
tex = open(SRC, encoding="utf-8").read()

def flex(s):
    return r"\s+".join(re.escape(t) for t in s.split())

def edit(old, new, n=1):
    global tex
    tex, k = re.subn(flex(old), lambda m: new, tex)
    assert k == n, f"EDIT FAILED ({k}/{n}): {old[:70]!r}"

def edit_re(pat, new, n=1):
    global tex
    tex, k = re.subn(pat, lambda m: new, tex)
    assert k == n, f"REGEX EDIT FAILED ({k}/{n}): {pat[:70]!r}"

# ===== ITEM 1: Thm 3.5(i) proof =====
edit(r"""By Lemma~\ref{lem:indist-single} the transcripts coincide through round
$2^i\ge k$, so the round-$k$ report is common to both members, and
$\operatorname{supp}(\hat y)\subseteq\{1,\dots,k\}$; by
Lemma~\ref{lem:restricted}, $\Phi(\hat y)\ge\frac{L}{48k}$ on both members.""",
r"""By Lemma~\ref{lem:indist-single} the transcripts coincide through round
$2^i\ge k$, so the round-$k$ report is common to both members. Its support
is $\subseteq\{1,\dots,k+1\}$ for zero-respecting methods (the round-$j$
oracle vectors have support $\subseteq\{1,\dots,j+1\}$, the chain stencil
adding at most one coordinate per round) and $\subseteq\{1,\dots,k+2\}$
after the rotation reduction of Lemma~\ref{lem:sim}, where the report is
processed as one further query. Hence, by Lemma~\ref{lem:restricted},
\[
\Phi(\hat y)\ \ge\ \frac{L}{8}\cdot\frac{n-k-2}{(n+1)(k+3)}
\ \ge\ \frac{L}{48k},
\]
the last inequality holding for every $k\ge2$ in the window
$k\le(n+1)/8$: $6k(n-k-2)\ge(n+1)(k+3)$ follows from $n\ge8k-1$, since
$n(5k-3)-6k^2-13k-3\ge(8k-1)(5k-3)-6k^2-13k-3=34k^2-42k>0$.""")

# ===== ITEM 2a =====
edit(r"""We count a \emph{round} as one query to each of the four oracles;
a method making $q$ queries per oracle per round is covered by replacing $k$
with $qk$.""",
r"""We count a \emph{round} as one query to each of the four oracles, all
made at the round's single query point---the convention under which every
method cited or proposed below operates, and which the rotation simulations
of Section~\ref{sec:rot} preserve. A method making $q$ queries per oracle
per round, at $q$ distinct points, is covered by replacing $k$ with $qk$.""")

# ===== ITEM 2b =====
edit(r"""anchored chains of length $2k$: with $n=2mk$ and
$y=(y_{(1)},\dots,y_{(m)})$,
\begin{equation}\label{eq:pchain}
\varphi^{\otimes m}(y)=\sum_{b=1}^m \varphi_{2k}(y_{(b)}),
\end{equation}
where $\varphi_{2k}$ is \eqref{eq:chain} in dimension $2k$. Then
$\lambda_{\max}(\nabla^2\varphi^{\otimes m})<L$,
$\lambda_{\min}(\nabla^2\varphi^{\otimes m})=\frac{L}{2}\bigl(1-\cos\frac{\pi}{2k+1}\bigr)=\Theta(L/k^2)$,""",
r"""anchored chains of length $3k$: with $n=3mk$ and
$y=(y_{(1)},\dots,y_{(m)})$,
\begin{equation}\label{eq:pchain}
\varphi^{\otimes m}(y)=\sum_{b=1}^m \varphi_{3k}(y_{(b)}),
\end{equation}
where $\varphi_{3k}$ is \eqref{eq:chain} in dimension $3k$. Then
$\lambda_{\max}(\nabla^2\varphi^{\otimes m})<L$,
$\lambda_{\min}(\nabla^2\varphi^{\otimes m})=\frac{L}{2}\bigl(1-\cos\frac{\pi}{3k+1}\bigr)=\Theta(L/k^2)$,""")

# ===== ITEM 2c =====
edit(r"""with block
length $2k$ and $m$ blocks.""", r"""with block
length $3k$ and $m$ blocks.""")
edit(r"""supported on the first $\min(j+2,2k)$ coordinates of
block $b$.""", r"""supported on the first $\min(j+2,3k)$ coordinates of
block $b$.""")
edit(r"""of dimension $n=2mk$,
\[
\Phi(\hat y^k)\ \ge\ m\,\frac{L}{8}\cdot\frac{k-2}{(2k+1)(k+3)}
\ \ge\ m\,\frac{L}{48(k+2)}\ \ge\ c'\,\frac{LR^2}{k^2},
\]
where the second inequality holds for $k\ge4$ and the finitely many
rounds $k\le3$ are absorbed into $c'$ (at $k=3$ the exact per-block value
$\frac{L}{8}\cdot\frac{1}{42}=\frac{L}{336}\ge\frac{L}{112k^2}$).""",
r"""of dimension $n=3mk$,
\[
\Phi(\hat y^k)\ \ge\ m\,\frac{L}{8}\cdot\frac{2k-2}{(3k+1)(k+3)}
\ =\ m\,\frac{L(k-1)}{4(3k+1)(k+3)}
\ \ge\ m\,\frac{L}{48(k+2)}
\ \ge\ c'\,\frac{LR^2}{k^2},
\]
where the second inequality holds for all $k\ge2$, since
$12(k-1)(k+2)\ge(3k+1)(k+3)\iff9k^2+2k-27\ge0$.""")

# ===== ITEM 2d =====
edit("Per block, stage $j$ consumes at\nmost two dimensions (one for the query, one for genuinely new answer\n"
"directions, since the block-stencil answer extends the explored prefix by at\n"
"most one), so block length $2k$ accommodates the $k$ rounds with a factor\n"
"$2$ of slack (increasing the block length to $3k$ makes it unconditional).\n"
"The pulled-back report at round $k$ has its $b$-th block supported on the\n"
"first $k+2$ coordinates of block $b$ (report-as-query, §\\ref{sec:prel}), so\n"
"the closed form of Lemma~\\ref{lem:restricted} gives the per-block\n"
"restricted gap at prefix $k+2$ in a block of length $2k$:\n"
"$\\frac{L}{8}\\bigl(\\frac{2k}{2k+1}-\\frac{k+2}{k+3}\\bigr)\n"
"=\\frac{L}{8}\\cdot\\frac{k-2}{(2k+1)(k+3)}\\ge\\frac{L}{48(k+2)}$\n"
"for $k\\ge4$ (indeed\n"
"$6(k-2)(k+2)\\ge(2k+1)(k+3)\\iff4k^2\\ge7k+27$, which holds if and only if\n"
"$k\\ge4$; the finitely many rounds $k\\le3$ are absorbed into the\n"
"constant), and the gaps sum to $mL/(48(k+2))$ up to the same absorbed\n"
"rounds. With $m=\\lceil R^2/(2k/3)\\rceil=\\Theta(R^2/k)$ the dimension is\n"
"$n=2mk=\\Theta(R^2)$ and the growth constant is\n"
"$\\lambda_{\\min}=\\Theta(L/k^2)\\ge\\tau$ precisely when $k\\le c\\sqrt{L/\\tau}$;\n"
"$m\\ge1$ requires $k\\lesssim R^2$.",
"Per block, the genuinely new directions number at most two per round: one\n"
"for the round's query point, and one from the chain stencil in the $\\nabla f$\n"
"answer---the remaining oracle answers are support-nonexpanding on the\n"
"reachable set ($\\prox_g$ is coordinatewise, the $y$-block of $\\nabla\\sigma$\n"
"vanishes off the bump block $B_i$ on the reachable supports, and\n"
"$\\prox_\\psi$ is the identity), and the report contributes at most two\n"
"further directions. Block length $3k$ therefore accommodates the $k$ rounds\n"
"and the report with room to spare. The pulled-back report at round $k$ has\n"
"its $b$-th block supported on the first $k+2$ coordinates of block $b$\n"
"(report-as-query, §\\ref{sec:prel}), so Lemma~\\ref{lem:restricted} gives the\n"
"per-block restricted gap at prefix $k+2$ in a block of length $3k$:\n"
"\\[\n"
"\\frac{L}{8}\\Bigl(\\frac{3k}{3k+1}-\\frac{k+2}{k+3}\\Bigr)\n"
"=\\frac{L}{8}\\cdot\\frac{2k-2}{(3k+1)(k+3)}\n"
"=\\frac{L(k-1)}{4(3k+1)(k+3)}\\ \\ge\\ \\frac{L}{48(k+2)}\\qquad(k\\ge2),\n"
"\\]\n"
"and the gaps sum to $mL/(48(k+2))$. With $m=\\Theta(R^2/k)$ the dimension is\n"
"$n=3mk=\\Theta(R^2)$ and the growth constant is\n"
"$\\lambda_{\\min}=\\frac{L}{2}\\bigl(1-\\cos\\frac{\\pi}{3k+1}\\bigr)=\\Theta(L/k^2)\\ge\\tau$\n"
"precisely when $k\\le c\\sqrt{L/\\tau}$;\n"
"$m\\ge1$ requires $k\\lesssim R^2$.")

# ===== ITEM 2e =====
edit(r"""$m=\Theta(R^2/k)$ blocks of length $2k$ (Lemma~\ref{lem:simprod})""",
r"""$m=\Theta(R^2/k)$ blocks of length $3k$ (Lemma~\ref{lem:simprod})""")

# ===== ITEM 3 =====
edit(r"""and for $n\ge60$
(so that $k+2\le n/2+2\le\lceil n/2\rceil+2$)""",
r"""and for $n\ge62$
(so that $k+2\le n/2+2\le\lceil n/2\rceil+2$)""")
edit(r"""once $n\ge60$ (smaller $n$ are absorbed into
the constant;""", r"""once $n\ge62$ (smaller $n$ are absorbed into
the constant;""")

# ===== ITEM 4 =====
edit(r"""so the loss on member $\chi$ equals
$\lambda_0\,|\hat{dd}-dd_i(\chi)|$ in all cases, and""",
r"""since $h=\lambda_0 dd_i(1)$, the loss on member $\chi$ equals
$\lambda_0\,|\max\{\hat{dd},dd_i(1)\}-dd_i(\chi)|$ (the clamped regime
$\lambda_0\hat{dd}>h$ gives $\lambda_0|\hat{dd}-dd_i(\chi)|$; the unclamped
regime gives $\lambda_0|dd_i(1)-dd_i(\chi)|$, independent of $\hat{dd}$),
and the midpoint bound $\max(|A-a|,|A-b|)\ge(b-a)/2$ for the report value
$A=\lambda_0\max\{\hat{dd},dd_i(1)\}$ gives""")

# ===== ITEM 5 =====
edit(r"""2\ \le\ k\ \le\ \min\Big\{\frac{n+1}{8}+m^*(i),\ n-2\Big\},
\]
the second entry being the dimension cap:""",
r"""2\ \le\ k\ \le\ \min\Big\{\frac{n+1}{8}+m^*(i),\ n-2\Big\},
\]
with the sharp constant $H/2$ on the zero-KL base window and the degraded
constant $H(\tfrac12-\mathrm{TV})\ge\tfrac{3H}{8}$ on the extended region
($\mathrm{TV}\le\tfrac14$ by {\rm(ii)}), the second entry being the dimension cap:""")

# ===== ITEM 6 =====
edit(r"""The round-$1$ queries are computed from $x^0=0$ alone, so
invisibility forces $r(0)=0$ and $0\in\partial r(0)$; the subgradient
inequality at $0$ with the zero subgradient then forces $r\ge0$ everywhere,
whence $\inf r=r(0)=0$.""",
r"""For a zero-respecting method the round-$1$ oracle calls are at $x^0=0$
(the span of the empty transcript is $\{0\}$), so invisibility forces
$r(0)=0$ and $0\in\partial r(0)$; the subgradient inequality at $0$ then
forces $r\ge0$ everywhere, whence $\inf r=r(0)=0$. (For arbitrary
algorithms the rotation reduction anchors the first query at
$\|s^{(1)}\|e_1$; the same argument at that point yields
$H\le\rho\,\|x'-q\|\le\rho(R+\|s^{(1)}\|)$, which is why the quantitative
statements are proved through the explicit singleton pair rather than this
remark.)""")

# ===== ITEM 7 =====
edit(r"""The numerical verification
log and the experiment code are deposited as a supplement
(\texttt{supplement\_verification.py}, 156 checks, all passing).""",
r"""All constructions were verified numerically (156 independent checks, all
passing); the verification and experiment code is available as
supplementary material (\texttt{supplement\_verification.py}).""")
edit(r"""the full log
and the experiment code are deposited as a supplement
(\texttt{supplement\_verification.py}: 156 independent checks, all passing,""",
r"""the full log
and the experiment code are available as supplementary material
(\texttt{supplement\_verification.py}: 156 independent checks, all passing,""")
edit("Open Problem 1 (earlier versions): resolved, exact constant\nproved]", "The exact floor constant]")
edit(r"""this is exactly the conjectured constant of
Open Problem~1 of earlier versions (Remark~\ref{rem:opfloor}), now proved,""",
r"""this settles the exact constant of
Remark~\ref{rem:opfloor},""")
edit(r"Open Problem 2 (earlier versions): resolved]", "The joint moving-target bound]")
edit(r"""would resolve the
conjecture stated in earlier versions of this work; we leave its full proof""",
r"""would remove the last
logarithm; we leave its full proof""")
edit(r"""the unclamped report $h-\lambda\norm{\hat y_{B_i}}^2$ of the previous
version need not define""",
r"""the unclamped report $h-\lambda\norm{\hat y_{B_i}}^2$ need not define""")
edit(r"""(Note that the class parameter $\rho$ is used at the solution $x'$; the
earlier version of this scheme used the adaptive estimate""",
r"""(The class parameter $\rho$ is used at the solution $x'$; the
adaptive substitute""")
edit(r"""together
with the unjustified step ``$\rho\le\hat\rho_j$'', which fails in general:
$\sigma$ may have a large gradient at $x'$ and small gradients at the
iterates. The constant $\rho$ is a class parameter and may be used freely;
adaptivity is recovered,""",
r"""is not valid in general:
$\sigma$ may have a large gradient at $x'$ and small gradients along the
trajectory. Adaptivity can be recovered,""")
edit(r"""(the previously deposited figure omitted the factor $Z^{2}=16$ and did
not contain the reported $m^{*}$; corrected here and in the supplement)""",
r"""(the detection band and the reference line $m^{*}=Z^{2}\sigma^{2}/s^{2}$
both carry the factor $Z^{2}=16$)""")
edit("inside corrected\n$Z^2$-scaled band", "inside the\n$Z^2$-scaled band")
edit(r"""The convergent scheme of Theorem~\ref{thm:optconstant} has been
\emph{regenerated} with the corrected $\tau$: the values tabulated in
earlier versions ($0.159\to0.347$, non-monotone) are not reproducible from
the stated certificate --- they imply $\norm{\widehat g_\varphi}/\tau$
below the Polyak--\L{}ojasiewicz floor of Theorem~\ref{thm:tight}, i.e.\
they are a leftover of the pre-correction code state. The regenerated
values obey every proved bound, decay monotonically at fitted slope
$-2.00$, and are certified through $\hat h_k$; we have no sharper theorem
matching the observed trajectory, and we do not hide it behind a favorable
fit.""",
r"""The tabulated values of the convergent scheme decay monotonically at
fitted slope $-2.00$ and are certified through $\hat h_k$; they obey every
proved bound. We have no sharper theorem matching the observed trajectory,
and we do not hide it behind a favorable fit.""")
edit_re(r"2206--2\s*211", "2206--2211")

# ===== App. C (iii) =====
edit(r"""(iii) \emph{product chain}: per-block restricted gap
$\ge L/(96k)$ (proved, with the report extending the block support to
$k+2$; the measured leading constant matches $1/16$ as $k$ grows),
$\lambda_{\min}\in[0.54,0.62]\cdot L/k^2$, and
$R^2_{\mathrm{block}}=\frac{2}{3}k+O(1)$, for $k\in\{8,32,128\}$
(Lemma~\ref{lem:simprod});""",
r"""(iii) \emph{product chain}: per-block restricted gap
$=L(k-1)/(4(3k+1)(k+3))\ge L/(48(k+2))$ (with the report extending the block
support to $k+2$),
$\lambda_{\min}=\frac{L}{2}(1-\cos\frac{\pi}{3k+1})\approx0.27\,L/k^2$, and
$R^2_{\mathrm{block}}=k+O(1)$, for $k\in\{8,32,128\}$
(Lemma~\ref{lem:simprod});""")

# ===== trimmed abstract =====
ABS = r"""\begin{abstract}
\noindent
We establish the first quantitative oracle complexity theory for the convex
composite bilevel problem $\min_{x\in X^*}\omega(x)$, $X^*=\argmin\varphi$,
with independent first-order access to the smooth and nonsmooth parts of
both levels. The optimal simultaneous rate is governed by the coupling
geometry of the outer objective, in three regimes. (i)
\emph{Invertible-affine coupling} $\omega(y,z)=\Omega(z-Ay)$: a
probe-and-report method achieves $(\varepsilon_\varphi,\varepsilon_\omega)
\sim(k^{-2},k^{-2})$ for all $k\le c\min\{R^2,\sqrt{L/\tau}\}$, tight in
that window, while the Tikhonov scheme FBi-PG of Merchav--Sabach--Teboulle
(SIOPT 2026) is rate-suboptimal on this class: its error is flat
$\Omega(R^2)$ on polynomial windows. (ii) \emph{General coupling}: the
uniform outer complexity is flat, $\Theta(\min\{V,cL_\omega n,c\rho\sqrt
n\})$, tight at every round of the lock-in window between a deep-scale
singleton-pair lower bound and an explicit value-hedging algorithm, with
matching constant $V/2$ when $V$ is binding; the third entry is a slope
cap, and spreading a slope budget over dyadic scales does not beat it.
\emph{Distance-coupled} outer levels exhibit an access dichotomy: under
value-only access to the nonsmooth outer term they impose a flat
$\Theta(\rho R)$ barrier, while under proximal access they are solved
exactly in two queries. (iii) \emph{Without quadratic growth}, a
value-space modulus with instance-dependent constants always exists, but
no uniform constant and no sublinear modulus exist, and value-certified
bounds are provably one-sided. Lower bounds are proved by exact support
induction for zero-respecting algorithms and extended to arbitrary
algorithms by Givens-product rotation simulations.
\end{abstract}"""
tex = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda m: ABS, tex, flags=re.S)

open(OUT, "w", encoding="utf-8").write(tex)
print(f"OK: wrote {OUT} ({len(tex)} chars)")

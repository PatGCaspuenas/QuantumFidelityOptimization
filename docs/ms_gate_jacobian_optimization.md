# MS Gate Jacobian Optimization

## Goal
This note explains the theory and implementation behind the MS-only calibration-sequence search used for the updated sensitivity figure. The practical goal is:

- stay within closed-loop Mølmer-Sørensen subgates,
- keep the nominal readout easy to interpret with low shot counts,
- and choose a sequence whose local response to the four calibration parameters is as balanced as possible.

The four parameters are the same ones used in the sensitivity figure:

- Rabi scale $\Omega$,
- center-line detuning $\omega_{cl}$,
- sideband detuning $\delta$,
- accumulated phase error $\Delta \phi$.

## Why The Existing Figure Splits Into Two Sensitivity Classes
The starting point is the MS Hamiltonian in the form used by Gerster et al.:

$$
H(t) = -\eta \Omega(t)\left(a^\dagger e^{i\delta t} + a e^{-i\delta t}\right)
\left[S_y \cos(\phi + \Lambda(t)) + S_x \sin(\phi + \Lambda(t))\right],
$$

with

$$
\Lambda(t) = \int_0^t \omega_{cl}(t')\,dt'.
$$

Two structural consequences matter for calibration:

1. $\omega_{cl}$ enters as an accumulated phase during the gate through $\Lambda(t)$, so it tends to resemble a phase-setting error.
2. $\Omega$ and $\delta$ both control the phase-space loop geometry and the net entangling angle, so they tend to resemble each other in population-only scans.

This is why the original Innsbruck figure naturally separates into:

- a sequence that is first-order sensitive to Rabi-like errors,
- and a sequence that is first-order sensitive to phase-like errors.

The similarity is not a bug in the code; it is already implied by the Hamiltonian.

## Why Closed-Loop MS Subgates
The search is restricted to concatenations of closed-loop $MS_\phi(\theta)$ subgates rather than arbitrary pulse segments.

This choice keeps the search in a physically interpretable regime:

- each subgate is still a recognizable MS operation,
- phase-space closure is enforced at the subgate level rather than being left to the optimizer,
- the resulting winner can be described in the same language as the original two sequences,
- and the final plot remains easy to compare against the Innsbruck-style figure.

In the code, a closed-loop $MS_\phi(\theta)$ subgate is realized by scaling the intensity relative to the $\pi/2$ reference gate and mapping the spin phase $\phi$ to optical phases as

$$
(\phi_1,\phi_2) = (2\phi, 0),
$$

with an additional accumulated common offset of $\Delta \phi / 2$ applied to later pulses when the phase-error axis is scanned.

That generalizes the previously validated conventions:

- $3 \times MS_0(\pi/2)$,
- $MS_0(\pi/2)$ then $MS_{\pi/4}(\pi/2)$.

## Why The Search Allows Off-Center Readout
For a low-shot calibration probe, the easiest nominal readout is not necessarily a population close to $1$. A point near $1$ is often an extremum, so the first derivative is small and the sequence becomes locally insensitive.

An exactly balanced null,

$$
P_{gg} \approx 0.5,\qquad
P_{ee} \approx 0.5,\qquad
P_{odd} \approx 0,
$$

is still attractive because:

1. it is easy to recognize experimentally,
2. the sign of $P_{gg} - P_{ee}$ is immediately informative,
3. small miscalibrations can move the readout linearly away from the null if the sequence is chosen well.

However, the current search no longer enforces that operating point tightly. Instead, it uses a weaker low-shot criterion:

- both even-parity populations should remain non-saturated,
- the nominal odd population should stay modest,
- and once the even populations are comfortably above a readout floor, the shot-noise-whitened Jacobian is allowed to dominate the ranking.

This broadens the search beyond the immediate neighborhood of $0.5 / 0.5 / 0$ while still excluding probes that would require very large shot counts to resolve one nearly empty population bin.

## Observable Choice
The raw grouped probabilities are

$$
p = \begin{bmatrix}
P_{gg} \\
P_{ee} \\
P_{odd}
\end{bmatrix}.
$$

Because these probabilities sum to one, they are not independent. The implementation therefore works with the two independent observables

$$
r =
\begin{bmatrix}
z \\
P_{odd}
\end{bmatrix}
=
\begin{bmatrix}
P_{gg} - P_{ee} \\
P_{odd}
\end{bmatrix}.
$$

This basis is also closer to the intuitive calibration task:

- $z = P_{gg} - P_{ee}$ tracks even-parity imbalance,
- $P_{odd}$ tracks leakage into the odd subspace.

## Jacobian
For a fixed nominal sequence, define the normalized parameter vector

$$
\vartheta =
\begin{bmatrix}
\Omega / \Omega_{\mathrm{ref}} \\
\omega_{cl} / (2\pi\,\mathrm{kHz}) \\
\delta / (2\pi\,\mathrm{kHz}) \\
\Delta \phi / \pi
\end{bmatrix}.
$$

The local linear response is

$$
r(\vartheta_0 + \Delta \vartheta)
\approx
r(\vartheta_0) + J \,\Delta \vartheta,
$$

with Jacobian

$$
J_{ij} = \left.\frac{\partial r_i}{\partial \vartheta_j}\right|_{\vartheta_0}.
$$

In the implementation, $J$ is estimated by central finite differences using the same deterministic IonSim sequence-evolution path as the sensitivity figure.

## Shot-Noise-Aware Fisher Whitening
For grouped probabilities $p$, the trinomial shot-noise covariance is

$$
\Sigma_p = \mathrm{diag}(p) - p p^T.
$$

The observable vector $r = A p$ with

$$
A =
\begin{bmatrix}
1 & -1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

has covariance

$$
\Sigma_r = A \Sigma_p A^T.
$$

The code then whitens the Jacobian:

$$
J_w = L^{-1} J,
\qquad
L L^T = \Sigma_r.
$$

This converts raw slopes into shot-noise-weighted slopes, so a candidate is rewarded only if its readout changes by more than the multinomial noise floor at the nominal operating point.

## Scalar Search Score
For one sequence, the Fisher matrix

$$
F = J_w^T J_w
$$

is necessarily rank-deficient when only the final population basis is measured, because there are only two independent observables in $r$. So the search does not pretend that a single population-only probe can fully identify all four parameters.

Instead, it uses a balanced scalar score built from:

- the product of the nonzero singular values of $J_w$ ($\mathrm{info\_area}$),
- the spread of the four column norms ($\mathrm{balance\_score}$),
- a penalty for nearly parallel parameter columns ($\mathrm{max\_corr}$),
- a low-shot readout weight that saturates once both even populations are comfortably nonzero ($\mathrm{readout\_weight}$),
- and a mild penalty for longer sequences.

Conceptually, the optimizer asks:

1. does the sequence move appreciably above shot noise,
2. do all four parameters leave some visible imprint,
3. are those imprints as distinct as possible in a population-only measurement,
4. and does the nominal readout stay easy to interpret?

## Search Space
The search is deliberately narrow and interpretable rather than completely free-form.

It considers two families:

1. two-pulse family

$$
MS_0(\theta_1)\,MS_{\phi_2}(\theta_2)
$$

2. three-pulse family

$$
MS_0(\theta_{\mathrm{out}})
\,MS_{\phi_{\mathrm{mid}}}(\theta_{\mathrm{mid}})
\,MS_{-\epsilon}(\theta_{\mathrm{out}})
$$

with discrete angle and phase grids chosen to:

- include the existing A/B baselines,
- allow asymmetric pulse areas,
- allow a small final phase offset to break residual symmetries,
- and keep the runtime manageable.

Before the expensive IonSim Jacobian stage, the search uses a cheap ideal even-subspace model to discard candidates whose nominal even-parity outputs are too close to saturation.

## Relation To The Updated Code
The implementation is split across three files:

- `src/ms_sequences.jl`: shared closed-loop subgate builder and deterministic multi-pulse evolution helpers,
- `src/ms_sequence_search.jl`: candidate families, finite-difference Jacobian, covariance whitening, and scoring,
- `scripts/ms_sensitivity.jl`: plotting and optional inclusion of the searched sequence.

The search is also runnable directly through:

- `scripts/ms_sequence_search.jl`

which writes a cached summary used by the plot script so the expensive search does not need to be repeated every time the figure is regenerated.

## Search Outcome
For the current low-shot search setting:

- `260` closed-loop candidates were generated,
- `116` passed the relaxed non-saturation prefilter,
- and those `116` candidates were scored with the full IonSim finite-difference Jacobian.

The best overall sequence is now a distinct three-pulse probe:

$$
MS_0(3\pi/8)\,MS_{3\pi/8}(5\pi/8)\,MS_{-\pi/16}(3\pi/8).
$$

Its shot-noise-aware score is approximately

$$
2.32 \times 10^{-2},
$$

with nominal grouped populations

$$
P_{gg} \approx 0.722,\qquad
P_{ee} \approx 0.273,\qquad
P_{odd} \approx 4.8 \times 10^{-3}.
$$

This sequence is intentionally off-center relative to the original $0.5 / 0.5 / 0$ bias, but it still leaves both even-parity populations comfortably populated. Under the relaxed low-shot criterion, that is enough to keep the readout practical while allowing a much stronger and more balanced local Jacobian.

For comparison, the previous phase-sensitive baseline

$$
MS_0(\pi/2)\,MS_{\pi/4}(\pi/2)
$$

now scores

$$
4.62 \times 10^{-3},
$$

and the Rabi-sensitive baseline

$$
3 \times MS_0(\pi/2)
$$

still scores only

$$
6.12 \times 10^{-6}.
$$

So after relaxing the centering bias, the search does identify a genuinely better non-baseline probe, and the regenerated sensitivity figure includes it as a third sequence.

## Interpretation
There are two important conceptual limits to keep in mind:

1. A single final-basis MS-only population measurement cannot make the four-parameter Fisher matrix full rank.
2. The search therefore finds the best single-sequence compromise, not a mathematically complete standalone four-parameter calibrator.

In practice, the searched sequence is best understood as:

- either a better third probe to complement the original A/B pair,
- or evidence that the paper-inspired A/B pair is already close to optimal within the closed-loop MS-only, population-only design space.

With the relaxed low-shot criterion, the search landed in the first regime: a distinct three-pulse probe overtook the old baselines, so the updated sensitivity plot now shows all three sequences.

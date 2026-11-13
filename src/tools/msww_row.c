/* msww_row: fit one class row of the multi-wakeword decoder's [15,512] head (kernel/Dense:20) on the Dot itself.
 *
 * Port of tools/msww/train.py fit() (competing-logit logistic regression, multiple-instance choice of the word's cycles, hard-negative
 * mining), written so that it fits a Dot: the shared negative pool stays int8 (+ one float scale per cycle) and is never expanded to
 * float -- the standardisation is folded into the weights -- so peak memory is the pool file plus ~10 KB per 1000 positive cycles.
 *
 *   msww_row POOL.bin POS.bin OUT.json [decay l2 pos_frac mine_thr mine_w rounds iters]
 *
 *   POOL.bin  'MSWP' u32 n, then n records { f32 L, f32 weight, f32 scale, u8 seq, 3 pad, i8 q[512] }   x ~= q * scale
 *             L = logsumexp of the engine's other 14 logits for that cycle; weight = how many pool cycles it stands for;
 *             seq: 0 = not in time order (mined unsmoothed, p as is, like train.py's seq=False streams); 1 = first cycle of a time-ordered run
 *             (smoothed from s=0, so (1-decay)*p, as spotter.smooth); 2 = continues the run (the previous record is the cycle 80 ms before)
 *   POS.bin   'MSWQ' u32 nwin, then per window u32 ncyc, ncyc x { f32 L, u32 inside_word_mask, f32 x[512] }   (the user's own words)
 *   OUT.json  {"w":[512],"b":..,"row":{"q":[512],"s":..,"zp":..,"sums":..,"bias":..}}  the engine's int8 row (rowfit.quantize_row)
 *
 * Inputs are validated (magic, sizes, finite values, ranges); any violation is an error exit (2) rather than undefined behaviour.
 *
 * Host check:  cc -O2 -Wall -Wextra -o build/msww_row src/tools/msww_row.c -lm
 * Arm:         same flags with the NDK clang; no dependencies beyond libm.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D 512
#define LB 8                                    /* L-BFGS memory */
#define MAX_POOL (2u << 20)                     /* cycles: 2M x 513 B = 1 GB, the most a 32-bit size_t can index here */
#define MAX_WIN  (1u << 20)
#define MAX_POS  (1u << 17)                     /* positive cycles: 131072 x 2 KB = 268 MB */
#define MAX_ABS  1e6f                           /* bound on any logit / feature value; real ones are O(10) */

struct pool { uint32_t n; float *L, *wt, *sc; uint8_t *seq; int8_t *q; };
struct pos { uint32_t nwin, *off, *len, n; float *L, *x; uint32_t *mask; };

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "out of memory (%zu)\n", n); exit(2); } return p; }
static double sigmoid(double z) { if (z > 30) z = 30; if (z < -30) z = -30; return 1.0 / (1.0 + exp(-z)); }

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n ? 0 : -1; }
static int okf(float v) { return isfinite(v) && fabsf(v) <= MAX_ABS; }

static int load_pool(const char *path, struct pool *P)
{
    FILE *f = fopen(path, "rb"); char magic[4]; uint32_t n;
    if (!f) return -1;
    if (rd(f, magic, 4) || memcmp(magic, "MSWP", 4) || rd(f, &n, 4) || n == 0 || n > MAX_POOL) { fclose(f); return -1; }
    P->n = n; P->L = xmalloc(4 * (size_t)n); P->wt = xmalloc(4 * (size_t)n); P->sc = xmalloc(4 * (size_t)n); P->seq = xmalloc(n); P->q = xmalloc((size_t)n * D);
    for (uint32_t i = 0; i < n; i++) {
        uint8_t h[4];
        if (rd(f, &P->L[i], 4) || rd(f, &P->wt[i], 4) || rd(f, &P->sc[i], 4) || rd(f, h, 4) || rd(f, P->q + (size_t)i * D, D)) { fclose(f); return -1; }
        if (!okf(P->L[i]) || !okf(P->wt[i]) || P->wt[i] < 0 || !okf(P->sc[i]) || P->sc[i] <= 0 || h[0] > 2) { fclose(f); return -1; }
        P->seq[i] = h[0];
    }
    fclose(f);
    return 0;
}

static int load_pos(const char *path, struct pos *Q)
{
    FILE *f = fopen(path, "rb"); char magic[4]; uint32_t nw, cap = 0;
    if (!f) return -1;
    if (rd(f, magic, 4) || memcmp(magic, "MSWQ", 4) || rd(f, &nw, 4) || nw == 0 || nw > MAX_WIN) { fclose(f); return -1; }
    Q->nwin = nw; Q->off = xmalloc(4 * (size_t)nw); Q->len = xmalloc(4 * (size_t)nw); Q->n = 0;
    Q->L = NULL; Q->x = NULL; Q->mask = NULL;
    for (uint32_t w = 0; w < nw; w++) {
        uint32_t nc;
        if (rd(f, &nc, 4) || nc > MAX_POS || Q->n + nc > MAX_POS) { fclose(f); return -1; }
        Q->off[w] = Q->n; Q->len[w] = nc;
        if (Q->n + nc > cap) {
            cap = (Q->n + nc) * 2 > MAX_POS ? MAX_POS : (Q->n + nc) * 2;
            Q->L = realloc(Q->L, 4 * (size_t)cap); Q->mask = realloc(Q->mask, 4 * (size_t)cap); Q->x = realloc(Q->x, (size_t)cap * D * 4);
            if (!Q->L || !Q->mask || !Q->x) { fclose(f); return -1; }
        }
        for (uint32_t i = 0; i < nc; i++, Q->n++) {
            if (rd(f, &Q->L[Q->n], 4) || rd(f, &Q->mask[Q->n], 4) || rd(f, Q->x + (size_t)Q->n * D, 4 * D)) { fclose(f); return -1; }
            if (!okf(Q->L[Q->n])) { fclose(f); return -1; }
            for (int k = 0; k < D; k++) if (!okf(Q->x[(size_t)Q->n * D + k])) { fclose(f); return -1; }
        }
    }
    fclose(f);
    return 0;
}

static float dotq(const int8_t *q, const float *a)
{
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int k = 0; k < D; k += 4) { s0 += q[k] * a[k]; s1 += q[k + 1] * a[k + 1]; s2 += q[k + 2] * a[k + 2]; s3 += q[k + 3] * a[k + 3]; }
    return (s0 + s1) + (s2 + s3);
}
static float dotf(const float *x, const float *a)
{
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int k = 0; k < D; k += 4) { s0 += x[k] * a[k]; s1 += x[k + 1] * a[k + 1]; s2 += x[k + 2] * a[k + 2]; s3 += x[k + 3] * a[k + 3]; }
    return (s0 + s1) + (s2 + s3);
}

/* the problem: negatives (pool, weights wn[i]) and the chosen positive cycles (rows of xp, offsets lp, weight wp each) */
struct prob {
    const struct pool *P; const double *wn; const float *mu, *sd;
    const float *xp; const float *lp; uint32_t np; double wp; double l2;
};

/* theta = [v(512), c]; z = x.(v/sd) + c - mu.(v/sd) - L; loss = sum wt BCE + l2 |v|^2 ; returns loss, fills grad (513) */
static double objective(const struct prob *pr, const double *theta, double *grad)
{
    float a[D]; double c = theta[D], c0 = c, loss = 0, sumg = 0; static double G[D]; double Gp[D];
    for (int k = 0; k < D; k++) { a[k] = (float)(theta[k] / pr->sd[k]); c0 -= (double)pr->mu[k] * a[k]; }
    memset(G, 0, sizeof G); memset(Gp, 0, sizeof Gp);
    for (uint32_t i = 0; i < pr->P->n; i++) {
        double sc = pr->P->sc[i], z = sc * dotq(pr->P->q + (size_t)i * D, a) + c0 - pr->P->L[i], p = sigmoid(z), w = pr->wn[i];
        loss -= w * log(1 - p + 1e-12);
        double g = w * p; sumg += g;
        const int8_t *q = pr->P->q + (size_t)i * D; double gs = g * sc;
        for (int k = 0; k < D; k++) G[k] += gs * q[k];
    }
    for (uint32_t i = 0; i < pr->np; i++) {
        const float *x = pr->xp + (size_t)i * D;
        double z = dotf(x, a) + c0 - pr->lp[i], p = sigmoid(z);
        loss -= pr->wp * log(p + 1e-12);
        double g = pr->wp * (p - 1); sumg += g;
        for (int k = 0; k < D; k++) Gp[k] += g * x[k];
    }
    double reg = 0;
    for (int k = 0; k < D; k++) { reg += theta[k] * theta[k]; grad[k] = ((G[k] + Gp[k]) - (double)pr->mu[k] * sumg) / pr->sd[k] + 2 * pr->l2 * theta[k]; }
    grad[D] = sumg;
    return loss + pr->l2 * reg;
}

/* L-BFGS with backtracking (Armijo) line search */
static void lbfgs(const struct prob *pr, double *th, int iters)
{
    const int n = D + 1; double S[LB][D + 1], Y[LB][D + 1], rho[LB], g[D + 1], gn[D + 1], d[D + 1], tn[D + 1], al[LB]; int m = 0, head = 0;
    double f = objective(pr, th, g);
    for (int it = 0; it < iters; it++) {
        double gmax = 0;
        for (int k = 0; k < n; k++) if (fabs(g[k]) > gmax) gmax = fabs(g[k]);
        if (gmax < 1e-5) break;
        memcpy(d, g, sizeof d);
        for (int j = 0; j < m; j++) {                                   /* two-loop recursion */
            int i = (head - 1 - j + LB * 2) % LB; double s = 0;
            for (int k = 0; k < n; k++) s += S[i][k] * d[k];
            al[i] = rho[i] * s;
            for (int k = 0; k < n; k++) d[k] -= al[i] * Y[i][k];
        }
        if (m) {
            int i = (head - 1 + LB) % LB; double yy = 0, sy = 0;
            for (int k = 0; k < n; k++) { yy += Y[i][k] * Y[i][k]; sy += S[i][k] * Y[i][k]; }
            for (int k = 0; k < n; k++) d[k] *= sy / yy;
        } else { double gg = 0; for (int k = 0; k < n; k++) gg += g[k] * g[k]; double sc = 1.0 / sqrt(gg); for (int k = 0; k < n; k++) d[k] *= sc; }
        for (int j = m - 1; j >= 0; j--) {
            int i = (head - 1 - j + LB * 2) % LB; double b = 0;
            for (int k = 0; k < n; k++) b += Y[i][k] * d[k];
            b *= rho[i];
            for (int k = 0; k < n; k++) d[k] += S[i][k] * (al[i] - b);
        }
        double gd = 0;
        for (int k = 0; k < n; k++) { d[k] = -d[k]; gd += g[k] * d[k]; }
        if (gd >= 0) { for (int k = 0; k < n; k++) d[k] = -g[k]; gd = 0; for (int k = 0; k < n; k++) gd += g[k] * d[k]; m = 0; }
        double step = 1, fn = 0; int ok = 0;
        for (int ls = 0; ls < 20; ls++) {
            for (int k = 0; k < n; k++) tn[k] = th[k] + step * d[k];
            fn = objective(pr, tn, gn);
            if (fn <= f + 1e-4 * step * gd) { ok = 1; break; }
            step *= 0.5;
        }
        if (!ok) break;
        double sy = 0;
        for (int k = 0; k < n; k++) { S[head][k] = tn[k] - th[k]; Y[head][k] = gn[k] - g[k]; sy += S[head][k] * Y[head][k]; }
        if (sy > 1e-12) { rho[head] = 1 / sy; head = (head + 1) % LB; if (m < LB) m++; }
        double rel = (f - fn) / fmax(fmax(fabs(f), fabs(fn)), 1.0);
        memcpy(th, tn, sizeof tn); memcpy(g, gn, sizeof g); f = fn;
        if (rel < 2.2e-9) break;
    }
}

static void quantize_row(const double *w, int8_t *q, float *s, int32_t *zp, int32_t *sums)
{
    double lo = 0, hi = 0;
    for (int k = 0; k < D; k++) { if (w[k] < lo) lo = w[k]; if (w[k] > hi) hi = w[k]; }
    double sc = (hi - lo) / 255.0; if (sc < 1e-8) sc = 1e-8;
    double z = rint(-128 - lo / sc); if (z < -128) z = -128; if (z > 127) z = 127;
    int32_t sm = 0;
    for (int k = 0; k < D; k++) { double v = rint(w[k] / sc + z); if (v < -128) v = -128; if (v > 127) v = 127; q[k] = (int8_t)v; sm += (int8_t)v; }
    *s = (float)sc; *zp = (int32_t)z; *sums = sm;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: msww_row POOL.bin POS.bin OUT.json [decay l2 pos_frac mine_thr mine_w rounds iters]\n"); return 1; }
    double decay = argc > 4 ? atof(argv[4]) : 0.6, l2 = argc > 5 ? atof(argv[5]) : 3.0, pos_frac = argc > 6 ? atof(argv[6]) : 0.05;
    double mine_thr = argc > 7 ? atof(argv[7]) : 0.08, mine_w = argc > 8 ? atof(argv[8]) : 8.0; int rounds = argc > 9 ? atoi(argv[9]) : 5, iters = argc > 10 ? atoi(argv[10]) : 150;
    if (!(decay >= 0 && decay < 1) || !(l2 >= 0 && l2 < 1e6) || !(pos_frac > 0 && pos_frac <= 1) || !(mine_thr >= 0 && mine_thr <= 1) || !(mine_w >= 1 && mine_w < 1e6) || rounds < 1 || rounds > 100 || iters < 1 || iters > 100000) {
        fprintf(stderr, "bad parameter (decay in [0,1), l2>=0, pos_frac in (0,1], mine_thr in [0,1], mine_w>=1, rounds 1..100, iters 1..100000)\n"); return 1;
    }
    struct pool P; struct pos Q;
    if (load_pool(argv[1], &P)) { fprintf(stderr, "cannot read pool %s\n", argv[1]); return 2; }
    if (load_pos(argv[2], &Q)) { fprintf(stderr, "cannot read positives %s\n", argv[2]); return 2; }
    /* standardisation over the pool and the positive windows (unweighted, as the python fit) */
    static float mu[D], sd[D]; double *sum = xmalloc(8 * D), *sq = xmalloc(8 * D), *negmean = xmalloc(8 * D);
    memset(sum, 0, 8 * D); memset(sq, 0, 8 * D); memset(negmean, 0, 8 * D);
    uint32_t keepw = 0; for (uint32_t w = 0; w < Q.nwin; w++) { int any = 0; for (uint32_t i = 0; i < Q.len[w]; i++) any |= Q.mask[Q.off[w] + i] != 0; keepw += any; }
    double N = (double)P.n + 0;
    for (uint32_t i = 0; i < P.n; i++) { const int8_t *q = P.q + (size_t)i * D; double sc = P.sc[i]; for (int k = 0; k < D; k++) { double v = q[k] * sc; sum[k] += v; sq[k] += v * v; negmean[k] += v; } }
    for (uint32_t w = 0; w < Q.nwin; w++) {
        int any = 0; for (uint32_t i = 0; i < Q.len[w]; i++) any |= Q.mask[Q.off[w] + i] != 0;
        if (!any) continue;
        for (uint32_t i = 0; i < Q.len[w]; i++) { const float *x = Q.x + (size_t)(Q.off[w] + i) * D; for (int k = 0; k < D; k++) { sum[k] += x[k]; sq[k] += (double)x[k] * x[k]; } N += 1; }
    }
    for (int k = 0; k < D; k++) { double m = sum[k] / N, v = sq[k] / N - m * m; mu[k] = (float)m; sd[k] = (float)(sqrt(v > 0 ? v : 0) + 1e-3); negmean[k] /= P.n; }
    /* v0: mean of the word-cycles in z-space minus the pool mean in z-space, norm 3 */
    double theta[D + 1], v0[D]; memset(v0, 0, sizeof v0); int nw = 0;
    for (uint32_t w = 0; w < Q.nwin; w++) {
        double acc[D] = {0}; int c = 0;
        for (uint32_t i = 0; i < Q.len[w]; i++) if (Q.mask[Q.off[w] + i]) { const float *x = Q.x + (size_t)(Q.off[w] + i) * D; for (int k = 0; k < D; k++) acc[k] += (x[k] - mu[k]) / sd[k]; c++; }
        if (!c) continue;
        for (int k = 0; k < D; k++) v0[k] += acc[k] / c;
        nw++;
    }
    if (!nw) { fprintf(stderr, "no window has a cycle inside the word mask\n"); return 2; }
    double nn = 0;
    for (int k = 0; k < D; k++) { v0[k] = v0[k] / nw - (negmean[k] - mu[k]) / sd[k]; nn += v0[k] * v0[k]; }
    if (!(nn > 1e-12) || !isfinite(nn)) { fprintf(stderr, "degenerate initial direction\n"); return 2; }
    for (int k = 0; k < D; k++) theta[k] = v0[k] / sqrt(nn) * 3;
    theta[D] = 0;
    double *wn = xmalloc(8 * P.n), *score = xmalloc(8 * P.n), *sm = xmalloc(8 * P.n); uint8_t *hard = xmalloc(P.n);
    float *xp = xmalloc((size_t)(keepw * 3 + 1) * D * 4), *lp = xmalloc(4 * (keepw * 3 + 1));
    float a[D];
    for (int r = 0; r < rounds; r++) {
        double c = theta[D], c0 = c;
        for (int k = 0; k < D; k++) { a[k] = (float)(theta[k] / sd[k]); c0 -= (double)mu[k] * a[k]; }
        /* positives: in every window the best smoothed cycle inside the word mask and the two before it */
        uint32_t np = 0;
        for (uint32_t w = 0; w < Q.nwin; w++) {
            double s = 0, best = -1; int bt = -1, any = 0;
            for (uint32_t i = 0; i < Q.len[w]; i++) {
                const float *x = Q.x + (size_t)(Q.off[w] + i) * D; double p = sigmoid(dotf(x, a) + c0 - Q.L[Q.off[w] + i]);
                s = decay * s + (1 - decay) * p;
                if (Q.mask[Q.off[w] + i]) { any = 1; if (s > best) { best = s; bt = (int)i; } }
            }
            if (!any) continue;
            for (int j = bt - 2; j <= bt; j++) if (j >= 0) { memcpy(xp + (size_t)np * D, Q.x + (size_t)(Q.off[w] + j) * D, 4 * D); lp[np++] = Q.L[Q.off[w] + j]; }
        }
        /* mining on the pool: smoothed along runs, dilated by 2 either side */
        double sacc = 0;
        for (uint32_t i = 0; i < P.n; i++) {
            double z = (double)P.sc[i] * dotq(P.q + (size_t)i * D, a) + c0 - P.L[i], p = sigmoid(z);
            if (P.seq[i] == 0) { sm[i] = p; sacc = 0; continue; }       /* not time-ordered: unsmoothed, as train.py _smooth_pool */
            if (P.seq[i] == 1) sacc = 0;                                  /* first cycle of a run: smooth from s=0, as spotter.smooth */
            sacc = decay * sacc + (1 - decay) * p; sm[i] = sacc;
        }
        double wsum = 0;
        for (uint32_t i = 0; i < P.n; i++) hard[i] = 0;
        for (uint32_t i = 0; i < P.n; i++) if (sm[i] >= mine_thr) for (int j = -2; j <= 2; j++) if ((int64_t)i + j >= 0 && (uint32_t)(i + j) < P.n) hard[i + j] = 1;
        for (uint32_t i = 0; i < P.n; i++) { wn[i] = P.wt[i] * (hard[i] ? mine_w : 1.0); wsum += wn[i]; }
        uint32_t nmined = 0; for (uint32_t i = 0; i < P.n; i++) nmined += hard[i];
        if (!np) { fprintf(stderr, "no positive cycles (no window has a cycle inside the word mask)\n"); return 2; }
        struct prob pr = { &P, wn, mu, sd, xp, lp, np, pos_frac * wsum / np, l2 };
        lbfgs(&pr, theta, iters);
        double g[D + 1]; double f = objective(&pr, theta, g);
        fprintf(stderr, "round %d: %u positive cycles, %.0f of %u negatives mined, loss %.3f\n", r, np, (double)nmined, P.n, f);
        (void)score;
    }
    double w[D], b = theta[D];
    for (int k = 0; k < D; k++) { w[k] = theta[k] / sd[k]; b -= (double)mu[k] * w[k]; }
    int8_t q[D]; float s; int32_t zp, sums; quantize_row(w, q, &s, &zp, &sums);
    FILE *o = fopen(argv[3], "w");
    if (!o) return 2;
    fprintf(o, "{\"w\":["); for (int k = 0; k < D; k++) fprintf(o, "%s%.9g", k ? "," : "", w[k]);
    fprintf(o, "],\"b\":%.9g,\"row\":{\"q\":[", b); for (int k = 0; k < D; k++) fprintf(o, "%s%d", k ? "," : "", q[k]);
    fprintf(o, "],\"s\":%.9g,\"zp\":%d,\"sums\":%d,\"bias\":%.9g}}\n", s, zp, sums, b);
    fclose(o);
    return 0;
}
